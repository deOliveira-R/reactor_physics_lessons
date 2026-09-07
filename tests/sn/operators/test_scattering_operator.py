"""Foundation tests for :class:`orpheus.transport.operators.scattering.ScatteringOperator`.

Round 1.2 of Wave D of the SN reshape campaign (Issue #162). The
operator carries the math :class:`SNSolver` used to expose under
``_add_scattering_source``, ``_build_aniso_scattering`` and
``_add_n2n_source`` — thin delegators that RETIRED at #448 together with
the operator's own in-place seams ``add_iso_source`` / ``build_aniso_source``
(the eigenvalue finalize they fed became one step of the driven iteration,
which acts through ``apply``); these tests pin the lifted math at the
operator level through the bodies that survive: the channel FIELD's P0
verb ``transfer.add_p0_source`` (the in-place P0 emission), the angular
end's ℓ ≥ 1 redistribution route ``_redistribute_ordinates`` (``(1/W)·kernel``,
:eq:`pn-scatter`), and ``apply`` (their combine).

The load-bearing test is the **bit-identical extraction** suite: a
synthetic ``(psi, phi, Q)`` triple is fed through :meth:`ScatteringOperator.apply`
(and the surviving bodies above) and the explicit per-cell reference
implementations from ``test_solver_components.py``. The two paths must agree
to round-off, because the operator is a structural extraction, not a
re-derivation.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.common.xs_library import get_mixture, make_mixture
from orpheus.geometry import Mesh2D
from orpheus.numerics.operator import LinearOperator
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.numerics.quadrature import Quadrature
from tests.sn._test_helpers import material_xs_from_raw
from orpheus.transport.operators.scattering import ScatteringOperator
from orpheus.sn.solver import SNSolver
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.source_sinks import AngularSourceSink
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
from orpheus.transport.timed_full_field import TimedFullField
from orpheus.numerics.spaces.moment_head import MomentHead
from orpheus.numerics.axis import Axis, BasisKind
from orpheus.numerics.space import FunctionSpace
from orpheus.transport.operators.isotropic_transfer import (
    IsotropicN2N,
    IsotropicScattering,
)
from orpheus.transport.full_field import FullField
from tests.sn.operators._composite_operand import (
    bulk_apply,
    zero_trace_composite,
)

pytestmark = pytest.mark.foundation  # software-invariant tier


class _StubQuad:
    """Minimal :class:`~orpheus.numerics.quadrature.Quadrature`
    stand-in for synthetic tests.

    Issue #197 PR-TYPED-1 — :class:`ScatteringOperator` now consumes
    a quadrature handle (instead of separate ``N`` / ``weights`` / ``Y``
    constructor args).  Tests that exercise the per-material dispatch
    in isolation use this stub to avoid building a full quadrature.
    """

    def __init__(self, *, N: int, weights: np.ndarray) -> None:
        self.N = N
        self.weights = np.asarray(weights)

    def spherical_harmonics(self, L: int) -> np.ndarray:
        # Only called when scattering_order > 0; synthetic tests
        # below pass scattering_order=0 or =1 with anisotropy unused.
        return np.zeros((self.N, L + 1, 2 * L + 1))


def _widened_scalar_space(sn_mesh, trailing):
    r"""The PLAIN scalar end an LD iterate rides — the moment tail in the SPACE.

    CS4c step 5 (R-4): an energy binding admits exactly the bare array of its
    bound end's SHAPE (``lift.admit_array``), so a ``(ng, nx, ny, 2^d)`` LD
    iterate is the operand of a binding bound on the ``2^d``-widened scalar
    space — never of the ``(ng, nx, ny)`` one. Before the carve the shape was
    unchecked and both fed the same instance.
    """
    bulk = sn_mesh.bulk_space
    if not trailing:
        return bulk
    return FunctionSpace.of_axes(
        *bulk.axes,
        Axis("spatial_moments", tuple(trailing), kind=BasisKind.NODAL),
    )


def _energy_pair_on(solver, trailing):
    r""":math:`K_{\rm iso} = \Sigma_{s,0} + 2\Sigma_{2n}` bound on the (possibly
    widened) scalar space — the solver's own composition, re-bound."""
    space = _widened_scalar_space(solver.sn_mesh, trailing)
    return (
        IsotropicScattering.from_material_xs(solver.mat_xs, space=space)
        + IsotropicN2N.from_material_xs(solver.mat_xs, space=space)
    )


def _uniform_2d(nx, ny, delta, mat_map):
    """Helper: build a uniform Mesh2D."""
    return Mesh2D(
        edges_x=np.linspace(0, nx * delta, nx + 1),
        edges_y=np.linspace(0, ny * delta, ny + 1),
        mat_map=np.asarray(mat_map, dtype=int),
    )


@pytest.fixture
def solver_2g_p0():
    """SNSolver fixture, 2-group, P0 only (no anisotropic data)."""
    fuel = get_mixture("A", "2g")
    mod = get_mixture("B", "2g")
    materials = {2: fuel, 0: mod}

    nx, ny = 6, 4
    delta = 0.2
    mat = np.zeros((nx, ny), dtype=int)
    mat[:3, :] = 2
    mat[3:, :] = 0

    mesh = _uniform_2d(nx, ny, delta, mat)
    quad = Quadrature.lebedev(order=17)
    sn_mesh = SNMesh(mesh, quad, materials)
    solver = SNSolver(sn_mesh)
    return solver


@pytest.fixture
def solver_2g_p0_n2n():
    """SNSolver fixture, 2-group, P0 only, with a NON-ZERO (n,2n) matrix.

    #269 (Mode-10 cure): the (n,2n)-term tests below must NOT ride a
    library mixture, whose ``Sig2`` is all-zero — that nulls the exact
    term they verify (``2·Σ_2n``), so a sign/factor mutation in the
    (n,2n) path cannot move the measured value (vacuous green).  This
    fixture builds the SAME 2-group, P0-only, fuel/moderator geometry as
    :func:`solver_2g_p0` but injects an asymmetric ``Sig2`` into the fuel
    material so the term is genuinely activated AND constrained.  ``Sig2``
    is cross-group only (the asymmetry catches a ``g_from↔g_to`` swap).
    """
    from scipy.sparse import csr_matrix

    fuel = get_mixture("A", "2g")
    mod = get_mixture("B", "2g")
    # Inject a non-zero, asymmetric (n,2n) matrix into the fuel mixture.
    # SigT must absorb the (n,2n) reaction XS so the mixture stays balanced.
    sig2 = np.array([[0.0, 0.03], [0.01, 0.0]])
    fuel.Sig2 = [csr_matrix(sig2)]
    fuel.SigT = np.asarray(fuel.SigT) + sig2.sum(axis=1)
    materials = {2: fuel, 0: mod}

    nx, ny = 6, 4
    delta = 0.2
    mat = np.zeros((nx, ny), dtype=int)
    mat[:3, :] = 2
    mat[3:, :] = 0

    mesh = _uniform_2d(nx, ny, delta, mat)
    quad = Quadrature.lebedev(order=17)
    sn_mesh = SNMesh(mesh, quad, materials)
    return SNSolver(sn_mesh)


# ── Reference implementations (per-cell loops, known correct) ─────────


def _ref_iso_scatter_inplace(solver, Q, phi):
    """Reference per-cell P0 in-scatter (bit-identical to the legacy code).

    Issue #196 PR-INDEX-4: ``Q`` / ``phi`` are principled
    ``(ng, nx, ny)``.  Per-cell update reads ``(ng,)`` slices over the
    spatial pair.
    """
    out = Q.copy()
    nx, ny = solver.sn_mesh.spatial_shape
    for ix in range(nx):
        for iy in range(ny):
            mid = int(solver.sn_mesh.mat_map[ix, iy])
            out[:, ix, iy] += {mid: solver.mat_xs.sig_s_legendre(mid)[0] for mid in solver.mat_xs.materials}[mid].T @ phi[:, ix, iy]
    return out


def _ref_n2n_inplace(solver, Q, phi):
    """Reference per-cell (n,2n) source (bit-identical to the legacy code).

    Issue #196 PR-INDEX-4 — principled ``(ng, nx, ny)`` (see
    :func:`_ref_iso_scatter_inplace`).
    """
    out = Q.copy()
    nx, ny = solver.sn_mesh.spatial_shape
    for ix in range(nx):
        for iy in range(ny):
            mid = int(solver.sn_mesh.mat_map[ix, iy])
            out[:, ix, iy] += 2.0 * ({mid: solver.mat_xs.n2n_matrix(mid) for mid in solver.mat_xs.materials}[mid].T @ phi[:, ix, iy])
    return out


# ──────────────────────────────────────────────────────────────────────
# Protocol contract
# ──────────────────────────────────────────────────────────────────────


class TestProtocolCompliance:
    """ScatteringOperator must satisfy the LinearOperator Protocol."""

    def test_implements_linear_operator(self, solver_2g_p0):
        """isinstance LinearOperator (runtime-checkable Protocol)."""
        assert isinstance(solver_2g_p0.scattering_op, LinearOperator)

    def test_predicates_adjointable_not_invertible(self, solver_2g_p0):
        """``is_adjointable`` True, ``is_invertible`` False — the adjoint S† is free
        via full_transfer_kernel (#276 A2b / #118); still no useful inverse."""
        op = solver_2g_p0.scattering_op
        assert op.is_adjointable and not op.is_invertible

    def test_apply_accepts_psi_shape(self, solver_2g_p0):
        """apply(psi) must accept typed AngularFlux ``(N, ng, nx, ny)`` (D-I.2)."""
        op = solver_2g_p0.scattering_op
        N = solver_2g_p0.sn_mesh.quad.N
        psi_values = np.ones((N, solver_2g_p0.ng, *solver_2g_p0.sn_mesh.spatial_shape))
        psi = AngularFlux(values=psi_values, space=solver_2g_p0.sn_mesh.angular_bulk_space)
        # CS4c step 5: the gain is composite-bound; the bulk action rides a
        # zero-trace composite (the trace the lift itself emits back).
        out = bulk_apply(op, psi)
        assert out.values.shape == psi.values.shape


# ──────────────────────────────────────────────────────────────────────
# Bit-identical extraction (load-bearing)
# ──────────────────────────────────────────────────────────────────────


class TestBitIdenticalExtractionP0:
    """The lifted math must match the legacy reference per-cell code."""

    @pytest.mark.sentinel
    def test_p0_scattering_emission_matches_reference(self, solver_2g_p0):
        """The P0 emission verb ``transfer.add_p0_source`` = the per-cell reference.

        Issue #196 PR-INDEX-4: principled ``(ng, nx, ny)`` end-to-end.
        """
        np.random.seed(42)
        (nx, ny), ng = solver_2g_p0.sn_mesh.spatial_shape, solver_2g_p0.ng
        phi = np.random.rand(ng, nx, ny) + 0.1
        Q = np.random.rand(ng, nx, ny)

        expected = _ref_iso_scatter_inplace(solver_2g_p0, Q, phi)

        Q_actual = Q.copy()
        solver_2g_p0.scattering_op.transfer.add_p0_source(Q_actual, phi)

        np.testing.assert_allclose(Q_actual, expected, rtol=1e-13)

    def test_p0_n2n_emission_matches_reference(self, solver_2g_p0_n2n):
        """The (n,2n) emission verb = the per-cell reference (§14.1: the
        verb lives on the solver-held N2NOperator's energy binding now).

        #269 (Mode-10 cure): rides ``solver_2g_p0_n2n`` (NON-zero,
        asymmetric ``Sig2`` in the fuel) rather than the library
        ``solver_2g_p0`` (``Sig2 = 0``), so the ``2·Σ_2n`` term is
        genuinely constrained — a sign/factor mutation in
        the (n,2n) field's ``add_p0_source`` reddens this test (see the in-process
        monkeypatch proof in #269 closeout).  The reference
        :func:`_ref_n2n_inplace` is a structurally-independent per-cell
        loop (explicit ``2·Σ_2nᵀ@φ``), not the SUT's reduction.
        """
        solver = solver_2g_p0_n2n
        np.random.seed(123)
        (nx, ny), ng = solver.sn_mesh.spatial_shape, solver.ng
        phi = np.random.rand(ng, nx, ny) + 0.1
        Q = np.random.rand(ng, nx, ny)

        expected = _ref_n2n_inplace(solver, Q, phi)

        Q_actual = Q.copy()
        solver.n2n_op.isotropic_energy.transfer.add_p0_source(Q_actual, phi)

        np.testing.assert_allclose(Q_actual, expected, rtol=1e-13)

    @pytest.mark.sentinel
    @pytest.mark.parametrize("trailing", [(), (4,)], ids=["scalar", "LD-2^d=4"])
    def test_isotropic_kernel_bit_identical_to_legacy_verbs(
        self, solver_2g_p0_n2n, trailing,
    ):
        r"""#276 P2 — the production ``isotropic_kernel`` (:math:`\Sigma_{s,0} +
        2\Sigma_{2n}`, the :class:`~orpheus.numerics.operator.OperatorSum` of
        the two energy bindings the lift's :math:`\ell = 0` half
        (:meth:`~orpheus.transport.operators.angular_lift.AngularLift._isotropic_source`,
        since CS4c step 5) routes the SN forward isotropic source through) is
        **bit-identical**
        (0-ULP) to the two channel fields' ``add_p0_source`` in-place
        accumulation.

        Both reach the SAME per-material ``mat_xs`` verbs, so the K_iso routing
        introduces zero numerical change — this is the 0-ULP-inheritance anchor
        for the #276 P2 forward re-expression (the structural CORRECTNESS of the
        verbs themselves is pinned independently by
        :meth:`test_p0_scattering_emission_matches_reference` /
        :meth:`test_p0_n2n_emission_matches_reference` against the per-cell
        reference loops).  Non-zero (n,2n) fixture (#269) + scalar AND LD
        (trailing :math:`2^d` spectator axis, #240 D5b-S3).  ``-O``-safe
        (``np.testing``).
        """
        op = solver_2g_p0_n2n.scattering_op
        rng = np.random.default_rng(0)
        (nx, ny), ng = solver_2g_p0_n2n.sn_mesh.spatial_shape, solver_2g_p0_n2n.ng
        phi = rng.uniform(0.1, 1.0, size=(ng, nx, ny, *trailing))

        # Production path: the SOLVER-composed K_iso (§14.1 — the sum
        # build_within_group_system assembles from the two cached energy
        # bindings). CS4c step 5 (R-4): an energy binding admits exactly the
        # bare array of its bound end's SHAPE, so the LD row binds the pair on
        # the 2^d-WIDENED scalar space the iterate lives on — the moment tail
        # rides in the SPACE, not in an array fed to a narrower binding.
        k_iso = _energy_pair_on(solver_2g_p0_n2n, trailing)
        got = k_iso.apply(phi)

        # Reference path: zeros → the two channel FIELDS' in-place verb (the
        # field's einsum is spatial-moment-agnostic, #240 D5b-S3, so it needs
        # no re-binding — which is exactly why the widened SPACE is the honest
        # spelling of what this row was always exercising).
        ref = np.zeros_like(phi)
        op.transfer.add_p0_source(ref, phi)
        solver_2g_p0_n2n.n2n_op.isotropic_energy.transfer.add_p0_source(ref, phi)

        np.testing.assert_array_equal(
            got, ref,
            err_msg="the solver-composed K_iso (#276 P2 → §14.1) must "
            "equal the two channel fields' add_p0_source accumulation (0-ULP).",
        )

    def test_zero_flux_zero_addition(self, solver_2g_p0):
        """φ = 0 => ScatteringOperator adds zero (linearity guard)."""
        (nx, ny), ng = solver_2g_p0.sn_mesh.spatial_shape, solver_2g_p0.ng
        Q = np.ones((ng, nx, ny))
        phi = np.zeros_like(Q)
        Q_before = Q.copy()
        solver_2g_p0.scattering_op.transfer.add_p0_source(Q, phi)
        np.testing.assert_array_equal(Q, Q_before)


# ──────────────────────────────────────────────────────────────────────
# Pℓ Galerkin reconstruction
# ──────────────────────────────────────────────────────────────────────


class TestAnisotropicScatteringExtraction:
    """The ℓ ≥ 1 Galerkin reconstruction (:eq:`pn-scatter`) — the angular
    end's redistribution route ``_redistribute_ordinates`` = ``(1/W)·kernel``,
    the body the retired ``build_aniso_source`` verb wrapped (#448)."""

    @pytest.fixture
    def solver_2g_p1(self):
        """4-group with P1 anisotropic scattering data."""
        # Use 421-group library which carries P1; if not available, skip.
        try:
            mix = get_mixture("A", "4g")
        except Exception:
            pytest.skip("4g library unavailable")
        if len(mix.SigS) < 2:
            pytest.skip("No P1 data in 4g library")

        mesh = _uniform_2d(2, 2, 0.5, np.zeros((2, 2), dtype=int))
        quad = Quadrature.lebedev(order=17)
        return SNSolver(SNMesh(mesh, quad, {0: mix}), scattering_order=1)

    def test_l0_binding_is_isotropic_and_selects_no_redistribution(self, solver_2g_p0):
        """L=0 ⟹ the binding is isotropic and no ℓ ≥ 1 body is selected —
        the predicate the retired ``build_aniso_source``'s ``None`` return
        used to encode (#448: the sentinel was the predicate wearing a
        return value)."""
        op = solver_2g_p0.scattering_op
        assert op.is_isotropic  # ``_redistribution is None`` is DERIVED from this in __post_init__

    def test_isotropic_flux_zero_aniso_source(self, solver_2g_p1):
        """Isotropic ψ_n = const for every ordinate => P1+ Galerkin moments = 0."""
        op = solver_2g_p1.scattering_op
        N = solver_2g_p1.sn_mesh.quad.N
        psi_iso_values = np.ones((N, solver_2g_p1.ng, *solver_2g_p1.sn_mesh.spatial_shape))
        psi_iso = AngularFlux(values=psi_iso_values, space=solver_2g_p1.sn_mesh.angular_bulk_space)
        Q_aniso = op._redistribute_ordinates(psi_iso)
        np.testing.assert_allclose(Q_aniso.values, 0, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# apply() — the LinearOperator surface
# ──────────────────────────────────────────────────────────────────────


class TestApplySemantics:
    """apply(psi) returns the per-ordinate scattering source.

    Combines P0 in-scatter + (n,2n) (broadcast across N) + Pℓ (genuine
    per-ordinate) into a single ``(N, ng, nx, ny)`` array (principled
    storage; see :ref:`theory-sn-index-convention`).
    """

    def test_apply_isotropic_flux_p0_only(self, solver_2g_p0):
        """For P0-only solver, apply(ψ) = P0-in-scatter(φ) / W broadcast.

        §14.1: the (n,2n) term is N2NOperator's — ``S.apply`` is the
        scattering channel alone.

        R-1 Step 4 A1 — ``ScatteringOperator.apply`` returns per-ordinate
        density at the producer boundary (the ``/sum_w`` projection
        lives at the producer per Pattern 7).  Pre-A1 this returned
        iso magnitude broadcast across N; post-A1 it returns
        ``Q_iso / sum_w`` broadcast — the per-ordinate value each
        ordinate sees in the per-ordinate transport equation
        ``(Ω·∇ + Σ_t) ψ_n = Q_iso/W + …``.

        D-I.2: typed AngularFlux carrier → AngularSourceSink output.
        """
        op = solver_2g_p0.scattering_op
        N = solver_2g_p0.sn_mesh.quad.N
        (nx, ny), ng = solver_2g_p0.sn_mesh.spatial_shape, solver_2g_p0.ng

        np.random.seed(5)
        psi_values = np.random.rand(N, ng, nx, ny) + 0.1
        psi = AngularFlux(values=psi_values, space=solver_2g_p0.sn_mesh.angular_bulk_space)

        # Compute scalar flux the same way apply() does internally.
        phi = np.einsum('n,ngxy->gxy', solver_2g_p0.sn_mesh.quad.weights, psi_values)

        # Reference: compute Q_iso explicitly, then project to
        # per-ordinate via /sum_w (R-1 Step 4 A1).
        Q_iso = np.zeros((ng, nx, ny))
        op.transfer.add_p0_source(Q_iso, phi)
        sum_w = float(solver_2g_p0.sn_mesh.quad.weights.sum())
        expected = np.broadcast_to(
            (Q_iso / sum_w)[None, :, :, :], psi_values.shape,
        )

        actual = bulk_apply(op, psi)
        np.testing.assert_allclose(actual.values, expected, rtol=1e-13)

    def test_apply_zero_psi_returns_zero(self, solver_2g_p0):
        """ψ = 0 => S·ψ = 0 (linearity guard)."""
        op = solver_2g_p0.scattering_op
        N = solver_2g_p0.sn_mesh.quad.N
        psi_values = np.zeros((N, solver_2g_p0.ng, *solver_2g_p0.sn_mesh.spatial_shape))
        psi = AngularFlux(values=psi_values, space=solver_2g_p0.sn_mesh.angular_bulk_space)
        out = bulk_apply(op, psi)
        np.testing.assert_array_equal(out.values, np.zeros_like(psi_values))

    def test_apply_linearity(self, solver_2g_p0):
        """S is linear — homogeneity and DIRECT additivity.

        Since campaign 1 CS3 (flux lives in V) the textbook laws are
        directly spellable: ``S(c·ψ) = c·S(ψ)`` and
        ``S(ψ₁+ψ₂) = S(ψ₁)+S(ψ₂)``. The additivity row alone reds an
        affine S (the pre-CS3 blend spelling
        ``S(ψ₁+λ(ψ₂⊖ψ₁)) = (1−λ)S(ψ₁)+λS(ψ₂)`` could not — affine maps
        preserve affine combinations; the sharpness argument is in
        ``test_declared_law_is_linear.py``). ``op.apply`` stays on flux
        states (its domain — S guards it, rejecting a non-flux input)."""
        op = solver_2g_p0.scattering_op
        N = solver_2g_p0.sn_mesh.quad.N
        (nx, ny), ng = solver_2g_p0.sn_mesh.spatial_shape, solver_2g_p0.ng
        m = solver_2g_p0.sn_mesh

        np.random.seed(13)
        psi1 = AngularFlux(values=np.random.rand(N, ng, nx, ny) + 0.1, space=m.angular_bulk_space)
        psi2 = AngularFlux(values=np.random.rand(N, ng, nx, ny) + 0.1, space=m.angular_bulk_space)
        c = 2.5

        np.testing.assert_allclose(
            bulk_apply(op, c * psi1).values, (c * bulk_apply(op, psi1)).values,
            rtol=1e-12, atol=1e-13,
        )
        lhs = bulk_apply(op, psi1 + psi2)
        rhs = bulk_apply(op, psi1) + bulk_apply(op, psi2)
        np.testing.assert_allclose(lhs.values, rhs.values, rtol=1e-12, atol=1e-13)


# ──────────────────────────────────────────────────────────────────────
# Producer-side normalisation invariant (R-1 Step 4 A1 — Pattern 7)
# ──────────────────────────────────────────────────────────────────────


class TestProducerSideNormalisation:
    """Producer-side ``/sum_w`` invariant on the typed ``apply`` boundary.

    R-1 Step 4 A1 lifted the per-ordinate projection from the sweep
    interior to the producer boundary (Pattern 7 per
    ``coding-elegance`` SKILL.md).  This test class pins the algebraic
    identity that makes the projection ``sum_w``-independent for uniform
    input: under a uniform AngularFlux ``ψ_n = c`` (so the iso magnitude
    is ``φ = c · sum_w``), the producer's apply returns per-ordinate
    density ``Q_n = c · Σ_{g'} (Σ_{s,0}[g'→g] + 2·Σ_{2n}[g'→g])`` —
    explicitly *without* the ``sum_w`` factor.  If any future refactor
    re-introduces a sweep-internal ``/W`` (or drops the producer's
    ``/sum_w``), this test fails.
    """

    @pytest.mark.l0
    @pytest.mark.verifies("matrix-eigenvalue")
    def test_typed_apply_returns_per_ordinate_already_normalised(
        self, solver_2g_p0,
    ):
        r"""Uniform :math:`\psi_n = c` ⇒ producer-side per-ord output is
        :math:`Q_n = c \sum_{g'}(\Sigma_{s,0}[g'\to g] + 2\Sigma_{2n}[g'\to g])`.

        Algebra:

        * :math:`\phi_g = \sum_n w_n\,c = c \cdot \mathrm{sum\_w}` (iso scalar).
        * Iso magnitude :math:`Q_{\rm iso}[g] = \sum_{g'} \big(\Sigma_{s,0}[g'\to g] + 2\Sigma_{2n}[g'\to g]\big) \cdot c \cdot \mathrm{sum\_w}`.
        * Producer-side :math:`1/W`: :math:`Q_n[g] = Q_{\rm iso}[g]/\mathrm{sum\_w} = c \sum_{g'}(\Sigma_{s,0}[g'\to g] + 2\Sigma_{2n}[g'\to g])`.

        The ``sum_w`` factor cancels by construction.  This is the
        load-bearing producer-side identity that Pattern 7 introduces.
        """
        solver = solver_2g_p0
        op = solver.scattering_op
        N = solver.sn_mesh.quad.N
        (nx, ny), ng = solver.sn_mesh.spatial_shape, solver.ng

        c = 0.37
        psi_values = np.full((N, ng, nx, ny), c)
        psi = AngularFlux(values=psi_values, space=solver.sn_mesh.angular_bulk_space)

        # Reference: compute Σ_{g'}(Σ_{s,0}[g'→g] + 2·Σ_{2n}[g'→g]) at each
        # cell from the cell's material.  This is the per-ord magnitude
        # the producer must emit (sum_w-independent).
        expected = np.zeros((N, ng, nx, ny))
        for ix in range(nx):
            for iy in range(ny):
                mid = int(solver.sn_mesh.mat_map[ix, iy])
                sig_s0 = solver.mat_xs.sig_s_legendre(mid)[0]   # (ng, ng) — [g'→g]
                sig_2n = solver.mat_xs.n2n_matrix(mid)          # (ng, ng) — [g'→g]
                # Σ_{g'} (Σ_s0[g'→g] + 2·Σ_2n[g'→g]) per target group.
                # sig_s0.T @ ones gives column sums (over g') indexed by g.
                col_sum = (sig_s0.T + 2.0 * sig_2n.T) @ np.ones(ng)
                expected[:, :, ix, iy] = c * col_sum[None, :]

        actual = bulk_apply(op, psi)
        np.testing.assert_allclose(actual.values, expected, rtol=1e-13, atol=1e-13)


# ──────────────────────────────────────────────────────────────────────
# D-H.2-C1 — Composite TimedFullField invariants (volumetric scattering).
#
# Per Option β3 (Issue #208), scattering is volumetric — the output
# boundary is the implicit-zero :class:`AngularBoundaryFlux`.  Bulk follows
# the full :math:`P_\ell` Galerkin path identical to the legacy
# :class:`AngularFlux` branch.  The parity tests vs. legacy retired
# with D-H.2-C1 (the legacy class itself retires in C5; both branches
# share the same Galerkin kernel — composite-branch exercise alone is
# sufficient).
# ──────────────────────────────────────────────────────────────────────


class TestCompositeInvariants:
    """Composite :class:`TimedFullField` variant: bulk-only scattering."""

    def test_returns_timeless_full_field(self, solver_2g_p0):
        """Composite input → TIMELESS composite output (#257 S8a base arrow)."""
        from dataclasses import replace

        from orpheus.transport.fields.angular_flux import (
            AngularFlux,
        )
        from orpheus.transport.full_field import FullField
        from orpheus.transport.source_sinks import AngularSourceSink
        from orpheus.transport.timed_full_field import TimedFullField

        sn_mesh = solver_2g_p0.sn_mesh
        state = TimedFullField.zeros(interior=AngularFlux, boundary=AngularBoundaryFlux, space=sn_mesh.full_field_space)
        np.random.seed(41)
        bulk_values = np.random.rand(*state.interior.values.shape) + 0.1
        state = replace(state, interior=replace(state.interior, values=bulk_values))

        out = solver_2g_p0.scattering_op.apply(state)

        # #257 S8a — the matvec leaf is a base arrow ``FullField -> FullField``,
        # so the output is the TIMELESS FullField (history-free).
        assert isinstance(out, FullField)
        assert not isinstance(out, TimedFullField)
        assert isinstance(out.interior, AngularSourceSink)
        assert out.interior.space is sn_mesh.angular_bulk_space

    def test_implicit_zero_boundary(self, solver_2g_p0):
        """Scattering is volumetric — boundary member is all zeros."""
        from dataclasses import replace

        sn_mesh = solver_2g_p0.sn_mesh
        state = TimedFullField.zeros(interior=AngularFlux, boundary=AngularBoundaryFlux, space=sn_mesh.full_field_space)
        np.random.seed(42)
        bulk_values = np.random.rand(*state.interior.values.shape) + 0.1
        state = replace(state, interior=replace(state.interior, values=bulk_values))

        out = solver_2g_p0.scattering_op.apply(state)

        # Implicit-zero boundary (Option β3 / Wave O #208).
        np.testing.assert_array_equal(out.boundary.values, 0.0)

    def test_zero_bulk_zero_output(self, solver_2g_p0):
        """ψ = 0 ⇒ S·ψ = 0 (linearity guard at composite layer)."""
        state = TimedFullField.zeros(interior=AngularFlux, boundary=AngularBoundaryFlux, space=solver_2g_p0.sn_mesh.full_field_space)
        out = solver_2g_p0.scattering_op.apply(state)
        np.testing.assert_array_equal(out.interior.values, 0.0)
        np.testing.assert_array_equal(out.boundary.values, 0.0)

    def test_output_is_timeless_full_field(self, solver_2g_p0):
        """#257 S8a — the matvec leaf is a base arrow ``FullField -> FullField``.

        The output is the TIMELESS FullField (history-free) regardless of the
        input iterate's ``history_depth`` (was: the old convention stamped
        ``history_depth`` onto the output — re-pointed).
        """
        from orpheus.transport.full_field import FullField
        from orpheus.transport.timed_full_field import TimedFullField

        sn_mesh = solver_2g_p0.sn_mesh
        for depth in (0, 1, 2, 4):
            state = TimedFullField.zeros(interior=AngularFlux, boundary=AngularBoundaryFlux, space=sn_mesh.full_field_space, history_depth=depth)
            out = solver_2g_p0.scattering_op.apply(state)
            assert isinstance(out, FullField)
            assert not isinstance(out, TimedFullField)


# ──────────────────────────────────────────────────────────────────────
# Algebraic identities (P0 + (n,2n))
# ──────────────────────────────────────────────────────────────────────


class TestP0AlgebraicIdentities:
    """Hand-checkable cases for the P0 + (n,2n) algebra."""

    def test_p0_uniform_flux_homogeneous(self):
        """Homogeneous medium, uniform φ_g = 1: Q_iso[g] = Σ_g' Σ_s0[g'->g].

        Issue #196 PR-INDEX-4: principled ``(ng, nx, ny)`` ψ / Q.
        """
        mix = make_mixture(
            sig_t=np.array([0.5, 1.0]),
            sig_c=np.array([0.01, 0.02]),
            sig_f=np.array([0.01, 0.08]),
            nu=np.array([2.5, 2.5]),
            chi=np.array([1.0, 0.0]),
            sig_s=np.array([[0.38, 0.10], [0.00, 0.90]]),
        )
        nx, ny = 2, 2
        mesh = _uniform_2d(nx, ny, 0.5, np.zeros((nx, ny), dtype=int))
        quad = Quadrature.lebedev(order=17)
        solver = SNSolver(SNMesh(mesh, quad, {0: mix}))
        op = solver.scattering_op

        phi = np.ones((solver.ng, nx, ny))
        Q = np.zeros_like(phi)
        op.transfer.add_p0_source(Q, phi)

        # Hand-computed: Q[g] = Σ_g' σ_s0[g'->g] · φ[g'] = column-sum · 1.
        sig_s0_dense = np.array(mix.SigS[0].todense())
        # Convention: ORPHEUS ``SigS[l]`` matrix entry ``[g_from, g_to]``.
        # phi @ sig_s0 sums over g_from for each g_to.
        expected_per_cell = np.ones(solver.ng) @ sig_s0_dense
        for ix in range(nx):
            for iy in range(ny):
                np.testing.assert_allclose(Q[:, ix, iy], expected_per_cell, rtol=1e-14)

    def test_n2n_doubling_factor(self):
        """For a pure-(n,2n) mixture (Σ_s0 = 0), Q = 2·φ·Σ_2n."""
        # Build a synthetic mixture with zero P0 scatter and known sig2.
        mix = make_mixture(
            sig_t=np.array([0.5, 1.0]),
            sig_c=np.array([0.01, 0.02]),
            sig_f=np.array([0.0, 0.0]),  # no fission
            nu=np.array([0.0, 0.0]),
            chi=np.zeros(2),  # non-fissile ⇒ null spectrum (S10a __post_init__ guard)
            sig_s=np.array([[0.0, 0.0], [0.0, 0.0]]),  # zero P0
        )
        # Inject a non-zero (n,2n) matrix manually after construction.
        from scipy.sparse import csr_matrix
        sig2_test = np.array([[0.0, 0.05], [0.0, 0.0]])
        mix.Sig2 = [csr_matrix(sig2_test)]

        nx, ny = 2, 2
        mesh = _uniform_2d(nx, ny, 0.5, np.zeros((nx, ny), dtype=int))
        quad = Quadrature.lebedev(order=17)
        solver = SNSolver(SNMesh(mesh, quad, {0: mix}))
        op = solver.scattering_op

        np.random.seed(31)
        # Issue #196 PR-INDEX-4: principled (ng, nx, ny).
        phi = np.random.rand(solver.ng, nx, ny) + 0.1
        Q = np.zeros_like(phi)
        op.transfer.add_p0_source(Q, phi)
        # P0 contribution should be zero
        np.testing.assert_allclose(Q, 0, atol=1e-15)

        # (n,2n) contribution — the solver-held N2N binding's verb (§14.1)
        solver.n2n_op.isotropic_energy.transfer.add_p0_source(Q, phi)
        # Hand-computed: Q[g, ix, iy] = 2 · sum_g' phi[g', ix, iy] · sig2[g'->g]
        for ix in range(nx):
            for iy in range(ny):
                expected = 2.0 * phi[:, ix, iy] @ sig2_test
                np.testing.assert_allclose(Q[:, ix, iy], expected, rtol=1e-14)


# ──────────────────────────────────────────────────────────────────────
# Foldable / residual split — Phase G Step 3+4.a (Issue #196)
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def solver_2g_p1_n2n():
    """2G solver with non-trivial cross-group P0 AND a Pℓ≥1 channel AND
    a non-zero (n,2n) matrix. Stresses every channel of the residual."""
    # Asymmetric P0 matrix: non-trivial diagonal AND off-diagonal entries.
    p0 = np.array([[0.38, 0.10], [0.05, 0.90]])
    # Non-trivial P1 block — Pℓ≥1 is unconditionally residual.
    p1 = np.array([[0.02, 0.01], [0.00, 0.04]])
    from scipy.sparse import csr_matrix
    mix = make_mixture(
        sig_t=np.array([0.5, 1.0]),
        sig_c=np.array([0.01, 0.02]),
        sig_f=np.array([0.01, 0.08]),
        nu=np.array([2.5, 2.5]),
        chi=np.array([1.0, 0.0]),
        sig_s=p0,
    )
    # Append a P1 block manually; ORPHEUS's SigS is a list[csr_matrix]
    # indexed by Legendre order.
    mix.SigS = [csr_matrix(p0), csr_matrix(p1)]
    # Inject (n,2n) — non-zero on a cross-group entry only (the brief
    # explicitly notes diagonal sig2 entries are rare but legal).
    mix.Sig2 = [csr_matrix(np.array([[0.0, 0.03], [0.01, 0.0]]))]

    nx, ny = 3, 2
    mesh = _uniform_2d(nx, ny, 0.4, np.zeros((nx, ny), dtype=int))
    quad = Quadrature.lebedev(order=17)
    return SNSolver(SNMesh(mesh, quad, {0: mix}), scattering_order=1)


class TestFoldablePart:
    """``foldable_part()`` carries ONLY the P0 within-group diagonal."""

    def test_returns_scattering_operator_instance(self, solver_2g_p0):
        """Mechanism criterion 1 — sibling class, not a new class."""
        S = solver_2g_p0.scattering_op
        assert isinstance(S.foldable_part(), ScatteringOperator)

    def test_scattering_order_is_zero(self, solver_2g_p0):
        """Mechanism criterion 2 — no Pℓ structure in foldable."""
        S = solver_2g_p0.scattering_op
        assert S.foldable_part().legendre_order == 0

    def test_faces_are_order_zero(self, solver_2g_p0):
        """Mechanism criterion 3 — the ℓ=0 sibling's faces are minted at
        order 0 (the retired ``Y is None`` claim, re-spelled on the
        rebound surface: the harmonics live on the faces' interned
        frame, and the sibling's frame is the L=0 mint)."""
        S = solver_2g_p0.scattering_op
        assert S.foldable_part().frame.basis.L == 0

    def test_faces_are_order_zero_even_for_p1_source(self, solver_2g_p1_n2n):
        """Even when S carries P1+ data, the foldable sibling's faces are
        the L=0 mint (re-minted from the SAME interned hub chain)."""
        S = solver_2g_p1_n2n.scattering_op
        assert S.foldable_part().frame.basis.L == 0

    def test_sig_s_is_diagonal_only(self, solver_2g_p1_n2n):
        """Mechanism criterion 4a — sig_s[mid][0] is diagonal-only."""
        S = solver_2g_p1_n2n.scattering_op
        S_fold = S.foldable_part()
        for mid in S.transfer.per_material:
            mat = S_fold.transfer.per_material[mid].moments[0]
            expected = np.diag(np.diag(S.transfer.per_material[mid].moments[0]))
            np.testing.assert_array_equal(mat, expected)
            # Off-diagonal is literally zero, not just small.
            off_diag = mat - np.diag(np.diag(mat))
            assert np.all(off_diag == 0.0)

    def test_sig_s0_matches_sig_s_l0(self, solver_2g_p1_n2n):
        """Mechanism criterion 4b — sig_s0 == sig_s[mid][0]."""
        S = solver_2g_p1_n2n.scattering_op
        S_fold = S.foldable_part()
        for mid in S.transfer.per_material:
            np.testing.assert_array_equal(
                S_fold.transfer.per_material[mid].p0, S_fold.transfer.per_material[mid].moments[0]
            )

    def test_sig_s_has_length_one(self, solver_2g_p1_n2n):
        """Mechanism criterion 4c — no Pℓ≥1 entries in foldable."""
        S = solver_2g_p1_n2n.scattering_op
        S_fold = S.foldable_part()
        for mid in S.transfer.per_material:
            assert len(S_fold.transfer.per_material[mid].moments) == 1

    # (Mechanism criterion 4d — "foldable's (n,2n) is zero" — DISSOLVED
    # with the §14.1 extraction: S carries no (n,2n) channel, so the
    # foldable/residual split cannot touch it BY CONSTRUCTION; the
    # channel lives on N2NOperator, outside the split entirely.)

    def test_does_not_mutate_parent_sig_s(self, solver_2g_p1_n2n):
        """Anti-rec 4 — split returns new arrays; parent unchanged."""
        S = solver_2g_p1_n2n.scattering_op
        # Snapshot every parent array (frozen kernels — belt+braces).
        before = {
            mid: [m.copy() for m in S.transfer.per_material[mid].moments]
            for mid in S.transfer.per_material
        }
        _ = S.foldable_part()
        # Parent unchanged.
        for mid in S.transfer.per_material:
            for l, m in enumerate(S.transfer.per_material[mid].moments):
                np.testing.assert_array_equal(m, before[mid][l])


class TestResidualPart:
    """``residual_part()`` carries everything but P0 within-group diagonal."""

    def test_returns_scattering_operator_instance(self, solver_2g_p0):
        """Mechanism criterion 1 — sibling class."""
        S = solver_2g_p0.scattering_op
        assert isinstance(S.residual_part(), ScatteringOperator)

    def test_sig_s_l0_diagonal_zeroed(self, solver_2g_p1_n2n):
        """Mechanism criterion 5a — cross-group only on P0."""
        S = solver_2g_p1_n2n.scattering_op
        S_res = S.residual_part()
        for mid in S.transfer.per_material:
            expected = S.transfer.per_material[mid].moments[0] - np.diag(np.diag(S.transfer.per_material[mid].moments[0]))
            np.testing.assert_array_equal(S_res.transfer.per_material[mid].moments[0], expected)
            # The diagonal IS zero, not just close.
            diag = np.diag(S_res.transfer.per_material[mid].moments[0])
            assert np.all(diag == 0.0)

    def test_sig_s0_matches_diagonal_zeroed(self, solver_2g_p1_n2n):
        """Mechanism criterion 5b — sig_s0 alias of sig_s[mid][0]."""
        S = solver_2g_p1_n2n.scattering_op
        S_res = S.residual_part()
        for mid in S.transfer.per_material:
            np.testing.assert_array_equal(
                S_res.transfer.per_material[mid].p0, S_res.transfer.per_material[mid].moments[0]
            )

    def test_pl_ge_1_carried_verbatim(self, solver_2g_p1_n2n):
        """Mechanism criterion 5c — Pℓ≥1 blocks unchanged."""
        S = solver_2g_p1_n2n.scattering_op
        assert S.legendre_order >= 1, "fixture must carry P1+ data"
        S_res = S.residual_part()
        for mid in S.transfer.per_material:
            for l in range(1, S.legendre_order + 1):
                np.testing.assert_array_equal(
                    S_res.transfer.per_material[mid].moments[l], S.transfer.per_material[mid].moments[l]
                )

    # (Mechanism criterion 5d — "(n,2n) unconditionally residual" —
    # DISSOLVED with the §14.1 extraction: S carries no (n,2n) channel,
    # so the residual cannot carry or drop it BY CONSTRUCTION.)

    def test_scattering_order_preserved(self, solver_2g_p1_n2n):
        """Mechanism criterion 5e — Pℓ structure preserved."""
        S = solver_2g_p1_n2n.scattering_op
        assert S.residual_part().legendre_order == S.legendre_order

    def test_residual_shares_the_interned_frame(self, solver_2g_p1_n2n):
        """Mechanism criterion 5f — precomputed harmonics reusable: the
        residual sibling keeps the SAME order, so its re-minted faces
        land on the SAME hub-interned frame OBJECT (one table, shared —
        strictly stronger than the retired ``Y is Y`` claim)."""
        S = solver_2g_p1_n2n.scattering_op
        S_res = S.residual_part()
        assert S_res.frame is S.frame

    def test_residual_frame_is_order_zero_for_p0_solver(self, solver_2g_p0):
        """If S has no harmonics (L=0), the residual's frame is the L=0
        mint too (the retired ``Y is None`` claim, on the faces)."""
        S = solver_2g_p0.scattering_op
        assert S.residual_part().frame.basis.L == 0

    def test_does_not_mutate_parent_sig_s(self, solver_2g_p1_n2n):
        """Anti-rec 4 — split returns new arrays; parent unchanged."""
        S = solver_2g_p1_n2n.scattering_op
        before = {
            mid: [m.copy() for m in S.transfer.per_material[mid].moments]
            for mid in S.transfer.per_material
        }
        _ = S.residual_part()
        for mid in S.transfer.per_material:
            for l, m in enumerate(S.transfer.per_material[mid].moments):
                np.testing.assert_array_equal(m, before[mid][l])


class TestFoldableSigma:
    """``foldable_sigma()`` returns the per-material (ng,) σ_{s,0}^{g→g}."""

    def test_returns_dict_of_ndarrays(self, solver_2g_p1_n2n):
        """Mechanism criterion 6a — dict[int, ndarray]."""
        S = solver_2g_p1_n2n.scattering_op
        result = S.foldable_sigma()
        assert isinstance(result, dict)
        for mid, arr in result.items():
            assert isinstance(mid, int)
            assert isinstance(arr, np.ndarray)

    def test_shape_is_ng(self, solver_2g_p1_n2n):
        """Mechanism criterion 6b — each value is (ng,)."""
        S = solver_2g_p1_n2n.scattering_op
        result = S.foldable_sigma()
        for arr in result.values():
            assert arr.shape == (solver_2g_p1_n2n.ng,)

    def test_values_are_diagonal_of_sig_s0(self, solver_2g_p1_n2n):
        """Mechanism criterion 6c — equals np.diag(sig_s[mid][0])."""
        S = solver_2g_p1_n2n.scattering_op
        result = S.foldable_sigma()
        for mid, arr in result.items():
            np.testing.assert_array_equal(arr, np.diag(S.transfer.per_material[mid].moments[0]))

    def test_returned_arrays_are_copies(self, solver_2g_p1_n2n):
        """Mutating the returned dict's values must not affect ``self``."""
        S = solver_2g_p1_n2n.scattering_op
        result = S.foldable_sigma()
        # Snapshot parent diagonal.
        before = {mid: np.diag(S.transfer.per_material[mid].moments[0]).copy() for mid in S.transfer.per_material}
        # Mutate the returned arrays.
        for arr in result.values():
            arr[:] = -999.0
        # Parent unchanged.
        for mid in S.transfer.per_material:
            np.testing.assert_array_equal(
                np.diag(S.transfer.per_material[mid].moments[0]), before[mid]
            )


class TestAlgebraicIdentity:
    """The load-bearing contract:
    ``S.apply(ψ) ≈ S.foldable_part().apply(ψ) + S.residual_part().apply(ψ)``
    at ``rtol=1e-14`` (FP-non-associativity precision).

    Covers P0-only, Pℓ≥1, non-zero (n,2n), and cross-group + diagonal
    coupling — the four cases enumerated in the brief's criterion 7.
    """

    def _check_identity(self, op, psi):
        """``psi`` is a typed :class:`AngularFlux`; ``apply`` returns
        :class:`AngularSourceSink`.  Compare via ``.values``."""
        full = bulk_apply(op, psi)
        split_sum = (
            bulk_apply(op.foldable_part(), psi) + bulk_apply(op.residual_part(), psi)
        )
        np.testing.assert_allclose(full.values, split_sum.values, rtol=1e-14, atol=1e-15)

    def test_identity_p0_only_random_psi(self, solver_2g_p0):
        """Case 1 — scattering_order == 0 only (no Pℓ).

        D-I.2: typed AngularFlux carrier.
        """
        op = solver_2g_p0.scattering_op
        assert op.legendre_order == 0
        N = solver_2g_p0.sn_mesh.quad.N
        np.random.seed(42)
        psi_values = np.random.rand(N, solver_2g_p0.ng, *solver_2g_p0.sn_mesh.spatial_shape) + 0.1
        psi = AngularFlux(values=psi_values, space=solver_2g_p0.sn_mesh.angular_bulk_space)
        self._check_identity(op, psi)

    def test_identity_p0_only_uniform_psi(self, solver_2g_p0):
        """Case 1b — uniform ψ probes the diagonal isolation path."""
        op = solver_2g_p0.scattering_op
        N = solver_2g_p0.sn_mesh.quad.N
        psi_values = np.ones((N, solver_2g_p0.ng, *solver_2g_p0.sn_mesh.spatial_shape))
        psi = AngularFlux(values=psi_values, space=solver_2g_p0.sn_mesh.angular_bulk_space)
        self._check_identity(op, psi)

    def test_identity_with_pl_ge_1(self, solver_2g_p1_n2n):
        """Case 2 — scattering_order >= 1 (with non-zero P1 block)."""
        op = solver_2g_p1_n2n.scattering_op
        assert op.legendre_order >= 1
        N = solver_2g_p1_n2n.sn_mesh.quad.N
        np.random.seed(101)
        psi_values = np.random.rand(N, solver_2g_p1_n2n.ng, *solver_2g_p1_n2n.sn_mesh.spatial_shape) + 0.1
        psi = AngularFlux(values=psi_values, space=solver_2g_p1_n2n.sn_mesh.angular_bulk_space)
        self._check_identity(op, psi)

    def test_identity_with_nonzero_n2n(self, solver_2g_p1_n2n):
        """Case 3 — non-zero (n,2n) coupling."""
        op = solver_2g_p1_n2n.scattering_op
        # Fixture explicitly sets (n,2n) cross-group entries — read off
        # the solver-held N2N field (§14.1: S carries no (n,2n) channel;
        # the identity below is about S alone, with n2n live in the
        # WORLD as the extraction demands).
        n2n_field = solver_2g_p1_n2n.n2n_op.isotropic_energy.transfer
        any_nonzero_n2n = any(
            np.any(k.p0 != 0.0) for k in n2n_field.per_material.values()
        )
        assert any_nonzero_n2n, "fixture must carry non-zero (n,2n)"
        N = solver_2g_p1_n2n.sn_mesh.quad.N
        np.random.seed(202)
        psi_values = np.random.rand(N, solver_2g_p1_n2n.ng, *solver_2g_p1_n2n.sn_mesh.spatial_shape) + 0.1
        psi = AngularFlux(values=psi_values, space=solver_2g_p1_n2n.sn_mesh.angular_bulk_space)
        self._check_identity(op, psi)

    def test_identity_multigroup_cross_group_plus_diagonal(self, solver_2g_p1_n2n):
        """Case 4 — non-trivial cross-group P0 + diagonal coupling."""
        op = solver_2g_p1_n2n.scattering_op
        # Fixture's P0 matrix has both diagonal AND off-diagonal entries.
        for mid in op.transfer.per_material:
            p0 = op.transfer.per_material[mid].moments[0]
            diag = np.diag(p0)
            off = p0 - np.diag(diag)
            assert np.any(diag != 0.0)
            assert np.any(off != 0.0)
        N = solver_2g_p1_n2n.sn_mesh.quad.N
        np.random.seed(303)
        psi_values = np.random.rand(N, solver_2g_p1_n2n.ng, *solver_2g_p1_n2n.sn_mesh.spatial_shape) + 0.1
        psi = AngularFlux(values=psi_values, space=solver_2g_p1_n2n.sn_mesh.angular_bulk_space)
        self._check_identity(op, psi)

    def test_residual_zero_when_p0_diagonal_only_no_n2n(self):
        """Pure-diagonal P0 with no (n,2n) and no Pℓ≥1 ⇒ residual.apply(ψ)=0
        and full == foldable.apply(ψ) by construction."""
        from scipy.sparse import csr_matrix
        # Strictly diagonal P0, zero (n,2n).
        mix = make_mixture(
            sig_t=np.array([0.5, 1.0]),
            sig_c=np.array([0.01, 0.02]),
            sig_f=np.array([0.0, 0.0]),
            nu=np.array([0.0, 0.0]),
            chi=np.zeros(2),  # non-fissile ⇒ null spectrum (S10a __post_init__ guard)
            sig_s=np.diag([0.3, 0.8]),
        )
        mix.Sig2 = [csr_matrix(np.zeros((2, 2)))]
        nx, ny = 2, 2
        mesh = _uniform_2d(nx, ny, 0.5, np.zeros((nx, ny), dtype=int))
        quad = Quadrature.lebedev(order=17)
        solver = SNSolver(SNMesh(mesh, quad, {0: mix}))
        op = solver.scattering_op

        N = solver.sn_mesh.quad.N
        np.random.seed(404)
        # D-I.2: typed AngularFlux carrier.
        psi_values = np.random.rand(N, solver.ng, nx, ny) + 0.1
        psi = AngularFlux(values=psi_values, space=solver.sn_mesh.angular_bulk_space)
        full = bulk_apply(op, psi)
        residual_part = bulk_apply(op.residual_part(), psi)
        np.testing.assert_allclose(residual_part.values, 0.0, atol=1e-15)
        # And full ≡ foldable up to FP-non-associativity.
        foldable_part = bulk_apply(op.foldable_part(), psi)
        np.testing.assert_allclose(full.values, foldable_part.values, rtol=1e-14, atol=1e-15)


class TestPurity:
    """``foldable_part()`` / ``residual_part()`` are pure functions —
    calling twice returns instances with equal per-material arrays
    (mechanism criterion 8)."""

    def test_foldable_part_pure(self, solver_2g_p1_n2n):
        S = solver_2g_p1_n2n.scattering_op
        a, b = S.foldable_part(), S.foldable_part()
        assert a.legendre_order == b.legendre_order == 0
        for mid in S.transfer.per_material:
            np.testing.assert_array_equal(a.transfer.per_material[mid].moments[0], b.transfer.per_material[mid].moments[0])
            np.testing.assert_array_equal(a.transfer.per_material[mid].p0, b.transfer.per_material[mid].p0)

    def test_residual_part_pure(self, solver_2g_p1_n2n):
        S = solver_2g_p1_n2n.scattering_op
        a, b = S.residual_part(), S.residual_part()
        assert a.legendre_order == b.legendre_order
        for mid in S.transfer.per_material:
            for l in range(S.legendre_order + 1):
                np.testing.assert_array_equal(
                    a.transfer.per_material[mid].moments[l], b.transfer.per_material[mid].moments[l]
                )

    def test_foldable_sigma_pure(self, solver_2g_p1_n2n):
        S = solver_2g_p1_n2n.scattering_op
        a, b = S.foldable_sigma(), S.foldable_sigma()
        assert set(a.keys()) == set(b.keys())
        for mid in a:
            np.testing.assert_array_equal(a[mid], b[mid])


# ──────────────────────────────────────────────────────────────────────
# is_foldable_into_sigma_r — Phase G Step 3+4.b.i (Issue #196)
# ──────────────────────────────────────────────────────────────────────


def _synthetic_p0(self_base, p0, extra_moments=()):
    """A synthetic ScatteringOperator carrying exactly the given Legendre
    stack, isolated from any fixture XS data (CS4c step 3 spelling: the
    predicate under test reads ONLY the kernel field, so the sibling is
    ``dataclasses.replace`` over a real solver's operator — new datum,
    same faces/ends, every admission re-run)."""
    import dataclasses

    from orpheus.transport.kernels import TransferKernel
    from orpheus.transport.material_field import TransferMaterialField

    from tests.sn._test_helpers import material_xs_from_raw

    ng = p0.shape[0]
    mat_xs = material_xs_from_raw(
        sig_s={0: [p0, *extra_moments]},
        cells_by_mat={0: (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))},
        ng=ng, nx=2, ny=2,
    )
    mesh = Mesh2D(
        edges_x=np.linspace(0.0, 1.0, 3),
        edges_y=np.linspace(0.0, 1.0, 3),
        mat_map=np.zeros((2, 2), dtype=int),
    )
    sn = SNMesh(mesh, Quadrature.lebedev(order=17), mat_xs.materials)
    return ScatteringOperator.from_solver_data(
        mat_xs=mat_xs,
        scattering_order=len(extra_moments),
        space=sn.full_field_space,
    )


class TestIsFoldableIntoSigmaR:
    """``S.is_foldable_into_sigma_r()`` returns True iff S carries only
    diagonal P0 + zero sig2.

    Consumed by substep 3+4.b.ii's ``OperatorSum.solve`` fusion hook
    to detect "this S is the foldable_part — fuse into σ_r and route
    to the within-group sweep". A STRUCTURAL predicate on the
    operator's data, not an identity claim about its action.
    """

    def test_full_scattering_returns_false(self, solver_2g_p1_n2n):
        """Full S with non-zero off-diagonal P0 + non-zero P1 + non-zero
        sig2 → NOT foldable."""
        S = solver_2g_p1_n2n.scattering_op
        # Sanity: the fixture's S has all three non-foldable channels.
        assert S.legendre_order >= 1
        assert S.is_foldable_into_sigma_r() is False

    def test_foldable_part_roundtrip_is_true(self, solver_2g_p1_n2n):
        """``S.foldable_part().is_foldable_into_sigma_r() == True``.

        The load-bearing round-trip: the operator constructed by
        ``foldable_part()`` IS, by definition, the foldable part of
        itself.
        """
        S = solver_2g_p1_n2n.scattering_op
        foldable = S.foldable_part()
        assert foldable.is_foldable_into_sigma_r() is True

    def test_residual_part_returns_false(self, solver_2g_p1_n2n):
        """``S.residual_part().is_foldable_into_sigma_r() == False``.

        The residual carries the cross-group off-diagonal P0
        unconditionally (every multi-group system has at least one
        cross-group entry) — so the diagonal-only check fails.
        """
        S = solver_2g_p1_n2n.scattering_op
        residual = S.residual_part()
        assert residual.is_foldable_into_sigma_r() is False

    def test_p0_only_diagonal_returns_true(self):
        """P0-only ScatteringOperator with diagonal sig_s + zero sig2 →
        True (positive control).

        Build a synthetic ScatteringOperator directly (bypassing
        SNSolver) to isolate the predicate from any fixture setup.
        """
        p0_diag = np.diag([0.38, 0.90])
        S = _synthetic_p0(self_base=None, p0=p0_diag)
        assert S.is_foldable_into_sigma_r() is True

    def test_p0_with_off_diagonal_returns_false(self):
        """scattering_order=0 with non-diagonal P0 → False.

        Off-diagonal P0 is cross-group scattering — couples distinct
        energy groups and cannot collapse into a per-cell scalar.
        """
        # Non-diagonal P0 — non-zero off-diagonal entry.
        p0 = np.array([[0.38, 0.10], [0.00, 0.90]])
        S = _synthetic_p0(self_base=None, p0=p0)
        assert S.is_foldable_into_sigma_r() is False

    # (The "diagonal P0 + non-zero sig2 → False" row DISSOLVED with the
    # §14.1 extraction: S carries no (n,2n) channel, so the predicate
    # cannot be defeated by one — the (n,2n)-never-folds physics now
    # lives in the STRUCTURE: N2NOperator is outside the fold entirely.)

    def test_scattering_order_ge_1_returns_false_even_with_diagonal_p0(
        self,
    ):
        """scattering_order >= 1 → False even if P0 is diagonal.

        Pℓ ≥ 1 is direction-dependent (Y_ℓ^m(Ω_n)) — unconditionally
        residual. The presence of ANY Pℓ ≥ 1 channel disqualifies the
        operator from foldability.
        """
        p0_diag = np.diag([0.38, 0.90])
        p1 = np.array([[0.02, 0.00], [0.00, 0.04]])
        S = _synthetic_p0(self_base=None, p0=p0_diag, extra_moments=(p1,))
        assert S.is_foldable_into_sigma_r() is False


# ──────────────────────────────────────────────────────────────────────
# Wave T step T.3 — per-ℓ kernel structure tests (substep T.3b).
#
# T.3 design context (test-architect spec §6 Q6, user resolution):
# the §15.2 form `Σ_ℓ (Σ_{s,ℓ} ⊗ A_ℓ ⊗ G_ℓ)` does NOT satisfy the
# disjoint-axes TensorProductOperator contract because the per-material
# per-ℓ einsum couples group + spatial axes via cells_by_material
# indexing.  Math-honest fallback: kernel is an OperatorSum of
# per-ℓ summands (each a custom LinearOperator, NOT a TP).  T.3 is
# therefore NOT the first SOTP production consumer — T.4 (streaming)
# inherits that role.
# ──────────────────────────────────────────────────────────────────────


class TestAnisoMomentSourcePath:
    """Anisotropic in-scatter via the moment→source map ``R·Λ_{ℓ≥1}``.

    Phase 5a (angular-windowing) retired the per-ℓ
    ``_PerLegendreOrderScattering`` kernel — which recomputed ``M`` for
    every Legendre order — in favour of the shared ``R·Λ`` reconstruction.
    The two aniso paths are:

    * the ANGULAR binding — the full-angular path, ``(1/W)·kernel`` where
      ``kernel = frame.conjugate(Λ) = R∘Λ∘M`` (``_redistribute_ordinates`` is
      its per-ordinate spelling — the body the angular end selects);
    * the MOMENT-DOMAIN sibling ``S.on_moment_domain()`` — the windowed SI
      driver's binding, whose operand IS ``φ = Mψ`` (the 2-D Cartesian
      angular-windowing iterate), so ``M`` is already done; its body is the
      EXPLICIT typed grid path ``Λ : HarmonicMomentFlux →
      HarmonicMomentSourceSink`` then the minted ``source_reconstruction``
      face (``HarmonicMomentSourceSink → AngularSourceSink``) — the
      role-changing edge materialised as a typed carrier.

    ⛔ Until CS4c step 5 the second bullet was an ARM of the first operator,
    reached by handing the moment iterate to the angular-bound instance
    (`[M]` 143 such feeds per windowed solve). It is now a separate BOUND
    operator whose body is selected at construction, and the ndarray
    ``R∘Λ`` oracle it used to be compared against
    (``_aniso_source_from_moment_values``) is RETIRED — the crosscheck moved
    up a tier to the two operators' own actions
    (``test_scattering_kernel_crosscheck.py``).
    ``_redistribute_ordinates``'s numerical correctness is pinned by the
    pre-T.3 bit-identical snapshot below (a structurally-independent
    reference captured BEFORE the kernel ever existed).  This class adds
    the load-bearing Phase-5a guard: the moment ``apply`` arm reproduces
    the full-angular arm bit-for-bit, plus a sentinel that the windowed arm
    actually executes the typed minted reconstruction face.
    """

    @pytest.fixture
    def op_p1(self, solver_2g_p1_n2n):
        """ScatteringOperator with P1 aniso (asymmetric SigS + n2n)."""
        return solver_2g_p1_n2n.scattering_op

    @pytest.fixture
    def op_p0(self, solver_2g_p0):
        """ScatteringOperator with scattering_order=0 (P0 only)."""
        return solver_2g_p0.scattering_op

    def test_moment_apply_arm_bit_identical_to_angular_arm(
        self, op_p1, solver_2g_p1_n2n,
    ):
        r"""**G5.3b (S)** — the MOMENT-domain sibling reproduces the ANGULAR
        binding BIT-FOR-BIT when the moments are :math:`\phi = M\psi`.

        This is the angular-windowing carve's correctness core — windowing
        the within-group SI iterate from a full per-ordinate
        :class:`AngularFlux` down to :class:`HarmonicMomentFlux` moments
        loses NO scattering-source information (``S`` is a pure function of
        the moments). The P1 asymmetric-``SigS`` fixture activates the ℓ=0
        (iso + cross-group) AND ℓ≥1 (aniso) paths, so a windowing that
        dropped a moment, swapped an index, or drifted a convention would
        fail here (`vv`: ≥2 groups, anisotropic, asymmetric scatter — Modes
        2/6 + the dropped-moment trap).

        ⛔ RE-KEYED (CS4c step 5). The moment iterate is no longer handed to
        the ANGULAR-bound operator (the shipped non-endomorphism, `[M]` 143
        such feeds per windowed solve): it goes to the SIBLING bound on the
        moment end, whose body — :math:`\Lambda` then the minted
        source-reconstruction face, with :math:`M` skipped — is selected at
        construction. So the two sides are now two BOUND OPERATORS, and
        neither calls the other (``coding-standards`` rewire-demotion check).

        `[M]` 2026-09-04, this fixture, 200 seeds: **200/200
        ``array_equal``, max |Δ| = 0.0** — bit-identity is a property of the
        FIXTURE, not of one draw (`vv` anti-#31).
        """
        op = op_p1
        psi = self._reproduce_psi(solver_2g_p1_n2n, seed=7)

        # The full-angular binding.
        src_angular = bulk_apply(op, psi)

        # The windowed path: project φ = Mψ through the operator's own
        # minted analysis FACE (typed), then the moment-domain sibling.
        moments = op.flux_analysis.apply(psi)
        op_w = op.on_moment_domain()
        src_moments = op_w.apply(
            zero_trace_composite(moments, op_w.domain.trace_space),
        ).interior

        np.testing.assert_array_equal(src_moments.values, src_angular.values)
        # Non-degeneracy: ℓ≥1 genuinely carries signal (else the moment
        # sibling collapses to the P0-only body and the guard is vacuous).
        assert op.legendre_order >= 1
        assert np.any(np.asarray(moments.values)[1:] != 0.0)

    def test_windowed_arm_executes_typed_role_changing_edge(
        self, op_p1, solver_2g_p1_n2n, monkeypatch,
    ):
        """Mode-11 sentinel (vv-principles): the MOMENT-DOMAIN SIBLING
        actually executes the EXPLICIT typed grid path — ``Λ`` constructs a
        :class:`HarmonicMomentSourceSink` (the role-changing edge) which the
        minted source-reconstruction FACE then synthesises. The Phase-5a value guard above proves
        the NUMBERS are right but cannot tell whether the rewired typed line
        ran or a bypass produced the same value; this counter-spy proves it.

        Spy point (re-keyed at CS4b S4, again at F-1 — the frame's carrier
        verb retired into the MINTED face): the minted
        ``HarmonicReconstructionOperator.apply`` is the seam that CONSUMES
        Λ's product — spying what it receives pins both halves at once (Λ
        emitted the typed HarmonicMomentSourceSink, and the face consumed
        exactly it). A bypass to the ndarray ``reconstruct_after`` reference
        would never enter this seam.
        """
        from orpheus.transport.frames import HarmonicReconstructionOperator
        from orpheus.transport.source_sinks import HarmonicMomentSourceSink

        calls = {"n": 0}
        original = HarmonicReconstructionOperator.apply

        def spying(self, moment):
            if isinstance(moment, HarmonicMomentSourceSink):
                calls["n"] += 1
            return original(self, moment)

        monkeypatch.setattr(HarmonicReconstructionOperator, "apply", spying)

        op = op_p1
        psi = self._reproduce_psi(solver_2g_p1_n2n, seed=7)
        moments = op.flux_analysis.apply(psi)
        op_w = op.on_moment_domain()
        _ = op_w.apply(zero_trace_composite(moments, op_w.domain.trace_space))

        assert op.legendre_order >= 1
        assert calls["n"] >= 1, (
            "the moment-domain sibling did not construct a "
            "HarmonicMomentSourceSink — the explicit typed role-changing "
            "edge (Λ: flux → source) was bypassed."
        )

        # NEGATIVE control (`vv` #19): the ANGULAR binding's body is the
        # fused frame conjugation and must NOT enter this seam — so a green
        # reading above genuinely discriminates the moment end's body.
        calls["n"] = 0
        _ = bulk_apply(op, psi)
        assert calls["n"] == 0, (
            "the ANGULAR binding entered the typed reconstruction face — the "
            "two ends are supposed to run DIFFERENT bodies, so this sentinel "
            "would not discriminate them."
        )

    def _load_snapshot(self):
        from tests.sn._test_helpers import SN_TESTS_ROOT

        snapshot_path = (
            SN_TESTS_ROOT / "_fixtures" / "wave_t_t3" / "pre_t3_snapshots.npz"
        )
        return np.load(snapshot_path)

    def _reproduce_psi(self, solver, seed: int):
        """Mirror `_capture_pre_t3_snapshots.py::_make_psi`."""
        from orpheus.transport.fields.angular_flux import AngularFlux

        N = solver.quad.N
        ng = solver.ng
        nx, ny = solver.sn_mesh.spatial_shape
        rng = np.random.default_rng(seed)
        psi_values = rng.uniform(0.05, 1.0, size=(N, ng, nx, ny))
        return AngularFlux(values=psi_values, space=solver.sn_mesh.angular_bulk_space)

    def _reproduce_phi(self, solver, seed: int):
        """Mirror `_capture_pre_t3_snapshots.py::_make_phi`."""
        from orpheus.transport.fields.scalar_flux import ScalarFlux

        ng = solver.ng
        nx, ny = solver.sn_mesh.spatial_shape
        rng = np.random.default_rng(seed)
        phi_values = rng.uniform(0.05, 1.0, size=(ng, nx, ny))
        return ScalarFlux(values=phi_values, space=solver.sn_mesh.bulk_space)

    def test_apply_angular_flux_bit_identical_to_pre_t3_snapshot(
        self, op_p1, solver_2g_p1_n2n,
    ):
        """L1-1 per spec §3 — `apply(AngularFlux)` per-ordinate output
        matches the pre-T.3 captured snapshot within
        `nulp ≤ 4·scattering_order`.

        Post-T.3c the AngularFlux arm inherits the kernel-routed
        numerics via `_redistribute_ordinates`.  P0 + (n,2n)
        contribution is bit-identical (unchanged code path).  The
        per-ℓ aniso contribution may drift by `(L+1) × ULP` per the
        principled-equivalence three-criteria gate; the
        `(iso/sum_w + aniso)` combination at the apply boundary
        preserves the drift bound.
        """
        psi = self._reproduce_psi(solver_2g_p1_n2n, seed=20260530)
        # §14.1: the snapshot froze the PRE-extraction fused source
        # (P0 + aniso + n2n in one accumulator) — the composed
        # ``S + N2N`` must reproduce it, so the frozen artifact is the
        # extraction's value-preservation anchor. The summation order
        # changed ((isoS+isoN2N)/W+aniso → (isoS/W+aniso)+isoN2N/W);
        # allclose at 1e-14 relative bounds the reassociation drift
        # (principled-equivalence, vv three-criteria).
        out_post_t3 = (
            bulk_apply(op_p1, psi).values
            + bulk_apply(solver_2g_p1_n2n.n2n_op, psi).values
        )
        expected = self._load_snapshot()["p1_apply_angular_flux"]
        np.testing.assert_allclose(
            out_post_t3, expected, rtol=1e-13, atol=1e-16,
        )

    def test_apply_scalar_flux_bit_identical_to_pre_t3_snapshot(
        self, op_p1, solver_2g_p1_n2n,
    ):
        r"""L1-2 per spec §3 — the ℓ=0 iso scalar output matches the pre-T.3
        captured snapshot **bit-identically**.

        ⛔ RE-KEYED (CS4c step 5, R-3). The frozen array is the value the
        retired ``S.apply(ScalarFlux)`` arm returned: P0 + (n,2n) in iso
        scalar magnitude, no :math:`1/W`, no aniso (the arm never called
        the redistribution route and therefore never the kernel). That arm is
        gone — a scalar operand is the ENERGY binding's, and the ENERGY
        binding is exactly what the arm delegated to — so the snapshot is
        now read against ``S.isotropic_energy.apply(φ.values) +
        N2N.isotropic_energy.apply(φ.values)``. The claim is UNCHANGED (the
        same numbers, from the same code, reached by its own name);
        ``np.array_equal`` stays the gate, and the arm's REFUSAL is pinned
        separately in ``test_n2n_operator.py``.
        """
        phi = self._reproduce_phi(solver_2g_p1_n2n, seed=20260530 + 1)
        # §14.1 composition (see the angular row): the scalar snapshot is
        # P0 + (n,2n) in scalar magnitude; the (n,2n) half is the ENERGY
        # binding (no /W on the scalar arm). Addition order matches the
        # old accumulator (P0 then n2n) ⟹ bit-equality survives.
        out_post_t3 = (
            np.asarray(op_p1.isotropic_energy.apply(phi.values))
            + solver_2g_p1_n2n.n2n_op.isotropic_energy.apply(phi.values)
        )
        expected = self._load_snapshot()["p1_apply_scalar_flux"]
        np.testing.assert_array_equal(out_post_t3, expected)

    def test_apply_timed_full_field_bit_identical_to_pre_t3_snapshot(
        self, op_p1, solver_2g_p1_n2n,
    ):
        """L1-3 per spec §3 — `apply(TimedFullField)` bulk + boundary
        output matches the pre-T.3 captured snapshot.

        Bulk: nulp ≤ 4·order (inherits from AngularFlux-style
        kernel-routed numerics).  Boundary: bit-identical (the
        implicit-zero AngularBoundaryFlux from Option β3 is unchanged
        across T.3).
        """
        from dataclasses import replace

        psi = self._reproduce_psi(solver_2g_p1_n2n, seed=20260530)
        state = TimedFullField.zeros(interior=AngularFlux, boundary=AngularBoundaryFlux, space=solver_2g_p1_n2n.sn_mesh.full_field_space)
        state = replace(state, interior=replace(state.interior, values=psi.values))

        # §14.1 composition (see the angular row): the frozen bulk is the
        # PRE-extraction fused source; ``S + N2N`` on the composite must
        # reproduce it (reassociation-bounded).
        s_out = op_p1.apply(state)
        n_out = solver_2g_p1_n2n.n2n_op.apply(state)
        out_post_t3 = s_out
        snapshots = self._load_snapshot()

        np.testing.assert_allclose(
            s_out.interior.values + n_out.interior.values,
            snapshots["p1_apply_timed_full_field_bulk"],
            rtol=1e-13, atol=1e-16,
        )
        # Boundary: bit-identical (implicit zero, untouched by T.3).
        np.testing.assert_array_equal(
            out_post_t3.boundary.values,
            snapshots["p1_apply_timed_full_field_boundary"],
        )

    def test_redistribution_bit_identical_to_pre_t3_snapshot(
        self, op_p1, solver_2g_p1_n2n,
    ):
        """L1-4 per spec §3 — the ℓ ≥ 1 redistribution route's output
        (`_redistribute_ordinates`; until #448 the `build_aniso_source` verb
        wrapping it) matches the pre-T.3 captured snapshot within
        `nulp ≤ 4·scattering_order`.

        Pre-T.3 the body inlined `R(Λ(M(psi))) / sum_w`.  Post-Phase-5a
        the body projected `φ = M(psi)` once, then applied the shared
        moment→source map `_aniso_source_from_moment_values` (= `R·Λ`)
        and the `/ sum_w` boundary normalisation; since CS4c step 5
        (2026-09-04) the angular end runs the cached `kernel` (`R∘Λ∘M`)
        directly and the moment sibling the typed route — the SAME composition,
        with the per-ℓ `_PerLegendreOrderScattering` kernel (which rebuilt
        `M`/`R` per ℓ) retired.  The reduction tree may differ from the
        capture at the `Σ_ℓ` outer sum; drift bounded by `(L+1) × ULP`
        per `vv-principles` §"Bit-identity vs principled-equivalence"
        three-criteria gate.

        Snapshot lives at
        `tests/sn/_fixtures/wave_t_t3/pre_t3_snapshots.npz`, captured
        in T.3a (commit `ed05ea3`) on the same fixture
        (`solver_2g_p1_n2n`) with the same seed (20260530).
        """
        from orpheus.transport.fields.angular_flux import AngularFlux

        # Reproduce the fixture's seed + psi construction from
        # `_capture_pre_t3_snapshots.py::_make_psi(solver, seed=20260530)`.
        op = op_p1
        sn_mesh = solver_2g_p1_n2n.sn_mesh
        rng = np.random.default_rng(20260530)
        N = solver_2g_p1_n2n.sn_mesh.quad.N
        psi_values = rng.uniform(
            0.05, 1.0,
            size=(
                N, solver_2g_p1_n2n.ng,
                *solver_2g_p1_n2n.sn_mesh.spatial_shape,
            ),
        )
        psi = AngularFlux(values=psi_values, space=sn_mesh.angular_bulk_space)

        # Post-T.3c output via the kernel-routed redistribution.
        out_post_t3 = op._redistribute_ordinates(psi).values

        # Pre-T.3 snapshot.
        from tests.sn._test_helpers import SN_TESTS_ROOT

        snapshot_path = (
            SN_TESTS_ROOT / "_fixtures" / "wave_t_t3" / "pre_t3_snapshots.npz"
        )
        snapshots = np.load(snapshot_path)
        expected = snapshots["p1_build_aniso_source"]

        nulp_bound = max(4, 4 * op.legendre_order)
        np.testing.assert_array_almost_equal_nulp(
            out_post_t3, expected, nulp=nulp_bound,
        )

    def test_per_material_einsum_invariance_p1(self, op_p1, solver_2g_p1_n2n):
        """L6-1 per spec §3 — `MaterialXSField.apply_legendre_scattering_moments`
        output is bit-identical at P=1 to the pre-T.3 snapshot.

        T.3 does NOT touch `material_xs_field.py:515-572` — this test
        defends against an unintentional modernisation while in the
        file.  The per-material per-ℓ einsum is the leaf primitive;
        no FP reduction reorder; `np.array_equal` is the appropriate
        gate.
        """
        from orpheus.transport.fields.angular_flux import AngularFlux

        # Reproduce the snapshot script's psi (seed=20260530).
        psi_p1 = self._reproduce_psi(solver_2g_p1_n2n, seed=20260530)
        L = 1
        moments_values = op_p1.frame.analysis.apply(psi_p1.values)

        # The moment verb (CS4c step 3: the arm moved to the kernel
        # field; the snapshot pins the einsum leaf unchanged).
        # skip_l0=False → full block coverage; L == the operator's own
        # truncation (== 1 on this fixture, asserted below).
        assert op_p1.legendre_order == L
        out = op_p1.transfer.moment_source(
            moments_values, skip_l0=False, head=op_p1.frame.basis.space,
        )
        expected = self._load_snapshot()["p1_apply_legendre_scattering_moments"]
        np.testing.assert_array_equal(out, expected)

    def test_per_material_einsum_invariance_p3(self):
        """L6-2 per spec §3 — same as L6-1 but at P=3, exercising the
        higher-order ℓ loop body.

        Builds an independent P3 solver (mirroring
        `_capture_pre_t3_snapshots.py::build_p3_solver`) to reach the
        captured snapshot.
        """
        from orpheus.transport.fields.angular_flux import AngularFlux
        from scipy.sparse import csr_matrix

        p0 = np.array([[0.38, 0.10], [0.05, 0.90]])
        p1 = np.array([[0.02, 0.01], [0.00, 0.04]])
        p2 = np.array([[0.005, 0.002], [0.000, 0.010]])
        p3 = np.array([[0.001, 0.0005], [0.000, 0.002]])
        mix = make_mixture(
            sig_t=np.array([0.5, 1.0]),
            sig_c=np.array([0.01, 0.02]),
            sig_f=np.array([0.01, 0.08]),
            nu=np.array([2.5, 2.5]),
            chi=np.array([1.0, 0.0]),
            sig_s=p0,
        )
        mix.SigS = [csr_matrix(p0), csr_matrix(p1), csr_matrix(p2), csr_matrix(p3)]
        mix.Sig2 = [csr_matrix(np.array([[0.0, 0.03], [0.01, 0.0]]))]

        nx, ny = 3, 2
        mesh = _uniform_2d(nx, ny, 0.4, np.zeros((nx, ny), dtype=int))
        quad = Quadrature.lebedev(order=17)
        solver_p3 = SNSolver(SNMesh(mesh, quad, {0: mix}), scattering_order=3)
        op_p3 = solver_p3.scattering_op

        rng = np.random.default_rng(20260530 + 2)
        psi_p3 = AngularFlux(values=rng.uniform(0.05, 1.0, size=(quad.N, 2, nx, ny)), space=solver_p3.sn_mesh.angular_bulk_space)
        L = 3
        moments_values = op_p3.frame.analysis.apply(psi_p3.values)
        assert op_p3.legendre_order == L
        head = op_p3.frame.basis.space
        assert isinstance(head, MomentHead)
        out = op_p3.transfer.moment_source(
            moments_values, skip_l0=False, head=head,
        )
        expected = self._load_snapshot()["p3_apply_legendre_scattering_moments"]
        np.testing.assert_array_equal(out, expected)
