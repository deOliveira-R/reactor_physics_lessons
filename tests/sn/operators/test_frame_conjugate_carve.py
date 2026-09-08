r"""P2 operator-algebra carve — new-convention catchers for ``frame.conjugate`` /
``frame.reconstruct_after`` and the ``LegendreMomentTransfer`` real-space leaf.

Born as SPECS written BEFORE the production carve (main-agent-direct, branch
``refactor/operator-inverse-algebra``, plan ``frame_projection_machinery.md`` P2);
the carve has since LANDED, so these are the live NEW-convention catchers. The
LEGACY-pinning half is the EXISTING suite (named in the plan, stays byte-for-byte green).

What P2 does (the SUT this file gates)
--------------------------------------

1. Two ``FrameBase`` methods:
     ``frame.conjugate(A)         -> R ∘ A ∘ M``   (OperatorProduct(R, OperatorProduct(A, M)))
     ``frame.reconstruct_after(A) -> R ∘ A``        (OperatorProduct(R, A))
2. ``LegendreMomentTransfer`` gets REAL spaces
     (``domain == codomain == frame.basis_space``), retiring the
     ``cast(LinearOperator, …)`` at ``scattering.py:663``.
3. Production ``apply`` arms call the composed operator:
     ``kernel`` becomes ``frame.conjugate(Λ)``; the windowed-moment arm uses
     ``frame.reconstruct_after(Λ)``. The hand-chains retire.

vv discipline carried by every row below
----------------------------------------

* **Mode-8 (-O strip):** every assertion is ``np.testing.*`` / ``require`` /
  ``pytest.raises`` (function calls — fire under ``python -O``). NEVER a bare
  ``assert`` (this file is under ``orpheus/``-adjacent test rules; bare asserts
  inside test modules ARE AST-rewritten by pytest even under -O, but we use the
  function-call form anyway so the SAME idiom carries into any helper).
* **Mode-11 (gate-never-executes-the-rewired-path):** the production-equivalence
  rows read ``S.kernel`` / the windowed ``apply`` arm OFF the LIVE operator and
  the §"call-graph sentinel" row monkeypatch-wraps ``frame.conjugate`` to PROVE
  the production ``kernel`` property actually calls it (counter > 0). A green
  twin that routed around ``conjugate`` would leave the counter at 0.
* **L11 structural independence:** ``frame.conjugate(Λ)`` is checked against a
  reference built from the SAME R/Λ/M *factors* (an EQUIVALENCE/de-risk leg, NOT
  a physics-correctness claim — that reference is the aniso MMS gate). The
  factor-order reference is deliberately the manual nesting so a swap reddens.

``foundation`` — software invariants on the operator-algebra surface + a
bit-identity equivalence check. No theory ``:label:`` (the physics labels
``pn-scatter`` / ``flux-moments`` are pinned by the MMS gate, NOT here).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from orpheus.derivations.common.xs_library import make_mixture
from orpheus.geometry import Mesh2D
from orpheus.numerics.operator import (
    LinearOperator,
    OperatorProduct,
)
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.transport.operators.transfer import LegendreMomentTransfer
from orpheus.sn.solver import SNSolver
from orpheus.transport.fields.angular_flux import AngularFlux

from tests.sn.operators._composite_operand import bulk_apply
from tests.transport._integral_kernel_helpers import require
from orpheus.transport.material_field import TransferMaterialField

pytestmark = pytest.mark.foundation


# ── -O-safe SUT probes (self-healing PRE-IMPL guards — live now the carve landed) ─


def _require_conjugate(frame):
    """``frame.conjugate`` — self-healing PRE-IMPL guard (the P2 carve has landed)."""
    if not hasattr(frame, "conjugate"):
        pytest.skip("P2 PRE-IMPL: FrameBase.conjugate not yet written.")
    return frame.conjugate


def _require_reconstruct_after(frame):
    """``frame.reconstruct_after`` — self-healing PRE-IMPL guard (the P2 carve has landed)."""
    if not hasattr(frame, "reconstruct_after"):
        pytest.skip("P2 PRE-IMPL: FrameBase.reconstruct_after not yet written.")
    return frame.reconstruct_after


# ── ANISOTROPIC (P1) + heterogeneous + 2G fixture (Λ's ℓ≥1 blocks active) ─────


def _uniform_2d(nx, ny, delta, mat_map):
    return Mesh2D(
        edges_x=np.linspace(0, nx * delta, nx + 1),
        edges_y=np.linspace(0, ny * delta, ny + 1),
        mat_map=np.asarray(mat_map, dtype=int),
    )


@pytest.fixture
def solver_p1_het():
    """P1 + heterogeneous + 2G — the same activation regime the kernel
    crosscheck uses (Λ's ℓ≥1 blocks genuinely exercised)."""
    p0_a = np.array([[0.38, 0.10], [0.05, 0.90]])
    p1_a = np.array([[0.02, 0.01], [0.00, 0.04]])
    p0_b = np.array([[0.55, 0.03], [0.12, 0.40]])
    p1_b = np.array([[0.06, 0.02], [0.01, 0.03]])

    def _mix(p0, p1):
        m = make_mixture(
            sig_t=np.array([0.5, 1.0]),
            sig_c=np.array([0.01, 0.02]),
            sig_f=np.array([0.0, 0.0]),
            nu=np.array([0.0, 0.0]),
            chi=np.zeros(2),
            sig_s=p0,
        )
        m.SigS = [csr_matrix(p0), csr_matrix(p1)]
        m.Sig2 = [csr_matrix(np.array([[0.0, 0.03], [0.01, 0.0]]))]
        return m

    nx, ny = 4, 3
    mat = np.zeros((nx, ny), dtype=int)
    mat[2:, :] = 1
    mesh = _uniform_2d(nx, ny, 0.4, mat)
    quad = Quadrature.lebedev(order=17)
    sn_mesh = SNMesh(mesh, quad, {0: _mix(p0_a, p1_a), 1: _mix(p0_b, p1_b)})
    return SNSolver(sn_mesh, scattering_order=1)


def _aniso_psi(solver, seed=20260624):
    N, ng = solver.quad.N, solver.ng
    nx, ny = solver.sn_mesh.spatial_shape
    rng = np.random.default_rng(seed)
    return AngularFlux(values=rng.uniform(0.05, 1.0, size=(N, ng, nx, ny)), space=solver.sn_mesh.angular_bulk_space)


# ═════════════════════════════════════════════════════════════════════════════
# (b)(i) — LegendreMomentTransfer advertises REAL spaces == basis_space,
#          and the kernel composes WITHOUT the cast.
# ═════════════════════════════════════════════════════════════════════════════


class TestLegendreMomentTransferHasRealSpaces:
    """Λ is an endomorphism on the SH coefficient space (``= frame.basis_space``).

    Catches: the cast at ``scattering.py:663`` is gone because Λ now carries
    real ``domain``/``codomain`` so ``OperatorProduct`` admits it natively.
    """

    def test_lambda_domain_is_codomain_is_basis_space(self, solver_p1_het):
        op = solver_p1_het.scattering_op
        frame = op.frame
        lam = LegendreMomentTransfer.on_frame(
            TransferMaterialField.scattering(solver_p1_het.mat_xs), op.frame, skip_l0=True,
        )
        # Λ is endomorphic on coefficient (basis) space.
        require(
            getattr(lam, "domain", None) is not None,
            "P2: LegendreMomentTransfer.domain must be a real FunctionSpace "
            "(== frame.basis_space), not None — the cast at scattering.py:663 "
            "papered over None spaces; the carve gives Λ real spaces.",
        )
        require(
            lam.domain == frame.basis_space and lam.codomain == frame.basis_space,
            f"P2: Λ must be an endomorphism on the SH coefficient space "
            f"(frame.basis_space={frame.basis_space!r}); got domain={lam.domain!r}, "
            f"codomain={lam.codomain!r}.",
        )

    def test_lambda_apply_and_transpose_not_solve(self, solver_p1_het):
        """Λ advertises apply + apply_transpose (Λᵀ, campaign #276), never solve.

        The P2 real-spaces (cast retirement) did not change the no-``solve``
        property (the ℓ=0 block is rank-deficient by design); campaign #276 added
        ``apply_transpose`` — the per-ℓ group-transpose Λᵀ, the only group-
        asymmetric factor of the frame-conjugated kernel ``R∘Λ∘M`` (so
        ``(R∘Λ∘M)ᵀ`` falls out for free).
        """
        op = solver_p1_het.scattering_op
        lam = LegendreMomentTransfer.on_frame(
            TransferMaterialField.scattering(solver_p1_het.mat_xs), op.frame, skip_l0=True,
        )
        require(
            lam.is_adjointable and not lam.is_invertible,
            f"Λ must carry a working apply_transpose (campaign #276) and NO "
            f"inverse; got is_adjointable={lam.is_adjointable}, "
            f"is_invertible={lam.is_invertible}.",
        )
        require(
            not hasattr(lam, "inverse") and not hasattr(lam, "solve"),
            "Λ is structurally non-invertible — no inverse()/solve declared "
            "(carve P4 rewire of the strict caps-equality pin).",
        )

    def test_inner_lambda_product_carries_real_spaces(self, solver_p1_het):
        """The INNER ``Λ∘M`` product's codomain must be Λ's (real) codomain.

        IMPORTANT (design note, surfaced at spec-write time): the OUTER
        ``S.kernel`` already reports real ``domain``/``codomain`` TODAY —
        ``OperatorProduct(R, OperatorProduct(Λ, M)).domain`` reads ``M.domain``
        (the measure_space) and ``.codomain`` reads ``R.codomain``, NEITHER of
        which is Λ. So an outer-product domain/codomain check does NOT gate the
        cast removal — it is green pre-carve.

        The fact that genuinely changes is the INNER product
        ``OperatorProduct(Λ, M).codomain``, which reads ``Λ.codomain``: ``None``
        pre-carve (Λ unparametrised), ``frame.basis_space`` post-carve. This
        row reads that inner codomain and demands it be real — so it is RED
        until Λ gets spaces. Build the inner product explicitly (mirrors the
        kernel's construction) and read its codomain.
        """
        op = solver_p1_het.scattering_op
        frame = op.frame
        lam = LegendreMomentTransfer.on_frame(
            TransferMaterialField.scattering(solver_p1_het.mat_xs), op.frame, skip_l0=True,
        )
        inner = OperatorProduct(lam, frame.analysis)
        require(
            inner.codomain is not None,
            "P2: OperatorProduct(Λ, M).codomain must be Λ's real codomain "
            "(== frame.basis_space), not None. It is None pre-carve because Λ "
            "is unparametrised; the carve gives Λ real spaces so the cast at "
            "scattering.py:663 retires.",
        )
        require(
            inner.codomain == frame.basis_space,
            f"P2: the inner Λ∘M product's codomain must be frame.basis_space "
            f"(Λ is endomorphic on coefficient space); got {inner.codomain!r}.",
        )

    def test_kernel_remains_typed_operator_product(self, solver_p1_het):
        """``S.kernel`` stays a typed OperatorProduct (R∘Λ∘M) post-carve —
        an invariant the carve must preserve (not gate)."""
        op = solver_p1_het.scattering_op
        require(
            isinstance(op.kernel, OperatorProduct),
            f"P2: S.kernel must remain a typed OperatorProduct (R∘Λ∘M); got "
            f"{type(op.kernel).__name__}.",
        )


# ═════════════════════════════════════════════════════════════════════════════
# (b)(ii) — frame.conjugate(Λ) == R∘Λ∘M and frame.reconstruct_after(Λ) == R∘Λ
#           on INDEPENDENT reference inputs.
# ═════════════════════════════════════════════════════════════════════════════


class TestFrameConjugateEqualsRLambdaM:
    """``frame.conjugate(A) == OperatorProduct(R, OperatorProduct(A, M))``."""

    def test_conjugate_equals_manual_R_A_M_nesting(self, solver_p1_het):
        """conjugate(Λ).apply(ψ) == R(Λ(M·ψ)) on a non-isotropic reference ψ.

        L11 EQUIVALENCE leg: the reference is the SAME R/Λ/M factors composed
        BY HAND in the canonical R∘Λ∘M order — so a swapped M↔R (mutation
        probe d-i) reddens, while the physics-correctness reference stays the
        aniso MMS gate.
        """
        op = solver_p1_het.scattering_op
        frame = op.frame
        conjugate = _require_conjugate(frame)
        lam = LegendreMomentTransfer.on_frame(
            TransferMaterialField.scattering(solver_p1_het.mat_xs), op.frame, skip_l0=True,
        )
        psi = _aniso_psi(solver_p1_het)

        via_conjugate = conjugate(lam).apply(psi.values)
        # Independent manual nesting (the canonical order R∘Λ∘M).
        manual = frame.reconstruction.apply(
            lam.apply(frame.analysis.apply(psi.values)),
        )
        np.testing.assert_array_equal(
            via_conjugate, manual,
            err_msg="P2: frame.conjugate(Λ).apply(ψ) must equal the manual "
            "R(Λ(M·ψ)) nesting BIT-IDENTICALLY — conjugate composes "
            "OperatorProduct(R, OperatorProduct(Λ, M)). A mismatch means the "
            "factor order (M↔R swap) or composition is wrong.",
        )

    def test_conjugate_is_non_degenerate(self, solver_p1_het):
        """The reference ψ genuinely activates ℓ≥1 (else the leg is vacuous)."""
        op = solver_p1_het.scattering_op
        psi = _aniso_psi(solver_p1_het)
        moments = op.frame.analysis.apply(psi.values)
        require(
            bool(np.any(moments[1:] != 0.0)),
            "P2: ℓ≥1 moments must be non-zero so R∘Λ∘M runs the aniso "
            "reconstruction — else conjugate(Λ) == 0 trivially.",
        )


class TestFrameReconstructAfterEqualsRLambda:
    """``frame.reconstruct_after(A) == OperatorProduct(R, A)`` (M already applied)."""

    def test_reconstruct_after_equals_manual_R_A(self, solver_p1_het):
        """reconstruct_after(Λ).apply(φ) == R(Λ·φ) on a moment-tensor input φ.

        This is the windowed-moment arm's case: the iterate bulk IS the moment
        tensor φ = M·ψ, so only R∘Λ remains. Feed a moment tensor directly so
        the reference shares NO M projection with the SUT (independence on the
        input side).
        """
        op = solver_p1_het.scattering_op
        frame = op.frame
        reconstruct_after = _require_reconstruct_after(frame)
        lam = LegendreMomentTransfer.on_frame(
            TransferMaterialField.scattering(solver_p1_het.mat_xs), op.frame, skip_l0=True,
        )
        psi = _aniso_psi(solver_p1_het)
        moments = frame.analysis.apply(psi.values)  # φ = M·ψ (the windowed bulk)

        via_recon_after = reconstruct_after(lam).apply(moments)
        manual = frame.reconstruction.apply(lam.apply(moments))
        np.testing.assert_array_equal(
            via_recon_after, manual,
            err_msg="P2: frame.reconstruct_after(Λ).apply(φ) must equal R(Λ·φ) "
            "BIT-IDENTICALLY — reconstruct_after composes OperatorProduct(R, Λ). "
            "It must NOT re-apply M (the moments are already M·ψ).",
        )

    def test_reconstruct_after_does_not_reapply_M(self, solver_p1_het):
        """``reconstruct_after`` consumes ALREADY-projected moments (no second M).

        Guards the carve's most plausible slip: wiring the windowed arm — whose
        input is the moment tensor φ = M·ψ — to ``conjugate`` (= R∘Λ∘M) instead of
        ``reconstruct_after`` (= R∘Λ). They are DISTINCT operators on distinct
        domains: ``reconstruct_after(Λ)`` accepts a moment tensor, while
        ``conjugate(Λ)`` expects per-ordinate ψ and would re-project (M cannot
        consume a moment tensor — it raises). Pinned positively:
        ``reconstruct_after(Λ)·(M·ψ) == conjugate(Λ)·ψ`` — feeding the windowed
        moments to reconstruct_after reproduces the full-arm result, so
        reconstruct_after is exactly "conjugate with M already done".
        """
        op = solver_p1_het.scattering_op
        frame = op.frame
        conjugate = _require_conjugate(frame)
        reconstruct_after = _require_reconstruct_after(frame)
        lam = LegendreMomentTransfer.on_frame(
            TransferMaterialField.scattering(solver_p1_het.mat_xs), op.frame, skip_l0=True,
        )
        psi = _aniso_psi(solver_p1_het)
        moments = frame.analysis.apply(psi.values)  # φ = M·ψ (the windowed bulk)

        np.testing.assert_array_equal(
            reconstruct_after(lam).apply(moments),
            conjugate(lam).apply(psi.values),
            err_msg="P2: reconstruct_after(Λ)(M·ψ) must equal conjugate(Λ)(ψ) — "
            "reconstruct_after is conjugate with M already applied; the windowed "
            "arm feeds M·ψ to it WITHOUT a second projection.",
        )
        # Distinct domains: conjugate re-applies M, which CANNOT consume a moment
        # tensor (per-ordinate input expected) — a windowed-arm-wired-to-conjugate
        # slip fails loudly here, not as a silent double-projection.
        with pytest.raises((ValueError, IndexError)):
            conjugate(lam).apply(moments)


# ═════════════════════════════════════════════════════════════════════════════
# (b)(iii) + (c) Mode-11 — production apply ≡ composed operator AND the
#                production path actually EXECUTES frame.conjugate.
# ═════════════════════════════════════════════════════════════════════════════


class TestProductionApplyEqualsComposedOperator:
    """After the carve, production ``apply`` runs through the composed operator."""

    def test_kernel_property_is_frame_conjugate_of_lambda(self, solver_p1_het):
        """``S.kernel.apply(ψ) == frame.conjugate(Λ).apply(ψ)`` (0 ULP).

        Post-carve, ``S.kernel`` IS ``frame.conjugate(Λ)``. This pins the
        production kernel property to the new composed operator. Reads
        ``S.kernel`` OFF the live operator (Mode-11: no routing around).
        """
        op = solver_p1_het.scattering_op
        frame = op.frame
        conjugate = _require_conjugate(frame)
        lam = LegendreMomentTransfer.on_frame(
            TransferMaterialField.scattering(solver_p1_het.mat_xs), op.frame, skip_l0=True,
        )
        psi = _aniso_psi(solver_p1_het)
        np.testing.assert_array_equal(
            op.kernel.apply(psi.values),
            conjugate(lam).apply(psi.values),
            err_msg="P2: S.kernel must BE frame.conjugate(Λ) post-carve "
            "(0 ULP). The production kernel is the composed operator now.",
        )

    def test_full_apply_unchanged_by_carve_vs_legacy_chain(self, solver_p1_het):
        """The full ``S.apply(ψ)`` (per-ordinate, iso+aniso, /W) equals a
        reference built from the composed kernel + the P0 fast path.

        The carve must NOT move the converged per-ordinate source. Builds the
        reference from the NEW composed kernel for the aniso piece, the P0
        in-place add for the local piece, and the producer-side /W — so a
        dropped 1/W (probe d-ii) or a kernel factor error reddens the FULL
        apply, not just the kernel sub-component. (The (n,2n) term left
        ``S`` with the §14.1 extraction — its lift is N2NOperator's own
        gate; ``S.apply`` is P0 + aniso.)
        """
        op = solver_p1_het.scattering_op
        sn_mesh = solver_p1_het.sn_mesh
        psi = _aniso_psi(solver_p1_het)
        # CS4c step 5: the gain is composite-bound — the bulk action rides a
        # zero-trace composite (the trace the lift itself emits).
        full = bulk_apply(op, psi).values

        # Reference: aniso = (1/W)·kernel(ψ); P0 via the scalar fast path.
        sum_w = op.total_weight
        aniso = np.asarray(op.kernel.apply(psi.values)) / sum_w
        phi = psi.integrate_angular()
        ng = solver_p1_het.ng
        nx, ny = sn_mesh.spatial_shape
        N = sn_mesh.quad.N
        iso = np.zeros((ng, nx, ny))
        # The P0 emission in place through the channel FIELD's verb (the
        # operator-level seam ``add_iso_source`` retired at #448): ndarray Q,
        # ndarray φ.
        op.transfer.add_p0_source(iso, phi.values)
        expected = np.broadcast_to(
            (iso / sum_w)[None, :, :, :], (N, ng, nx, ny),
        ) + aniso
        np.testing.assert_allclose(
            full, expected, rtol=1e-13, atol=1e-14,
            err_msg="P2: full S.apply must equal iso/W broadcast + "
            "kernel(ψ)/W. A dropped 1/W or a kernel factor error reddens here.",
        )


class TestProductionExecutesFrameConjugate:
    """vv Mode-11 SENTINEL: the production ``kernel`` property MUST call
    ``frame.conjugate`` — proven by an in-process counter wrap, NOT a bare
    assert (which -O strips) and NOT trust in a green twin.

    This is the gold-standard "the gate executes the rewired line" proof: an
    autouse-free monkeypatch that WRAPS the production reader (``FrameBase.conjugate``)
    and counts entries. A green path that routed around ``conjugate`` (e.g. the
    carve left ``kernel`` building the manual nesting) leaves the counter at 0
    and reddens this gate.
    """

    def test_kernel_property_actually_calls_frame_conjugate(
        self, solver_p1_het, monkeypatch,
    ):
        op = solver_p1_het.scattering_op
        frame = op.frame
        _require_conjugate(frame)  # skip PRE-IMPL

        from orpheus.numerics.frame import FrameBase

        calls: list[object] = []
        real_conjugate = FrameBase.conjugate  # type: ignore[attr-defined]

        def _counting_conjugate(self, A):
            calls.append(A)
            return real_conjugate(self, A)

        monkeypatch.setattr(
            FrameBase, "conjugate", _counting_conjugate, raising=True,
        )

        # Access the production property — must route through frame.conjugate.
        _ = op.kernel
        require(
            len(calls) > 0,
            "vv Mode-11: ScatteringOperator.kernel did NOT call "
            "frame.conjugate — the production property routes AROUND the new "
            "reader (e.g. still builds the manual OperatorProduct nesting). "
            "The kernel-equivalence gate would then be vacuous FOR the carve "
            "claim. Wire S.kernel to frame.conjugate(Λ).",
        )
