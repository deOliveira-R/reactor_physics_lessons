r"""Adjoint scattering — Λᵀ leaf + frame-conjugated kernel transpose (campaign #276, A2).

The SN scattering kernel is the frame conjugation :math:`R\circ\Lambda\circ M`
(Funk–Hecke: the SH angular frame is scattering's eigenbasis). Its Euclidean
transpose is :math:`M^{T}\circ\Lambda^{T}\circ R^{T}` — and since the angular
:math:`M`/:math:`R` faces already transpose (the Frame carve, Phase D), the ONLY
genuinely-new piece is :math:`\Lambda^{T}`, the per-ℓ group-axis transpose of the
block-diagonal :math:`\Sigma_{s,\ell}` matmul. Once :math:`\Lambda` advertises
``apply_transpose``, ``(R∘Λ∘M).apply_transpose`` falls out of
:meth:`OperatorProduct.apply_transpose` for free.

This file gates the A2 leaf:

* **Λᵀ** — the moment-space transpose identity ``⟨Λ m, c⟩ = ⟨m, Λᵀ c⟩`` (the
  DEFINING transpose property), a structurally-independent per-material dense
  ``sigᵀ`` reference, the group-flip mutation (Λᵀ ≠ Λ on asymmetric Σ_s), and the
  capability flip.
* **kernel = R∘Λ∘M** — the Euclidean reciprocity ``⟨kernel ψ, c⟩ = ⟨ψ, kernelᵀ c⟩``
  (the aniso transpose now live via the operator algebra) + capability propagation.
* **LD trailing-axis threading** — the frame form reproduces the fast-path
  forward on a TRUE LD :class:`AngularFlux` (trailing :math:`2^d` φ̂ axis), and
  the ``(Ellipsis,*idx)`` → ``(slice,slice,*idx)`` cells-index fix (#276 A2,
  ``0b3275d``) has mutation teeth (promoted from
  ``diag_276_full_scatter_kernel_ld_trailing_axis``, campaign #276 A4).

The metric-correct Hilbert adjoint ``S† = G⁻¹SᵀG`` is the ``.H`` wrapper's job
(A3/A4); these gates pin the BARE Euclidean transpose (per L27: per-group / full
tensor contraction, never a weight-summed scalar that telescopes).

vv Mode-8: ``np.testing.*`` / :func:`require` only (fire under ``python -O``).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from orpheus.derivations.common.xs_library import make_mixture
from orpheus.geometry import Mesh2D
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.sn.solver import SNSolver
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.operators.transfer import LegendreMomentTransfer
from orpheus.numerics.basis.spherical_harmonic_basis import SphericalHarmonicBasis
from orpheus.transport.material_field import TransferMaterialField
from tests.sn.operators._composite_operand import bulk_apply, transpose_values

pytestmark = pytest.mark.foundation


def require(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _uniform_2d(nx, ny, delta, mat_map):
    return Mesh2D(
        edges_x=np.linspace(0, nx * delta, nx + 1),
        edges_y=np.linspace(0, ny * delta, ny + 1),
        mat_map=np.asarray(mat_map, dtype=int),
    )


def _mix(p0, p1):
    m = make_mixture(
        sig_t=np.array([0.5, 1.0]), sig_c=np.array([0.01, 0.02]),
        sig_f=np.array([0.0, 0.0]), nu=np.array([0.0, 0.0]),
        chi=np.zeros(2), sig_s=p0,
    )
    m.SigS = [csr_matrix(p0), csr_matrix(p1)]
    m.Sig2 = [csr_matrix(np.array([[0.0, 0.03], [0.01, 0.0]]))]
    return m


# ASYMMETRIC P0 + P1 blocks per material (so a group-axis transpose is detectable).
_P0_A = np.array([[0.38, 0.10], [0.05, 0.90]]); _P1_A = np.array([[0.02, 0.01], [0.00, 0.04]])
_P0_B = np.array([[0.55, 0.03], [0.12, 0.40]]); _P1_B = np.array([[0.06, 0.02], [0.01, 0.03]])


@pytest.fixture
def solver_p1_het():
    nx, ny = 4, 3
    mat = np.zeros((nx, ny), dtype=int); mat[:2, :] = 0; mat[2:, :] = 1
    sn_mesh = SNMesh(_uniform_2d(nx, ny, 0.4, mat), Quadrature.lebedev(order=17),
                     {0: _mix(_P0_A, _P1_A), 1: _mix(_P0_B, _P1_B)})
    return SNSolver(sn_mesh, scattering_order=1)


def _ld_solver_het(order: int, nx: int = 4, ny: int = 3) -> SNSolver:
    """Heterogeneous 2-D LD solver — RECTANGULAR (nx != ny) so a wrong-axis
    index over-runs (the cleanest Mode-2 mutation tell)."""
    from orpheus.transport.spatial import LinearDiscontinuous

    mat = np.zeros((nx, ny), dtype=int); mat[nx // 2:, :] = 1
    sn_mesh = SNMesh(
        _uniform_2d(nx, ny, 0.1, mat), Quadrature.product(n_mu=4, n_phi=4),
        {0: _mix(_P0_A, _P1_A), 1: _mix(_P0_B, _P1_B)},
        scheme=LinearDiscontinuous(),
    )
    return SNSolver(sn_mesh, scattering_order=order)


def _ld_flux(solver: SNSolver, seed: int = 123) -> AngularFlux:
    """Random LD :class:`AngularFlux` — trailing ``2^d = 4`` φ̂-moment axis."""
    N = solver.quad.N
    nx, ny = solver.sn_mesh.spatial_shape
    vals = np.random.default_rng(seed).uniform(
        0.05, 1.0, size=(N, solver.ng, nx, ny, 4),
    )
    return AngularFlux(values=vals, space=solver.sn_mesh.angular_trial_space)


def _moment_field(op, nx, ny, seed):
    return np.random.default_rng(seed).uniform(0.05, 1.0, size=(2, 3, op.transfer.ng, nx, ny))


# ═══════════════════════════════════════════════════════════════════════
# Λᵀ — the one genuinely-new leaf (per-ℓ group-axis transpose).
# ═══════════════════════════════════════════════════════════════════════


class TestLambdaTranspose:
    def test_predicates_adjointable_not_invertible(self, solver_p1_het):
        lam = LegendreMomentTransfer.on_basis(
            TransferMaterialField.scattering(solver_p1_het.mat_xs), SphericalHarmonicBasis(L=1),
        )
        require(lam.is_adjointable,
                "Λ must advertise the adjoint axis (campaign #276).")
        require(not lam.is_invertible,
                "Λ must NOT be invertible (ℓ=0 block rank-deficient).")

    def test_moment_space_transpose_identity(self, solver_p1_het):
        r"""``⟨Λ m, c⟩ = ⟨m, Λᵀ c⟩`` (full moment-tensor contraction, per L27)."""
        op = solver_p1_het.scattering_op
        nx, ny = solver_p1_het.mat_xs.spatial_shape
        lam = LegendreMomentTransfer.on_basis(
            TransferMaterialField.scattering(solver_p1_het.mat_xs), SphericalHarmonicBasis(L=1), skip_l0=False,
        )
        m = _moment_field(op, nx, ny, 1); c = _moment_field(op, nx, ny, 2)
        lhs = float((lam.apply(m) * c).sum())            # ⟨Λ m, c⟩
        rhs = float((m * lam.apply_transpose(c)).sum())  # ⟨m, Λᵀ c⟩
        np.testing.assert_allclose(
            lhs, rhs, rtol=1e-12,
            err_msg="Λ moment-space transpose identity ⟨Λm,c⟩=⟨m,Λᵀc⟩ violated.",
        )

    def test_transpose_matches_dense_per_material(self, solver_p1_het):
        r"""STRUCTURALLY-INDEPENDENT: Λᵀ block = transpose of the forward matrix.

        The forward verb (``einsum("mfc,fg->mgc")``) applies, per (cell, ℓ, m),
        the matrix :math:`A = \Sigma_{s,\ell}^{T}` to the group vector
        (``out_g = Σ_f in_f·sig[f,g]``). The transpose therefore applies
        :math:`A^{T} = \Sigma_{s,\ell}` (un-transposed) — built here by an explicit
        per-(material, ℓ, cell) ``sig @ vec`` Python loop (no einsum). A wrong group
        axis in the production transpose verb disagrees with it.
        """
        op = solver_p1_het.scattering_op
        nx, ny = solver_p1_het.mat_xs.spatial_shape
        lam = LegendreMomentTransfer.on_basis(
            TransferMaterialField.scattering(solver_p1_het.mat_xs), SphericalHarmonicBasis(L=1), skip_l0=False,
        )
        c = _moment_field(op, nx, ny, 3)
        got = lam.apply_transpose(c)

        ref = np.zeros_like(c)
        for mid, idx in solver_p1_het.mat_xs.cells_by_material.items():
            sig = solver_p1_het.mat_xs.sig_s_legendre(mid)  # list over ℓ of (ng, ng) [g_from, g_to]
            for l in range(2):
                n_m = 2 * l + 1
                # Forward applies sigᵀ ⇒ transpose applies sig (un-transposed).
                sig_l = np.asarray(sig[l])
                for (ix, iy) in zip(*idx):
                    for mom in range(n_m):
                        ref[l, mom, :, ix, iy] = sig_l @ c[l, mom, :, ix, iy]
        np.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=0.0,
            err_msg="Λᵀ disagrees with the explicit per-material sig_s[ℓ] matmul "
            "(the transpose of the forward sigᵀ) — a wrong group axis in the verb.",
        )

    def test_group_flip_is_nontrivial(self, solver_p1_het):
        r"""Discriminator: with asymmetric Σ_s, Λᵀ ≠ Λ (the transpose has teeth)."""
        op = solver_p1_het.scattering_op
        nx, ny = solver_p1_het.mat_xs.spatial_shape
        lam = LegendreMomentTransfer.on_basis(
            TransferMaterialField.scattering(solver_p1_het.mat_xs), SphericalHarmonicBasis(L=1), skip_l0=False,
        )
        m = _moment_field(op, nx, ny, 4)
        require(
            not np.allclose(lam.apply(m), lam.apply_transpose(m)),
            "Λ and Λᵀ agreed on asymmetric Σ_s — the fixture lost its asymmetry, "
            "so the group-flip gate is blind to a transpose error.",
        )


# ═══════════════════════════════════════════════════════════════════════
# kernel = R∘Λ∘M — the aniso transpose, now free via the operator algebra.
# ═══════════════════════════════════════════════════════════════════════


class TestKernelTranspose:
    def test_kernel_advertises_apply_transpose(self, solver_p1_het):
        kernel = solver_p1_het.scattering_op.kernel
        require(
            kernel.is_adjointable,
            "kernel (R∘Λ∘M) must propagate adjointability once Λ has it "
            "(OperatorProduct all-factors law).",
        )

    def test_kernel_euclidean_reciprocity(self, solver_p1_het):
        r"""``⟨kernel ψ, c⟩ = ⟨ψ, kernelᵀ c⟩`` — the aniso R∘Λ∘M Euclidean transpose.

        Confirms ``(R∘Λ∘M)ᵀ = Mᵀ∘Λᵀ∘Rᵀ`` composes correctly through
        ``OperatorProduct.apply_transpose`` with the Phase-D M/R face transposes.
        Full per-ordinate/per-group contraction (NOT weight-summed — L27).
        """
        op = solver_p1_het.scattering_op
        nx, ny = solver_p1_het.mat_xs.spatial_shape
        N = solver_p1_het.sn_mesh.quad.N
        kernel = op.kernel
        rng = np.random.default_rng(5)
        psi = rng.uniform(0.05, 1.0, size=(N, solver_p1_het.ng, nx, ny))
        c = rng.uniform(0.05, 1.0, size=(N, solver_p1_het.ng, nx, ny))
        lhs = float((kernel.apply(psi) * c).sum())
        rhs = float((psi * kernel.apply_transpose(c)).sum())
        np.testing.assert_allclose(
            lhs, rhs, rtol=1e-12,
            err_msg="kernel (R∘Λ∘M) Euclidean reciprocity violated — Mᵀ∘Λᵀ∘Rᵀ "
            "did not compose correctly through OperatorProduct.apply_transpose.",
        )


# ═══════════════════════════════════════════════════════════════════════
# N2N — the (n,2n) ℓ=0 moment operator (distinct, in-frame).  The single ℓ is
# ORPHEUS's P0 truncation of the evaluated data, not the reaction (#426).
# ═══════════════════════════════════════════════════════════════════════


class TestN2NMomentTransfer:
    def test_predicates_adjointable_not_invertible(self, solver_p1_het):
        n2n = LegendreMomentTransfer.on_basis(
            TransferMaterialField.n2n(solver_p1_het.mat_xs), SphericalHarmonicBasis(L=1), skip_l0=False,
        )
        require(n2n.is_adjointable, "N2N must advertise the adjoint axis.")
        require(not n2n.is_invertible, "N2N must NOT be invertible.")

    def test_acts_only_on_ell0(self, solver_p1_het):
        r"""This FIXTURE's (n,2n) stack is P0-only, so Λ₂ₙ touches only that block.

        ⚠ Not a property of the channel: since #426 step 2 (2026-09-04) the
        (n,2n) binding reads the tape's whole stack at the solve's order
        (ERR-082). What this row pins is that a P0-only stack produces exactly
        zero above ℓ = 0 — the padding rule (O-1), not a truncation. Until
        2026-09-04 this docstring called the single block the data layer's
        P0 truncation; it stopped being the data layer's at step 1 and
        stopped being a truncation at step 2.
        """
        op = solver_p1_het.scattering_op
        nx, ny = solver_p1_het.mat_xs.spatial_shape
        n2n = LegendreMomentTransfer.on_basis(
            TransferMaterialField.n2n(solver_p1_het.mat_xs), SphericalHarmonicBasis(L=1), skip_l0=False,
        )
        m = _moment_field(op, nx, ny, 6)
        out = n2n.apply(m)
        np.testing.assert_array_equal(
            out[1:], np.zeros_like(out[1:]),
            err_msg=(
                "N2N wrote a non-zero ℓ≥1 block — the shipped kernel is ℓ=0 only "
                "(ORPHEUS's P0 model of the channel; #426)."
            ),
        )

    def test_moment_space_transpose_identity(self, solver_p1_het):
        op = solver_p1_het.scattering_op
        nx, ny = solver_p1_het.mat_xs.spatial_shape
        n2n = LegendreMomentTransfer.on_basis(
            TransferMaterialField.n2n(solver_p1_het.mat_xs), SphericalHarmonicBasis(L=1), skip_l0=False,
        )
        m = _moment_field(op, nx, ny, 7); c = _moment_field(op, nx, ny, 8)
        lhs = float((n2n.apply(m) * c).sum())
        rhs = float((m * n2n.apply_transpose(c)).sum())
        np.testing.assert_allclose(
            lhs, rhs, rtol=1e-12,
            err_msg="N2N moment-space transpose identity ⟨N2N m,c⟩=⟨m,N2Nᵀc⟩ violated.",
        )


# ═══════════════════════════════════════════════════════════════════════
# Full scatter kernel = frame.conjugate(Λ_{ℓ≥0}) — the A2a readiness gate:
# the frame form reproduces the CURRENT forward S (P0+aniso; the (n,2n)
# term is N2NOperator's own lift since the §14.1 extraction).
# ═══════════════════════════════════════════════════════════════════════


class TestFullScatterKernel:
    def _full_kernel(self, op):
        # The production property: frame.conjugate(Λ_{ℓ≥0}) — R∘Λ∘M (§14.1: n2n extracted).
        # The forward apply does NOT use this (it keeps the fast-path for perf,
        # campaign #276 A2a finding); it is the validated frame form for the
        # adjoint transpose (A2b) + the Option-2 forward-unification reference.
        return op.full_transfer_kernel

    def test_reproduces_forward_scattering_source(self, solver_p1_het):
        r"""``(1/W)·frame.conjugate(Λ_{ℓ≥0}).apply(ψ) == S.apply(ψ)`` (principled-equiv).

        The load-bearing A2a equivalence: the frame path reproduces the
        fast-path forward (P0 via add_iso, aniso via kernel) —
        principled-equiv (Y₀⁰=1; same math, reduction-order differs ⟹
        ~1e-14, NOT 0-ULP). The (n,2n) term left S with the §14.1
        extraction; its lift ≡ conjugation gate lives in
        ``test_n2n_operator.py``.
        """
        op = solver_p1_het.scattering_op
        nx, ny = solver_p1_het.mat_xs.spatial_shape
        W = op.total_weight
        psi = AngularFlux(values=np.random.default_rng(10).uniform(0.05, 1.0, size=(solver_p1_het.sn_mesh.quad.N, solver_p1_het.ng, nx, ny)), space=solver_p1_het.sn_mesh.angular_bulk_space)
        candidate = self._full_kernel(op).apply(psi.values) / W
        # CS4c step 5: the gain is composite-bound; the bulk action rides a
        # zero-trace composite (the trace the lift itself emits back).
        forward = bulk_apply(op, psi).values
        np.testing.assert_allclose(
            candidate, forward, rtol=1e-12, atol=0.0,
            err_msg="frame.conjugate(Λ_{ℓ≥0})/W does NOT reproduce the forward "
            "scattering source — the iso-modernization is not equivalent to the "
            "legacy fast-path.",
        )

    @pytest.mark.parametrize("trailing", [(), (4,)], ids=["scalar", "LD-2^d=4"])
    def test_full_kernel_euclidean_reciprocity(self, solver_p1_het, trailing):
        r"""``⟨S ψ, c⟩ = ⟨ψ, Sᵀ c⟩`` for the full P0+aniso kernel (the A2b transpose; §14.1: n2n is N2NOperator's).

        Scalar AND LD (trailing :math:`2^d` spectator, #240 D5b-S3): the transpose
        must thread the spatial-moment axis the same way the forward does (#276 P2).
        """
        op = solver_p1_het.scattering_op
        nx, ny = solver_p1_het.mat_xs.spatial_shape
        N = solver_p1_het.sn_mesh.quad.N
        fk = self._full_kernel(op)
        rng = np.random.default_rng(11)
        psi = rng.uniform(0.05, 1.0, size=(N, solver_p1_het.ng, nx, ny, *trailing))
        c = rng.uniform(0.05, 1.0, size=(N, solver_p1_het.ng, nx, ny, *trailing))
        lhs = float((fk.apply(psi) * c).sum())
        rhs = float((psi * fk.apply_transpose(c)).sum())
        np.testing.assert_allclose(
            lhs, rhs, rtol=1e-12,
            err_msg="full scatter kernel (P0+aniso+n2n) Euclidean reciprocity violated.",
        )

    # ── The PRODUCTION operator S† (campaign #276 A2b, closes #118) ──────────

    def test_S_advertises_apply_transpose(self, solver_p1_het):
        r"""``ScatteringOperator`` is adjointable (the #118 flip) but still NOT
        invertible (rank-deficient :math:`\ell=0` block)."""
        op = solver_p1_het.scattering_op
        require(op.is_adjointable,
                "S must advertise the adjoint axis (#276 A2b / #118).")
        require(not op.is_invertible,
                "S must NOT be invertible (the ℓ=0 group-transfer block is singular).")

    def test_S_apply_transpose_is_kernel_transpose_over_W(self, solver_p1_het):
        r"""WIRING (R4 near-tautology — the cheap catch for a missing ``1/W`` or a
        ``.apply``-not-``.apply_transpose`` typo):
        ``S.apply_transpose(χ) == (1/W)·full_transfer_kernel.apply_transpose(χ)``.
        """
        op = solver_p1_het.scattering_op
        nx, ny = solver_p1_het.mat_xs.spatial_shape
        N = solver_p1_het.sn_mesh.quad.N
        W = op.total_weight
        chi = np.random.default_rng(13).uniform(0.05, 1.0, size=(N, solver_p1_het.ng, nx, ny))
        np.testing.assert_allclose(
            transpose_values(op, chi),
            self._full_kernel(op).apply_transpose(chi) / W,
            rtol=1e-12, atol=0.0,
            err_msg="S.apply_transpose must route through (1/W)·full_transfer_kernel.apply_transpose.",
        )

    def test_S_euclidean_reciprocity(self, solver_p1_het):
        r"""**[LOAD-BEARING]** ``⟨S ψ, χ⟩ = ⟨ψ, Sᵀ χ⟩`` — the production FORWARD
        (scalar fast-path: ``isotropic_kernel`` + the ℓ ≥ 1 redistribution route) vs the
        production ADJOINT (frame form: ``full_transfer_kernelᵀ``).

        Two structurally-DIFFERENT representations of the same operator (spec R4),
        so reciprocity genuinely cross-checks the transpose against an INDEPENDENT
        forward — unlike the self-equivalence wiring gate
        (:meth:`test_S_apply_transpose_is_kernel_transpose_over_W`).  P0+P1, het,
        asymmetric SigS + Sig2≠0 (``solver_p1_het``).  ``rtol=1e-12`` is the
        fast-path↔frame-form forward-equivalence floor
        (:meth:`test_reproduces_forward_scattering_source`); a group-flip / dropped
        n2n / missing ``1/W`` in Sᵀ breaks it O(1).  ``-O``-safe.
        """
        op = solver_p1_het.scattering_op
        nx, ny = solver_p1_het.mat_xs.spatial_shape
        N = solver_p1_het.sn_mesh.quad.N
        rng = np.random.default_rng(12)
        psi = AngularFlux(values=rng.uniform(0.05, 1.0, size=(N, solver_p1_het.ng, nx, ny)), space=solver_p1_het.sn_mesh.angular_bulk_space)
        chi = rng.uniform(0.05, 1.0, size=(N, solver_p1_het.ng, nx, ny))
        sn_mesh = solver_p1_het.sn_mesh
        lhs = float((bulk_apply(op, psi).values * chi).sum())      # ⟨S ψ, χ⟩
        rhs = float((psi.values * transpose_values(op, chi)).sum())  # ⟨ψ, Sᵀ χ⟩
        np.testing.assert_allclose(
            lhs, rhs, rtol=1e-12,
            err_msg="S Euclidean reciprocity ⟨Sψ,χ⟩=⟨ψ,Sᵀχ⟩ violated (production "
            "forward fast-path vs adjoint frame form).",
        )


# ═══════════════════════════════════════════════════════════════════════
# LD trailing-axis threading — the frame form on a TRUE LD AngularFlux
# (promoted from diag_276_full_scatter_kernel_ld_trailing_axis, #276 A4;
# diagnostic authored by numerics-investigator 2026-06-28).
# ═══════════════════════════════════════════════════════════════════════


class TestFullScatterKernelLDTrailingAxis:
    r"""The #276/``0b3275d`` cells-index fix: LD's trailing φ̂ axis is threaded.

    ``TestFullScatterKernel`` above exercises non-LD fluxes plus a raw
    trailing-SPECTATOR reciprocity; these gates close the remaining gap —
    the frame form ``(1/W)·R∘(Λ+N2N)∘M`` reproducing the fast-path forward
    on a TRUE LD :class:`AngularFlux` (trailing :math:`2^d = 4`
    spatial-moment axis), heterogeneous, on a RECTANGULAR grid.

    Failure-mode class: Mode 2 (variable swap / wrong-axis), LD-gated: the
    pre-fix ``cells = (Ellipsis, *idx)`` indexing let ``Ellipsis`` greedily
    absorb the leading ``(m, g)`` axes so the spatial cell indices landed on
    (one spatial + the φ̂ axis) — invisible on every non-LD config
    (``Ellipsis ≡ (slice, slice)`` with no trailing axis), which is exactly
    why the regression gate must run an LD flux.
    """

    @pytest.mark.parametrize("order", [0, 1])
    def test_reproduces_forward_on_ld_flux(self, order):
        r"""``(1/W)·full_transfer_kernel.apply(ψ) == S.apply(ψ)`` on LD.

        The LD sibling of
        :meth:`TestFullScatterKernel.test_reproduces_forward_scattering_source`
        (principled-equiv: same math, reduction order differs ⟹ ~1e-15)."""
        solver = _ld_solver_het(order)
        op = solver.scattering_op
        psi = _ld_flux(solver)
        W = op.total_weight

        fast = bulk_apply(op, psi).values
        frame = np.asarray(op.full_transfer_kernel.apply(psi.values)) / W

        require(
            frame.shape == fast.shape,
            f"full_transfer_kernel output shape {frame.shape} != fast-path "
            f"{fast.shape} on LD flux (trailing φ̂ axis dropped/misplaced).",
        )
        np.testing.assert_allclose(
            frame, fast, rtol=1e-12, atol=1e-14,
            err_msg=(
                f"P{order}: (1/W)·full_transfer_kernel.apply(ψ) does NOT "
                f"reproduce the fast-path S.apply(ψ) on an LD (spatial-moment) "
                f"flux — the moment-scatter cell indexers mis-target the "
                f"trailing φ̂ axis (the #276/0b3275d (Ellipsis,*idx) → "
                f"(slice,slice,*idx) fix regressed)."
            ),
        )

    def test_ld_cells_index_fix_has_mutation_teeth(self, monkeypatch):
        r"""ISOLATION: the OLD ``(Ellipsis, *idx)`` indexing reddens on LD.

        Monkeypatch the moment-scatter verbs back to the pre-fix bodies and
        confirm the frame form fails on an LD flux (IndexError on the
        rectangular grid, or a wrong value) — the consumption proof that the
        cells-index fix is what the green above actually rests on.  Mutation
        in-process (monkeypatch); NEVER ``git checkout`` (uncommitted-state
        hazard, process-discipline rule).

        CS4c 3b-A re-point: the moment verbs moved from the
        ``MaterialXSField`` facade arms to the kernel fields — THIS sentinel
        is what caught the re-route (a mutation of the retired arms reddened
        nothing), so the surrogate patches the field verb. Since #426 step 2
        both channels ride the ONE verb ``TransferMaterialField.moment_source``
        (the (n,2n) twin ``moment_emission`` retired with the transfer
        family), so one surrogate covers the whole moment path.
        """
        from orpheus.transport.material_field import TransferMaterialField

        def _old_leg(self, moments, *, skip_l0, head=None):
            # ``head=`` arrived with #429 (the angular head is a parameter of
            # the moment verbs now); the surrogate accepts and IGNORES it —
            # the PRE-FIX arm transcribed here predates it and hard-codes the
            # rectangular layout, which is exactly what it must do to stand in
            # for the retired spelling.
            del head
            out = np.zeros_like(moments)
            l_start = 1 if skip_l0 else 0
            for mid, idx in self.cells_by_material.items():
                kern = self.per_material[mid]
                cells = (Ellipsis, *idx)  # PRE-FIX (buggy under LD)
                for l in range(l_start, self.order + 1):
                    n_m = 2 * l + 1
                    mv = moments[l, :n_m][cells]
                    out[l, :n_m][cells] = (
                        np.einsum("mfc...,fg->mgc...", mv, kern.moments[l])
                        + out[l, :n_m][cells]
                    )
            return out

        solver = _ld_solver_het(order=0)  # rectangular nx=4, ny=3
        op = solver.scattering_op
        psi = _ld_flux(solver)
        W = op.total_weight

        # Sanity: the FIXED code is clean before the mutation.
        np.testing.assert_allclose(
            np.asarray(op.full_transfer_kernel.apply(psi.values)) / W,
            bulk_apply(op, psi).values, rtol=1e-12, atol=1e-14,
            err_msg="precondition: fixed code must reproduce the fast-path on LD.",
        )

        monkeypatch.setattr(
            TransferMaterialField, "moment_source", _old_leg,
        )

        reddened = False
        try:
            mutated = np.asarray(op.full_transfer_kernel.apply(psi.values)) / W
            if not np.allclose(
                mutated, bulk_apply(op, psi).values,
                rtol=1e-9, atol=1e-12,
            ):
                reddened = True
        except (IndexError, ValueError):
            reddened = True  # wrong-axis over-run on the rectangular grid
        require(
            reddened,
            "The OLD (Ellipsis,*idx) indexing did NOT redden on an LD flux — "
            "the cells-index fix (0b3275d) has no teeth; the LD trailing-axis "
            "correctness is not actually being verified.",
        )
