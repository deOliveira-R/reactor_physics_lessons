r"""Tests for :class:`LegendreMomentTransfer`.

The §15.2 / §10 sum-of-tensor-products form
:math:`\Lambda = \sum_\ell \mathbf{P}_\ell \otimes \Sigma_{s,\ell}` on
moment space. The contract verified here:

* **Per-ℓ block-diagonal action**: scattering at order ℓ acts only on
  the ℓ-block of the moment tensor; off-block entries pass through
  unchanged (or, with skip_l0=True, zero out the ℓ=0 block).
* **Per-material partition**: only cells of material ``mid`` get
  scattered with ``sig_s[mid]``; cells of other materials are zero.
* **Energy contraction direction**: the operator contracts ``g_from``
  (matching the existing ``moment @ sig_s_l[l]`` convention used by
  :meth:`TransferOperator._redistribute_ordinates`).
* **Bit-identical against the legacy inlined math** for the case the
  legacy code computed (ℓ ≥ 1 only).
* **Predicates**: ``is_adjointable`` True (the per-ℓ group-transpose Λᵀ,
  campaign #276); ``is_invertible`` False (ℓ=0 block rank-deficient).
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.sn._test_helpers import material_xs_from_raw
from orpheus.transport.operators.transfer import LegendreMomentTransfer
from orpheus.numerics.basis.spherical_harmonic_basis import SphericalHarmonicBasis
from orpheus.transport.material_field import TransferMaterialField


def _make_simple_lambda(
    L: int = 2,
    nx: int = 3,
    ny: int = 1,
    ng: int = 2,
    n_materials: int = 2,
    seed: int = 0,
    skip_l0: bool = True,
) -> tuple[LegendreMomentTransfer, dict, dict]:
    """Build a small LegendreMomentTransfer instance for tests."""
    rng = np.random.default_rng(seed)
    # Per-material per-ℓ cross sections
    sig_s: dict[int, list[np.ndarray]] = {}
    for mid in range(n_materials):
        sig_s[mid] = [
            rng.uniform(0.0, 0.5, size=(ng, ng)) for _ in range(L + 1)
        ]
    # Cell partition: alternate cells by material
    ix_arr = np.arange(nx)
    iy_arr = np.zeros(nx, dtype=int)
    cells_by_mat: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for mid in range(n_materials):
        mask = (ix_arr % n_materials) == mid
        cells_by_mat[mid] = (ix_arr[mask], iy_arr[mask])
    mat_xs = material_xs_from_raw(
        sig_s=sig_s,
        cells_by_mat=cells_by_mat,
        ng=ng,
        nx=nx,
        ny=ny,
    )
    Lam = LegendreMomentTransfer.on_basis(
            TransferMaterialField.scattering(mat_xs), SphericalHarmonicBasis(L=L), skip_l0=skip_l0,
        )
    return Lam, sig_s, cells_by_mat


# ─────────────────────────────────────────────────────────────────────
# Structural predicates
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.l0
class TestPredicates:
    def test_apply_and_transpose_not_solve(self):
        # Λ is adjointable (the per-ℓ group-transpose Λᵀ, campaign #276) —
        # but NOT invertible (the ℓ=0 block is rank-deficient by design).
        # The transpose is the bare Euclidean Λᵀ that the frame-conjugated
        # kernel (R∘Λ∘M)ᵀ distributes onto.
        Lam, _, _ = _make_simple_lambda()
        assert Lam.is_adjointable
        assert not Lam.is_invertible


# ─────────────────────────────────────────────────────────────────────
# Per-ℓ block-diagonal action
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.l0
class TestPerEllBlockDiagonal:
    def test_skip_l0_zeroes_l0_block(self):
        Lam, _, _ = _make_simple_lambda(skip_l0=True)
        L = Lam.L
        rng = np.random.default_rng(seed=42)
        moments = rng.standard_normal((L + 1, 2 * L + 1, 2, 3, 1))
        out = Lam.apply(moments)
        np.testing.assert_array_equal(out[0, ...], 0.0)

    def test_no_skip_l0_includes_l0_block(self):
        Lam, sig_s, cells_by_mat = _make_simple_lambda(skip_l0=False)
        L = Lam.L
        rng = np.random.default_rng(seed=43)
        moments = rng.standard_normal((L + 1, 2 * L + 1, 2, 3, 1))
        out = Lam.apply(moments)
        # ℓ=0 block must be non-zero (random input, non-zero sig_s[0])
        assert not np.array_equal(out[0, ...], np.zeros_like(out[0, ...]))

    def test_each_ell_acts_on_its_own_block(self):
        """Setting one ℓ-block to zero in input → corresponding ℓ-block in output is zero."""
        Lam, _, _ = _make_simple_lambda(skip_l0=True)
        L = Lam.L
        rng = np.random.default_rng(seed=44)
        moments = rng.standard_normal((L + 1, 2 * L + 1, 2, 3, 1))
        # Zero out the ℓ=2 input block
        moments_modified = moments.copy()
        moments_modified[2, ...] = 0.0
        out_full = Lam.apply(moments)
        out_modified = Lam.apply(moments_modified)
        # ℓ=2 output block must differ; ℓ=1 must be unchanged.
        assert not np.array_equal(out_full[2, ...], out_modified[2, ...])
        np.testing.assert_array_equal(out_full[1, ...], out_modified[1, ...])


# ─────────────────────────────────────────────────────────────────────
# Per-material partition
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.l0
class TestPerMaterialPartition:
    def test_only_mat0_cells_scattered_with_sig_s_mat0(self):
        Lam, sig_s, cells_by_mat = _make_simple_lambda(
            n_materials=2, nx=4, skip_l0=True,
        )
        L = Lam.L
        rng = np.random.default_rng(seed=45)
        # Set ℓ=1 input to a constant for ALL cells
        moments = np.zeros((L + 1, 2 * L + 1, 2, 4, 1))
        moments[1, 1, :, :, 0] = 1.0  # ℓ=1, m=0, all cells, both groups
        out = Lam.apply(moments)
        # Cells 0, 2 belong to material 0; cells 1, 3 to material 1.
        ix_mat0, iy_mat0 = cells_by_mat[0]
        ix_mat1, iy_mat1 = cells_by_mat[1]
        # Output for mat-0 cells must equal sig_s[0][1].sum(axis=0)
        # (column-sum: out[g_to] = Σ_g_from sig_s[g_from, g_to] · 1.0)
        expected_mat0 = sig_s[0][1].sum(axis=0)
        expected_mat1 = sig_s[1][1].sum(axis=0)
        # Advanced indices (ix, iy) separated by ``:`` (ng axis) →
        # numpy moves the advanced-index axis to the FRONT, giving
        # ``(n_cells, ng)`` directly — no transpose needed.
        out_mat0 = out[1, 1, :, ix_mat0, iy_mat0]   # (n_cells_mat0, ng)
        out_mat1 = out[1, 1, :, ix_mat1, iy_mat1]   # (n_cells_mat1, ng)
        # Every row equals the expected per-material vector
        for row in out_mat0:
            np.testing.assert_allclose(row, expected_mat0, rtol=1e-15)
        for row in out_mat1:
            np.testing.assert_allclose(row, expected_mat1, rtol=1e-15)


# ─────────────────────────────────────────────────────────────────────
# Energy contraction direction (g_from contracted)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.l0
class TestEnergyContractionDirection:
    def test_contracts_g_from_axis(self):
        """Verify out[g_to] = Σ_g_from moments[g_from] · sig_s[g_from, g_to].

        Given a moments tensor with only g_from = 0 nonzero, output
        must be sig_s[0, :] (the row).
        """
        L = 1
        ng = 3
        sig_s_l1 = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])
        sig_s_l0 = np.zeros((ng, ng))
        mat_xs = material_xs_from_raw(
            sig_s={0: [sig_s_l0, sig_s_l1]},
            cells_by_mat={0: (np.array([0]), np.array([0]))},
            ng=ng,
            nx=1,
            ny=1,
        )
        Lam = LegendreMomentTransfer.on_basis(
            TransferMaterialField.scattering(mat_xs), SphericalHarmonicBasis(L=L), skip_l0=True,
        )
        # ℓ=1, m=0, single cell, only g_from=0 nonzero
        moments = np.zeros((L + 1, 2 * L + 1, ng, 1, 1))
        moments[1, 1, 0, 0, 0] = 1.0
        out = Lam.apply(moments)
        # Expected: out[1, 1, :, 0, 0] = sig_s_l1[0, :]
        np.testing.assert_allclose(
            out[1, 1, :, 0, 0], sig_s_l1[0, :], rtol=1e-15,
        )

    def test_two_groups_full_matrix(self):
        """Full 2-group cross-check on a (1, 0, 0, 0) input excitation."""
        ng = 2
        sig_s_l1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        mat_xs = material_xs_from_raw(
            sig_s={0: [np.zeros((ng, ng)), sig_s_l1]},
            cells_by_mat={0: (np.array([0]), np.array([0]))},
            ng=ng,
            nx=1,
            ny=1,
        )
        Lam = LegendreMomentTransfer.on_basis(
            TransferMaterialField.scattering(mat_xs), SphericalHarmonicBasis(L=1), skip_l0=True,
        )
        # All groups equal 1.0 in the (ℓ=1, m=0) slot
        moments = np.zeros((2, 3, ng, 1, 1))
        moments[1, 1, :, 0, 0] = np.array([1.0, 1.0])
        out = Lam.apply(moments)
        # out_g_to = Σ_g_from sig_s[g_from, g_to]
        # = column sums of sig_s_l1
        expected = sig_s_l1.sum(axis=0)  # = [4.0, 6.0]
        np.testing.assert_allclose(
            out[1, 1, :, 0, 0], expected, rtol=1e-15,
        )


# ─────────────────────────────────────────────────────────────────────
# Bit-identical to the legacy per-ℓ redistribution math
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.l0
@pytest.mark.regression
class TestBitIdenticalToLegacyInlinedMath:
    """L0: Λ.apply matches the legacy inline ``moment @ sig_s_l[l]`` step."""

    def test_legacy_inline_per_ell_per_mat_matches_lambda(self):
        L = 3
        nx, ny, ng = 4, 1, 3
        n_mat = 2
        rng = np.random.default_rng(seed=2026)
        sig_s = {
            mid: [rng.uniform(0, 0.5, size=(ng, ng)) for _ in range(L + 1)]
            for mid in range(n_mat)
        }
        ix_arr = np.arange(nx)
        cells_by_mat = {
            mid: (ix_arr[ix_arr % n_mat == mid], np.zeros(2, dtype=int))
            for mid in range(n_mat)
        }
        mat_xs = material_xs_from_raw(
            sig_s=sig_s, cells_by_mat=cells_by_mat,
            ng=ng, nx=nx, ny=ny,
        )
        Lam = LegendreMomentTransfer.on_basis(
            TransferMaterialField.scattering(mat_xs), SphericalHarmonicBasis(L=L), skip_l0=True,
        )
        moments = rng.standard_normal((L + 1, 2 * L + 1, ng, nx, ny))

        # Legacy reference: inline ``moment @ sig_s_l[l]`` per (mid, l, m).
        # PR-INDEX-5: moments principled (L+1, 2L+1, ng, nx, ny); advanced
        # indices (ix, iy) separated by ``:`` → numpy puts advanced axis
        # to the front, so ``moments[l, l+m, :, ix, iy]`` is
        # ``(n_cells, ng)`` — exactly the legacy shape, no transpose.
        legacy = np.zeros_like(moments)
        for mid, (ix, iy) in cells_by_mat.items():
            sig_s_l = sig_s[mid]
            for l in range(1, L + 1):
                for m in range(-l, l + 1):
                    moment = moments[l, l + m, :, ix, iy]    # (n_cells, ng)
                    scattered = moment @ sig_s_l[l]          # (n_cells, ng)
                    legacy[l, l + m, :, ix, iy] = scattered  # numpy infers shapes

        out = Lam.apply(moments)
        np.testing.assert_allclose(out, legacy, rtol=1e-15, atol=0.0)


# ─────────────────────────────────────────────────────────────────────
# Composition (operator algebra)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.l0
class TestComposesUnderOperatorAlgebra:
    def test_lambda_composes_under_operator_product(self):
        """Λ should compose correctly with other LinearOperators via `@`."""
        from orpheus.numerics.operator import IdentityOperator
        Lam, _, _ = _make_simple_lambda()
        # Λ @ I  — composition machinery (the ctor apply-guard admits Λ).
        composed = Lam @ IdentityOperator()
        assert callable(getattr(composed, "apply", None))
