r"""Verification of the factored :class:`BoundaryClosureOperator`.

Created 2026-04-18. The boundary-closure kernel :math:`K_{\mathrm{bc}}`
factors as a tensor network

.. math::

   K_{\mathrm{bc}} \;=\; G \cdot R \cdot P

through a finite mode space :math:`A = \mathbb R^N` (see
:ref:`theory-peierls-unified` Part IV). This file checks:

1. **Algebraic consistency** between :meth:`apply` and :meth:`as_matrix`
   (matrix-free application matches the dense-matrix product for random
   source vectors — the fundamental contract of the factorisation).

2. **Rank structure** — :attr:`closure_rank` matches :math:`\mathrm{rank}(R)`
   for the canonical vacuum / Mark / Marshak reflection operators.

3. **BC-as-choice-of-R**: the Marshak DP\ :sub:`N-1` correction equals
   the result of :func:`build_white_bc_correction_rank_n` (thin wrapper),
   Mark matches legacy :func:`build_white_bc_correction` at
   ``n_bc_modes = 1``, and vacuum gives the zero matrix.

4. **Factored storage efficiency** — for :math:`N_r \gg N`, the
   three-tensor representation is substantially smaller than the
   :math:`N_r \times N_r` dense matrix.

These are **foundation** tests (software-invariant contracts of the
factored representation), not equation-level verification; no
``verifies()`` label is attached.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    BoundaryClosureOperator,
    CYLINDER_1D,
    SPHERE_1D,
    build_closure_operator,
    build_white_bc_correction,
    build_white_bc_correction_rank_n,
    composite_gl_r,
    reflection_mark,
    reflection_marshak,
    reflection_vacuum,
)


_SIG_T = np.array([1.0])
_GEOMETRIES = [
    pytest.param(CYLINDER_1D, id="cylinder-1d"),
    pytest.param(SPHERE_1D, id="sphere-1d"),
]


def _build(R: float, p_order: int = 5, n_panels: int = 2):
    radii = np.array([R])
    r_nodes, r_wts, _panels = composite_gl_r(radii, n_panels, p_order, dps=25)
    return radii, r_nodes, r_wts


# ═══════════════════════════════════════════════════════════════════════
# 1. Algebraic consistency: apply(q) = as_matrix() @ q
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.foundation
@pytest.mark.parametrize("geometry", _GEOMETRIES)
@pytest.mark.parametrize("R", [1.0, 5.0])
@pytest.mark.parametrize("n_bc_modes", [1, 2, 3])
def test_apply_matches_as_matrix(geometry, R, n_bc_modes):
    r"""The factored application :math:`G\,R\,P\,q` must equal the
    product :math:`(G R P)\,q` to machine precision.

    This is the **definitional contract** of
    :class:`BoundaryClosureOperator`: :meth:`apply` and
    :meth:`as_matrix` are two implementations of the same linear map
    and must agree bitwise modulo numpy's default matrix-multiply
    round-off (rtol = 1e-13 is a conservative bound on three matmul
    accumulations).
    """
    radii, r_nodes, r_wts = _build(R)
    op = build_closure_operator(
        geometry, r_nodes, r_wts, radii, _SIG_T,
        n_angular=24, n_surf_quad=24, dps=25,
        n_bc_modes=n_bc_modes, reflection="marshak",
    )

    K_dense = op.as_matrix()
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        q = rng.normal(size=op.n_nodes)
        applied = op.apply(q)
        dense = K_dense @ q
        np.testing.assert_allclose(
            applied, dense, rtol=1e-13, atol=1e-13,
            err_msg=(
                f"[{geometry.kind} R={R} N={n_bc_modes}] "
                f"apply vs as_matrix disagree; max |Δ| = "
                f"{np.abs(applied - dense).max():.3e}"
            ),
        )


# ═══════════════════════════════════════════════════════════════════════
# 2. Rank structure vs R matrix
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.foundation
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_closure_rank_matches_reflection_rank(geometry):
    r"""Generically, :math:`\mathrm{rank}(K_{\mathrm{bc}}) =
    \mathrm{rank}(R)` because :math:`P` and :math:`G` have full mode
    rank for a non-degenerate radial grid.

    - Vacuum: :math:`R = 0 \Rightarrow K_{\mathrm{bc}} = 0`, rank 0.
    - Mark:  :math:`R = e_0 e_0^{\top} \Rightarrow` rank 1.
    - Marshak rank-:math:`N`: :math:`R` diagonal with nonzero
      entries :math:`\Rightarrow` rank :math:`N`.
    """
    radii, r_nodes, r_wts = _build(R=5.0)

    # Vacuum
    op_vac = build_closure_operator(
        geometry, r_nodes, r_wts, radii, _SIG_T,
        n_bc_modes=3, reflection="vacuum",
    )
    assert op_vac.closure_rank == 0
    assert np.allclose(op_vac.as_matrix(), 0.0, atol=1e-14)

    # Mark (rank-1)
    op_mark = build_closure_operator(
        geometry, r_nodes, r_wts, radii, _SIG_T,
        n_bc_modes=3, reflection="mark",
    )
    assert op_mark.closure_rank == 1

    # Marshak rank-3
    op_marshak = build_closure_operator(
        geometry, r_nodes, r_wts, radii, _SIG_T,
        n_bc_modes=3, reflection="marshak",
    )
    assert op_marshak.closure_rank == 3


# ═══════════════════════════════════════════════════════════════════════
# 3. BC-as-choice-of-R consistency
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.foundation
@pytest.mark.parametrize("geometry", _GEOMETRIES)
@pytest.mark.parametrize("R", [1.0, 5.0, 10.0])
def test_marshak_matches_build_white_bc_correction_rank_n(geometry, R):
    """``as_matrix`` of the Marshak closure matches the thin wrapper.

    This gates that :func:`build_white_bc_correction_rank_n` remains
    structurally ``BoundaryClosureOperator(reflection="marshak").as_matrix()``;
    any future rewrite that doesn't preserve this identity would be
    caught here.
    """
    radii, r_nodes, r_wts = _build(R)
    for n_bc_modes in (1, 2, 3):
        op = build_closure_operator(
            geometry, r_nodes, r_wts, radii, _SIG_T,
            n_bc_modes=n_bc_modes, reflection="marshak",
        )
        K_thin = build_white_bc_correction_rank_n(
            geometry, r_nodes, r_wts, radii, _SIG_T,
            n_bc_modes=n_bc_modes,
        )
        np.testing.assert_allclose(
            op.as_matrix(), K_thin, rtol=1e-14, atol=1e-15,
        )


@pytest.mark.foundation
@pytest.mark.parametrize("geometry", _GEOMETRIES)
@pytest.mark.parametrize("R", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_mark_rank1_equals_legacy_build_white_bc_correction(geometry, R):
    r"""At ``n_bc_modes = 1``, Mark and Marshak both reduce to the
    legacy :func:`build_white_bc_correction` bit-exactly.

    The :math:`R_{00} = 1` entry is the only surviving mode; the
    mode-0 :math:`P` and :math:`G` tensors use the legacy
    :func:`compute_P_esc` / :func:`compute_G_bc` integrands (not the
    ``_mode`` Jacobian-weighted versions), so the product
    :math:`G R P = u_0 v_0^{\top}` matches the legacy rank-1 form
    exactly.
    """
    radii, r_nodes, r_wts = _build(R)
    K_legacy = build_white_bc_correction(
        geometry, r_nodes, r_wts, radii, _SIG_T,
        n_angular=32, n_surf_quad=32, dps=25,
    )

    for reflection in ("mark", "marshak"):
        op = build_closure_operator(
            geometry, r_nodes, r_wts, radii, _SIG_T,
            n_angular=32, n_surf_quad=32, dps=25,
            n_bc_modes=1, reflection=reflection,
        )
        np.testing.assert_allclose(
            op.as_matrix(), K_legacy, rtol=1e-14, atol=1e-15,
            err_msg=(
                f"[{geometry.kind} R={R}] reflection={reflection!r} "
                f"rank-1 does not match legacy build_white_bc_correction."
            ),
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. Reflection-operator constructors
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.foundation
def test_reflection_vacuum_is_zero():
    R = reflection_vacuum(5)
    assert np.allclose(R, 0.0)


@pytest.mark.foundation
def test_reflection_mark_projects_onto_scalar_mode():
    R = reflection_mark(5)
    expected = np.zeros((5, 5))
    expected[0, 0] = 1.0
    np.testing.assert_array_equal(R, expected)
    assert np.linalg.matrix_rank(R) == 1


@pytest.mark.foundation
def test_reflection_marshak_has_2nplus1_diagonal():
    R = reflection_marshak(5)
    expected = np.diag([1.0, 3.0, 5.0, 7.0, 9.0])
    np.testing.assert_array_equal(R, expected)
    assert np.linalg.matrix_rank(R) == 5


# ═══════════════════════════════════════════════════════════════════════
# 5. Custom reflection matrix (user-supplied)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.foundation
def test_custom_reflection_matrix_albedo():
    r"""A user-supplied reflection matrix :math:`R = \alpha \cdot
    R_{\mathrm{Marshak}}` (albedo BC) is accepted and scales the
    kernel linearly in :math:`\alpha`.

    This exercises the non-string ``reflection`` argument pathway
    and the general tensor-network flexibility — any BC physics
    expressible as a mode-space matrix can be dropped in without
    touching :math:`P` or :math:`G`.
    """
    radii, r_nodes, r_wts = _build(R=5.0)
    alpha = 0.5

    op_marshak = build_closure_operator(
        CYLINDER_1D, r_nodes, r_wts, radii, _SIG_T,
        n_bc_modes=3, reflection="marshak",
    )
    op_albedo = build_closure_operator(
        CYLINDER_1D, r_nodes, r_wts, radii, _SIG_T,
        n_bc_modes=3, reflection=alpha * reflection_marshak(3),
    )
    np.testing.assert_allclose(
        op_albedo.as_matrix(), alpha * op_marshak.as_matrix(),
        rtol=1e-14, atol=1e-15,
    )


# ═══════════════════════════════════════════════════════════════════════
# 6. Storage claim: factored form is smaller for N_r >> N
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.foundation
def test_factored_storage_scales_as_NrN():
    r"""For :math:`N_r \gg N`, the factored form stores
    :math:`O(N_r N + N^2)` floats vs :math:`O(N_r^2)` for the dense
    matrix. This test reports the ratio for a realistic case
    (:math:`N_r = 50`, :math:`N = 4`) and asserts it is substantially
    less than 1.
    """
    radii, r_nodes, r_wts = _build(R=5.0, p_order=25, n_panels=1)
    n_bc_modes = 4
    op = build_closure_operator(
        CYLINDER_1D, r_nodes, r_wts, radii, _SIG_T,
        n_bc_modes=n_bc_modes, reflection="marshak",
    )

    factored_floats = (
        op.P.size + op.G.size + op.R.size
    )
    dense_floats = op.n_nodes * op.n_nodes
    ratio = factored_floats / dense_floats
    assert ratio < 0.5, (
        f"Factored storage ratio = {ratio:.3f} (factored "
        f"{factored_floats} vs dense {dense_floats}); expected < 0.5 "
        f"for N_r={op.n_nodes}, N={n_bc_modes}."
    )
