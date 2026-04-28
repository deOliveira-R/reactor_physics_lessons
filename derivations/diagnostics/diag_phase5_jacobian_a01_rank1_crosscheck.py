"""Diagnostic: Phase 5+ Front A — Sanchez↔ORPHEUS K_bc Jacobian conversion

Created by numerics-investigator on 2026-04-28.

GOAL
====
Determine whether the kernel produced by
:func:`compute_K_bc_specular_continuous_mu_sphere` (Sanchez 1986 Eq. (A6)
verbatim, with α=1, β=0, ω₁=0) is related to ORPHEUS's
``closure="white_hebert"`` rank-1 K_bc by a SEPARABLE Jacobian factor

    K_bc^Hebert[i,j] = α(r_i, r_j) · K_bc^Phase5[i,j]
    α(r_i, r_j) ?= α_recv(r_i) · α_src(r_j)

The Phase 5a closeout established that Sanchez's `g_h(ρ' → ρ)` includes
a `4π·ρ'²` surface-area Jacobian baked in via Sanchez Eq. (2), while
ORPHEUS uses explicit `rv = 4π·r²` radial-volume weights and `r_wts`
quadrature weights in the Nyström convention.

DECISION CRITERION
==================
- Separable to <1e-3 rel error → identify α_recv, α_src, ship a
  wrapper closure="specular_continuous_mu" for sphere homogeneous.
- Separable to ~1e-2 → likely a simple-but-quadrature-limited mapping;
  identify factors and document residual error.
- Non-separable → Front A is dead, document for parent.

REPRODUCER
==========
- Homogeneous sphere: 3 thicknesses τ_R ∈ {2.5, 5, 10}
- A 4-panel composite GL mesh on r ∈ [0, R]
- Phase5 K_bc at n_quad = 64 and n_quad = 128
- Hebert K_bc at the same r_nodes

Run:
    python -m pytest derivations/diagnostics/diag_phase5_jacobian_a01_rank1_crosscheck.py -v -s
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    composite_gl_r,
    build_volume_kernel,
    compute_K_bc_specular_continuous_mu_sphere,
    _build_full_K_per_group,
)


# ───────────────────────── Helper ──────────────────────────

_QUAD = dict(
    n_panels_per_region=2,
    p_order=4,
    n_angular=24,
    n_rho=24,
    n_surf_quad=64,
    dps=15,
)


def _build_kbc_pair(R: float, sig_t_val: float, n_quad: int = 64):
    """Build K_bc^Hebert and K_bc^Phase5 at the same r_nodes.

    Returns
    -------
    r_nodes : ndarray
    K_bc_hebert : ndarray (N, N)
    K_bc_phase5 : ndarray (N, N)
    """
    radii = np.array([R], dtype=float)
    sig_t = np.array([sig_t_val], dtype=float)

    r_nodes, r_wts, panels = composite_gl_r(
        radii,
        n_panels_per_region=_QUAD["n_panels_per_region"],
        p_order=_QUAD["p_order"],
        dps=_QUAD["dps"],
    )

    # Full K = K_vol + K_bc_hebert at same r_nodes
    K_full_hebert = _build_full_K_per_group(
        SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t,
        closure="white_hebert",
        n_angular=_QUAD["n_angular"],
        n_rho=_QUAD["n_rho"],
        n_surf_quad=_QUAD["n_surf_quad"],
        n_bc_modes=1,
        dps=_QUAD["dps"],
    )

    # K_vol alone (subtract for clean K_bc^Hebert)
    K_vol = build_volume_kernel(
        SPHERE_1D, r_nodes, panels, radii, sig_t,
        n_angular=_QUAD["n_angular"],
        n_rho=_QUAD["n_rho"],
        dps=_QUAD["dps"],
    )

    K_bc_hebert = K_full_hebert - K_vol

    # Phase 5 reference kernel
    K_bc_phase5 = compute_K_bc_specular_continuous_mu_sphere(
        r_nodes, radii, sig_t, n_quad=n_quad,
    )

    return r_nodes, r_wts, K_bc_hebert, K_bc_phase5


def _check_separable(R_mat: np.ndarray, atol_rel: float = 1e-2):
    """Check if R[i, j] = a(i) * b(j) is separable.

    Method: SVD. If R is rank-1, σ_2/σ_1 << 1.
    Returns: rank1_residual (relative), a, b.
    """
    U, S, Vt = np.linalg.svd(R_mat, full_matrices=False)
    if S[0] == 0.0:
        return np.inf, None, None
    rank1_residual = float(S[1] / S[0]) if len(S) >= 2 else 0.0
    a = U[:, 0] * S[0]
    b = Vt[0, :]
    # Fix sign convention (a positive)
    if a.sum() < 0:
        a = -a
        b = -b
    return rank1_residual, a, b


# ───────────────────────── Tests ────────────────────────────

@pytest.mark.parametrize("R, sig_t_val, label", [
    (5.0, 0.5, "thin τ_R=2.5"),
    (5.0, 1.0, "medium τ_R=5"),
    (5.0, 2.0, "thick τ_R=10"),
])
def test_a01_ratio_matrix_separability(R, sig_t_val, label):
    """Compute K_bc^Hebert / K_bc^Phase5 and test separability."""
    r_nodes, r_wts, K_bc_hebert, K_bc_phase5 = _build_kbc_pair(
        R, sig_t_val, n_quad=64,
    )
    print(f"\n=== {label} (R={R}, σ_t={sig_t_val}, τ_R={R*sig_t_val}) ===")
    print(f"r_nodes ({len(r_nodes)}): {r_nodes}")
    print(f"\n[Hebert K_bc magnitudes]:")
    print(f"  min={K_bc_hebert.min():.3e}, max={K_bc_hebert.max():.3e}, "
          f"mean={K_bc_hebert.mean():.3e}")
    print(f"\n[Phase5 K_bc magnitudes]:")
    print(f"  min={K_bc_phase5.min():.3e}, max={K_bc_phase5.max():.3e}, "
          f"mean={K_bc_phase5.mean():.3e}")

    # Compute ratio matrix where K_bc_phase5 != 0
    mask = np.abs(K_bc_phase5) > 1e-15
    R_mat = np.zeros_like(K_bc_phase5)
    R_mat[mask] = K_bc_hebert[mask] / K_bc_phase5[mask]

    print(f"\n[Ratio K_bc^Hebert / K_bc^Phase5]:")
    print(f"  shape={R_mat.shape}, min={R_mat[mask].min():.3e}, "
          f"max={R_mat[mask].max():.3e}, mean={R_mat[mask].mean():.3e}")
    print(f"  ratio max/min = {R_mat[mask].max() / R_mat[mask].min():.3e}")
    print(f"\n[Ratio matrix sample (first 5x5)]:")
    print(R_mat[:5, :5])

    # SVD separability test
    rank1_res, a_recv, b_src = _check_separable(R_mat)
    print(f"\n[SVD separability]")
    print(f"  σ_2/σ_1 = {rank1_res:.3e}  (rank-1 residual)")
    if a_recv is not None:
        print(f"  α_recv(r_i): {a_recv}")
        print(f"  α_src(r_j): {b_src}")
        # Test against simple candidates
        # Candidate 1: α_src ∝ r²
        ratio_to_rsq = b_src / (r_nodes**2)
        print(f"  α_src / r² (should be ~const if α_src = c·r²): "
              f"{ratio_to_rsq}")
        ratio_recv_const = a_recv  # See if ~const
        print(f"  α_recv (just listed; should be ~const "
              f"if α_recv = const): {a_recv}")

    # No assertion — this is investigative. Save the result for the next test.
    # Just store data
    test_a01_ratio_matrix_separability.last_result = {
        'R': R, 'sig_t': sig_t_val, 'r_nodes': r_nodes,
        'K_bc_hebert': K_bc_hebert, 'K_bc_phase5': K_bc_phase5,
        'R_mat': R_mat, 'rank1_residual': rank1_res,
        'a_recv': a_recv, 'b_src': b_src,
    }


@pytest.mark.parametrize("R, sig_t_val", [
    (5.0, 0.5),
    (5.0, 1.0),
    (5.0, 2.0),
])
def test_a02_factor_identification(R, sig_t_val):
    """Once separable, identify α_recv(r), α_src(r) functional forms.

    Educated guesses to test:
    - α_src ∝ r_j² · w_j (volume weight in ORPHEUS Nyström convention)
    - α_recv ∝ Σ_t / R² (surface area divisor) or ∝ Σ_t
    - Possibly involves 1/(1-P_ss) (the Hebert geometric series)
    """
    r_nodes, r_wts, K_bc_hebert, K_bc_phase5 = _build_kbc_pair(
        R, sig_t_val, n_quad=64,
    )
    mask = np.abs(K_bc_phase5) > 1e-15
    R_mat = np.zeros_like(K_bc_phase5)
    R_mat[mask] = K_bc_hebert[mask] / K_bc_phase5[mask]

    rank1_res, a_recv, b_src = _check_separable(R_mat)
    print(f"\n=== Factor ID R={R}, σ_t={sig_t_val}, τ_R={R*sig_t_val} ===")
    print(f"  r_nodes: {r_nodes}")
    print(f"  r_wts: {r_wts}")
    print(f"  rank1_residual = {rank1_res:.3e}")
    if rank1_res > 0.05:
        pytest.skip(f"Not rank-1 separable (residual {rank1_res:.3e})")

    # Test a = α_recv(r_i)
    print(f"\n[α_recv(r_i) profile]:")
    print(f"  Values: {a_recv}")
    # Check if proportional to constants we can interpret
    # Hebert form: u_i = sig_t * G_bc(r_i) / divisor
    #              v_j = rv(r_j) * w_j * P_esc(r_j)
    #              K_bc_hebert = u_i * v_j / (1 - P_ss)
    # Phase5 form: K_bc_phase5 = whatever Sanchez gives directly
    #
    # If structure separable as α_recv * α_src then
    #   α_recv ∝ u_i  (via Hebert structure)
    #   α_src ∝ v_j (via Hebert structure)
    # but that's only true if Phase5 K_bc[i,j] is ALSO separable in (i,j).
    # Otherwise, α_recv/α_src capture residual differences.

    # Compute Hebert's u/v factors directly to compare
    from orpheus.derivations.peierls_geometry import (
        compute_P_esc, compute_G_bc, compute_P_ss_sphere,
    )
    radii = np.array([R], dtype=float)
    sig_t = np.array([sig_t_val], dtype=float)
    P_esc_n = compute_P_esc(SPHERE_1D, r_nodes, radii, sig_t,
                             n_angular=_QUAD["n_angular"],
                             dps=_QUAD["dps"])
    G_bc_n = compute_G_bc(SPHERE_1D, r_nodes, radii, sig_t,
                           n_surf_quad=_QUAD["n_surf_quad"],
                           dps=_QUAD["dps"])
    P_ss = compute_P_ss_sphere(radii, sig_t, n_quad=_QUAD["n_surf_quad"],
                                dps=_QUAD["dps"])
    print(f"  P_esc(r_i): {P_esc_n}")
    print(f"  G_bc(r_i):  {G_bc_n}")
    print(f"  P_ss = {P_ss}")
    # Hebert u_i = sig_t * G_bc / R²; v_j = r_j² * w_j * P_esc
    u_i = sig_t_val * G_bc_n / (R * R)
    v_j = r_nodes**2 * r_wts * P_esc_n
    K_bc_hebert_factored = np.outer(u_i, v_j) / (1.0 - P_ss)
    print(f"\n[Hebert factored form vs full form]:")
    print(f"  max relative diff: "
          f"{np.max(np.abs(K_bc_hebert_factored - K_bc_hebert)) / np.max(np.abs(K_bc_hebert)):.3e}")

    # Test if Phase5 K_bc is ALSO factorable into u'_i v'_j
    rank1_phase5, u5, v5 = _check_separable(K_bc_phase5)
    print(f"\n[Phase5 K_bc separability]:")
    print(f"  σ_2/σ_1 = {rank1_phase5:.3e}")
    print(f"  u5(r_i): {u5}")
    print(f"  v5(r_j): {v5}")

    # If both Hebert and Phase5 are rank-1 outer products, then ratio is also
    # If Phase5 is NOT rank-1, then it carries (i,j)-dependent structure
    # that breaks separability.

    # Quantitative ratios
    if rank1_phase5 < 0.05:
        print(f"\n[Both rank-1: comparing u/u5 and v/v5]:")
        # Sign convention
        if u5.sum() < 0:
            u5 = -u5
            v5 = -v5
        u_ratio = u_i / u5
        v_ratio = v_j / v5
        print(f"  u_i / u5 (should be ~const if α_recv = const): {u_ratio}")
        print(f"  v_j / v5 (should be ~const if α_src = const): {v_ratio}")
        print(f"  product (u/u5)(v/v5) at every (i,j): mean ratio "
              f"= {(u_ratio[:, None] * v_ratio[None, :]).mean():.3e}")


@pytest.mark.parametrize("R, sig_t_val", [
    (5.0, 0.5),
    (5.0, 1.0),
    (5.0, 2.0),
])
def test_a03_quadrature_convergence(R, sig_t_val):
    """Check Phase 5 kernel converges with n_quad."""
    r_nodes, _, _, K64 = _build_kbc_pair(R, sig_t_val, n_quad=64)
    _, _, _, K128 = _build_kbc_pair(R, sig_t_val, n_quad=128)
    _, _, _, K256 = _build_kbc_pair(R, sig_t_val, n_quad=256)

    diff_64_128 = np.max(np.abs(K128 - K64)) / np.max(np.abs(K128))
    diff_128_256 = np.max(np.abs(K256 - K128)) / np.max(np.abs(K256))
    print(f"\n=== Phase5 quadrature convergence (R={R}, σ_t={sig_t_val}) ===")
    print(f"  ‖K128 − K64 ‖ rel  = {diff_64_128:.3e}")
    print(f"  ‖K256 − K128‖ rel  = {diff_128_256:.3e}")
