"""Diagnostic: SYNTHESIS — plateau / overshoot per geometry.

Created by numerics-investigator on 2026-04-28.

PROMOTE-CANDIDATE: pins these characteristic regression numbers:
  - SPHERE thin (R=5, σ=0.5, fuel-A-like XS):
      MB k_eff overshoots k_inf at N ≥ 4
      (N=4: +0.06%, N=8: +1.6%) — divergent matrix construction.
  - CYL thin (R=5, σ=0.5):
      MB k_eff overshoots k_inf at N ≥ 4
      (N=4: +0.03%, N=8: +1.3%) — bare specular itself drifts toward
      and past k_inf at high N due to R-conditioning blowup.
  - SLAB thin (L=5, σ=0.5):
      MB k_eff CONVERGES MONOTONICALLY to a small undershoot ≈ -0.16%
      across N = 1..20 — NO overshoot, NO divergence. Slab MB is the
      ONLY geometry where the matrix-Galerkin form converges as N → ∞.

GEOMETRIC ROOT CAUSE
--------------------
Sphere:  chord(µ) = 2Rµ.       At grazing µ→0, transmission → 1, RESOLVENT
                                continuous-limit DIVERGES at µ=0.
Cyl   :  in-plane d_2D(α) = 2R cos α. At grazing α→π/2, chord→0,
                                Ki_3(0) finite, but the cos α partial-
                                current factor multiplies in → finite
                                T_op^cyl. Resolvent BOUNDED but R is
                                ill-conditioned at high N → bare and
                                MB both drift past k_inf.
Slab  :  chord(µ) = L/µ.       At grazing µ→0, chord→∞, transmission
                                → 0, T_op^slab → 0 → resolvent BOUNDED
                                with very small spectrum. Slab MB
                                converges cleanly.
"""
from __future__ import annotations

import numpy as np
import pytest
import warnings

from orpheus.derivations.peierls_geometry import (
    SLAB_POLAR_1D,
    SPHERE_1D,
    _build_full_K_per_group,
    composite_gl_r,
    reflection_specular,
)

from derivations.diagnostics.diag_specular_mb_phase4_03_pathology_resolvent import (
    build_T_slab,
)
from derivations.diagnostics.diag_specular_mb_phase4_06_keff_endtoend import (
    _build_slab_PG_perface,
    keff_from_K,
    SIG_T, SIG_S, NU_SIG_F, K_INF,
)


def test_sphere_overshoot_pinned():
    """Pin: sphere MB at thin τ_R=2.5 overshoots k_inf at N=4."""
    R = 5.0
    radii = np.array([R])
    sig_t_g = np.array([SIG_T])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, 2, 5, dps=20, inner_radius=0.0,
    )
    sig_s_arr = np.full(len(r_nodes), SIG_S)
    nu_sf_arr = np.full(len(r_nodes), NU_SIG_F)
    sig_t_arr = np.full(len(r_nodes), SIG_T)

    # N=2 must undershoot, N=8 must overshoot
    K2 = _build_full_K_per_group(
        SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
        "specular_multibounce",
        n_angular=24, n_rho=24, n_surf_quad=24,
        n_bc_modes=2, dps=20,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        K8 = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "specular_multibounce",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=8, dps=20,
        )
    k2 = keff_from_K(K2, sig_t_arr, nu_sf_arr, sig_s_arr)
    k8 = keff_from_K(K8, sig_t_arr, nu_sf_arr, sig_s_arr)
    assert k2 < K_INF, f"Sphere MB N=2 should undershoot: k_2={k2}"
    assert k8 > K_INF, f"Sphere MB N=8 should OVERSHOOT: k_8={k8} vs k_inf={K_INF}"


def test_slab_mb_monotonic_convergence(capsys):
    """Pin: slab MB at thin τ_L=2.5 converges monotonically with N
    (no overshoot)."""
    with capsys.disabled():
        L = 5.0
        radii = np.array([L])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 1, 5, dps=15, inner_radius=0.0,
        )
        sig_s_arr = np.full(len(r_nodes), SIG_S)
        nu_sf_arr = np.full(len(r_nodes), NU_SIG_F)
        sig_t_arr = np.full(len(r_nodes), SIG_T)

        # Build K_vol once.
        K_full_n1 = _build_full_K_per_group(
            SLAB_POLAR_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "specular",
            n_angular=12, n_rho=12, n_surf_quad=12,
            n_bc_modes=1, dps=15,
        )
        P1, G1, _ = _build_slab_PG_perface(L, SIG_T, r_nodes, r_wts, 1)
        R1_face = reflection_specular(1)
        R1_slab = np.zeros((2, 2)); R1_slab[:1, :1] = R1_face; R1_slab[1:, 1:] = R1_face
        K_vol = K_full_n1 - G1 @ R1_slab @ P1

        keffs = []
        for N in (1, 2, 4, 8, 16, 20):
            P, G, _ = _build_slab_PG_perface(L, SIG_T, r_nodes, r_wts, N)
            R_face = reflection_specular(N)
            R_slab = np.zeros((2 * N, 2 * N))
            R_slab[:N, :N] = R_face; R_slab[N:, N:] = R_face
            T = build_T_slab(SIG_T, L, N)
            ITR = np.eye(2 * N) - T @ R_slab
            K_mb = K_vol + G @ R_slab @ np.linalg.solve(ITR, P)
            k = keff_from_K(K_mb, sig_t_arr, nu_sf_arr, sig_s_arr)
            keffs.append((N, k))
            print(f"  N={N:2d}: k_mb={k:.6f}, rel={(k-K_INF)/K_INF*100:+.4f}%")
        # All k_eff < k_inf (no overshoot)
        for N, k in keffs:
            assert k < K_INF, f"Slab MB N={N} OVERSHOT: k={k} vs k_inf={K_INF}"
        # All k_eff > k_eff(N=1) (monotonic toward k_inf)
        for i in range(1, len(keffs)):
            assert keffs[i][1] >= keffs[i-1][1] - 1e-8, (
                f"Slab MB non-monotonic at N={keffs[i][0]}: "
                f"k={keffs[i][1]} < prev={keffs[i-1][1]}"
            )


def test_cyl_mb_overshoots_at_thin(capsys):
    """Pin: cylinder MB at thin τ_R=2.5 overshoots k_inf at N=4
    (regime-persistent pathology, same as sphere)."""
    with capsys.disabled():
        from derivations.diagnostics.diag_specular_mb_phase4_09_cyl_robustness import (
            _cyl_keff_sweep_N,
        )
        results = _cyl_keff_sweep_N(R=5.0, sig_t=0.5, N_list=(2, 8))
        ks = {N: k for N, k, _ in results}
        assert ks[2] < K_INF, f"Cyl MB N=2 should undershoot: k_2={ks[2]}"
        assert ks[8] > K_INF, (
            f"Cyl MB N=8 should OVERSHOOT: k_8={ks[8]} vs k_inf={K_INF}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
