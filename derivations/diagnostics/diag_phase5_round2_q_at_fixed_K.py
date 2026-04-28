"""Round 2 follow-up — Q-convergence at FIXED K_max.

The intuition: each individual K_bc^(k) (bounded integrand) should
Q-converge spectrally. The SUM converges to white_hebert as K_max→∞,
but the SUM at K_max=∞ inherits the singularity. Test:

- Per-bounce K_bc^(k) Q-convergence (small k=0,1,2)
- Truncated K_max=K_FIX Q-convergence
- Spot the K_max where Q stops converging
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _build_full_K_per_group,
    composite_gl_r,
)

from derivations.diagnostics.diag_phase5_round2_m2_bounce_resolved import (
    K_INF,
    NU_SIG_F,
    R_THIN,
    SIG_S,
    SIG_T,
    compute_K_bc_M2_bounce_sphere,
    keff_from_K,
)


def test_q_convergence_per_bounce_k0_k1_k2(capsys):
    """For k=0, 1, 2: K_bc^(k) at Q=16,32,64,128,256,512."""
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, _ = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=0.0,
        )

        for k_target in (0, 1, 2, 5):
            print(f"\n=== Q-conv per-bounce k={k_target} (weight=no_mu) ===")
            print(f"  Q   | ||K^({k_target})||_F     | rel_diff_to_Q=512")
            K_ref = None
            results = []
            for Q in (16, 32, 64, 128, 256, 512):
                _, K_list = compute_K_bc_M2_bounce_sphere(
                    SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                    n_quad=Q, K_max=k_target, weight_form="no_mu",
                    return_per_bounce=True,
                )
                Kk = K_list[k_target]
                results.append((Q, Kk))
            K_ref = results[-1][1]
            for Q, Kk in results:
                fr = float(np.linalg.norm(Kk))
                rel = float(np.linalg.norm(Kk - K_ref) / np.linalg.norm(K_ref))
                print(f"  {Q:3d} | {fr:18.6e} | {rel:.6e}")


def test_q_convergence_at_fixed_K_max(capsys):
    """k_eff vs Q at fixed K_max ∈ {0, 1, 2, 5, 10, 20, 50}."""
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=0.0,
        )
        sig_s_arr = np.full(len(r_nodes), SIG_S)
        nu_sf_arr = np.full(len(r_nodes), NU_SIG_F)
        sig_t_arr = np.full(len(r_nodes), SIG_T)

        K_vol = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        K_heb = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "white_hebert",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        k_heb = keff_from_K(K_heb, sig_t_arr, nu_sf_arr, sig_s_arr)
        print(f"\n  white_hebert: k_eff = {k_heb:.6f} ({(k_heb-K_INF)/K_INF*100:+.4f}%)")

        for K_max in (0, 1, 2, 5, 10):
            print(f"\n=== Q convergence at K_max={K_max} (weight=no_mu) ===")
            print(f"  Q   | k_eff       | rel_inf   | rel_heb")
            for Q in (16, 32, 64, 128, 256, 512):
                K_bc_n = compute_K_bc_M2_bounce_sphere(
                    SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                    n_quad=Q, K_max=K_max, weight_form="no_mu",
                )
                k_n = keff_from_K(
                    K_vol + K_bc_n, sig_t_arr, nu_sf_arr, sig_s_arr,
                )
                rel_inf = (k_n - K_INF) / K_INF * 100
                rel_h = (k_n - k_heb) / k_heb * 100
                print(f"  {Q:3d} | {k_n:.6f}   | {rel_inf:+.4f}%  | {rel_h:+.4f}%")
