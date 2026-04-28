"""Round 3 — Phase 4 K_bc structure analysis.

Look at Phase 4 K_bc^specular_mb at varying rank N. Does the
diagonal grow with rank? Does the off-diagonal? This will tell us
what the "right" continuous-µ K_bc[i,j] should look like.
"""
from __future__ import annotations

import numpy as np

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _build_full_K_per_group,
    composite_gl_r,
)

from derivations.diagnostics.diag_phase5_round3_adaptive_quadrature import (
    K_INF, NU_SIG_F, R_THIN, SIG_S, SIG_T, keff_from_K,
)


def test_phase4_kbc_structure_vs_rank(capsys):
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 1, 3, dps=15, inner_radius=0.0,
        )
        N = len(r_nodes)
        sig_s_arr = np.full(N, SIG_S)
        nu_sf_arr = np.full(N, NU_SIG_F)
        sig_t_arr = np.full(N, SIG_T)

        K_vol = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )

        print(f"\n=== Phase 4 K_bc structure vs rank ===")
        print(f"r_nodes = {r_nodes.tolist()}")
        for n_bc in (1, 2, 3, 4, 5, 6):
            try:
                K_p4 = _build_full_K_per_group(
                    SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
                    "specular_multibounce",
                    n_angular=24, n_rho=24, n_surf_quad=24,
                    n_bc_modes=n_bc, dps=20,
                )
            except Exception as e:
                print(f"  rank-{n_bc}: FAIL — {e}")
                continue
            K_bc = K_p4 - K_vol
            k = keff_from_K(K_p4, sig_t_arr, nu_sf_arr, sig_s_arr)
            print(f"\n  rank-{n_bc}: k_eff = {k:.6f} ({(k-K_INF)/K_INF*100:+.4f}%)")
            print(f"  K_bc:")
            print(K_bc)
            print(f"  diag: {np.diag(K_bc)}")
            print(f"  row sums: {np.sum(K_bc, axis=1)}")
