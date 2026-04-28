"""Diagnostic: cyl MB k_eff across τ_R regime.

Created by numerics-investigator on 2026-04-28.

Slab MB converges monotonically across τ_L ∈ [0.5, 10] (verified in
diag_phase4_08). Sphere MB overshoots at N ≥ 4 across thin regime.
Does cyl MB pathology persist across τ_R regime, or is it specific to
the thin τ_R = 2.5 case?
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    CurvilinearGeometry,
    _build_full_K_per_group,
    composite_gl_r,
    reflection_specular,
)

from derivations.diagnostics.diag_specular_mb_phase4_03_pathology_resolvent import (
    build_T_cyl,
)
from derivations.diagnostics.diag_specular_mb_phase4_06_keff_endtoend import (
    _build_cyl_PG,
    keff_from_K,
    SIG_S, NU_SIG_F, K_INF,
)

CYL_1D = CurvilinearGeometry(kind="cylinder-1d")


def _cyl_keff_sweep_N(R, sig_t, N_list, sig_s=None, nu_sf=None,
                       n_angular=12, dps=15):
    if sig_s is None:
        sig_s = SIG_S
    if nu_sf is None:
        nu_sf = NU_SIG_F
    radii = np.array([R])
    sig_t_g = np.array([sig_t])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, 1, 5, dps=dps, inner_radius=0.0,
    )
    sig_s_arr = np.full(len(r_nodes), sig_s)
    nu_sf_arr = np.full(len(r_nodes), nu_sf)
    sig_t_arr = np.full(len(r_nodes), sig_t)

    K_full_n1 = _build_full_K_per_group(
        CYL_1D, r_nodes, r_wts, panels, radii, sig_t_g,
        "specular",
        n_angular=n_angular, n_rho=n_angular, n_surf_quad=n_angular,
        n_bc_modes=1, dps=dps,
    )
    P1, G1, _ = _build_cyl_PG(R, sig_t_g, r_nodes, r_wts, 1,
                               n_angular=n_angular, n_surf_quad=n_angular,
                               dps=dps)
    R1 = reflection_specular(1)
    K_vol = K_full_n1 - G1 @ R1 @ P1

    out = []
    for N in N_list:
        P, G, _ = _build_cyl_PG(R, sig_t_g, r_nodes, r_wts, N,
                                 n_angular=n_angular, n_surf_quad=n_angular,
                                 dps=dps)
        R_op = reflection_specular(N)
        T = build_T_cyl(sig_t, R, N)
        ITR = np.eye(N) - T @ R_op
        K_mb = K_vol + G @ R_op @ np.linalg.solve(ITR, P)
        k = keff_from_K(K_mb, sig_t_arr, nu_sf_arr, sig_s_arr)
        k_inf_local = nu_sf / (sig_t - sig_s)
        out.append((N, k, (k - k_inf_local) / k_inf_local * 100))
    return out


def test_cyl_mb_regime_sweep(capsys):
    """Cyl MB regime sweep: confirm overshoot pathology at high N."""
    with capsys.disabled():
        N_list = (1, 2, 3, 4, 6, 8)
        c = SIG_S / 0.5
        f = NU_SIG_F / 0.5
        for sig_t, R in [(0.2, 5.0), (0.5, 5.0), (1.0, 5.0)]:
            sig_s = c * sig_t
            nu_sf = f * sig_t
            tau_R = sig_t * R
            print(f"\n  τ_R = {tau_R} (σ_t={sig_t}, R={R})")
            print(f"     N |  k_mb       |  rel(%)")
            results = _cyl_keff_sweep_N(R, sig_t, N_list, sig_s=sig_s,
                                          nu_sf=nu_sf)
            for N, k, rel in results:
                print(f"     {N:2d}|  {k:.6f}  |  {rel:+.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
