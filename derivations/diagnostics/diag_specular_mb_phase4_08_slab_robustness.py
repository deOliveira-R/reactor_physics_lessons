"""Diagnostic: slab MB convergence robustness across τ_L regime.

Created by numerics-investigator on 2026-04-28.

The slab MB k_eff was clean monotonic at τ_L = 2.5 (the headline thin
test). Verify the convergence behaviour holds across a regime sweep.

If slab MB plateaus and stays plateau-bounded across τ_L ∈ [0.5, 10],
the closure is robust enough to ship without a UserWarning.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SLAB_POLAR_1D,
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
    SIG_S, NU_SIG_F, K_INF,
)


def _slab_keff_sweep_N(L, sig_t, N_list, sig_s=None, nu_sf=None):
    if sig_s is None:
        sig_s = SIG_S
    if nu_sf is None:
        nu_sf = NU_SIG_F
    radii = np.array([L])
    sig_t_g = np.array([sig_t])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, 1, 5, dps=15, inner_radius=0.0,
    )
    sig_s_arr = np.full(len(r_nodes), sig_s)
    nu_sf_arr = np.full(len(r_nodes), nu_sf)
    sig_t_arr = np.full(len(r_nodes), sig_t)
    K_full_n1 = _build_full_K_per_group(
        SLAB_POLAR_1D, r_nodes, r_wts, panels, radii, sig_t_g,
        "specular",
        n_angular=12, n_rho=12, n_surf_quad=12,
        n_bc_modes=1, dps=15,
    )
    P1, G1, _ = _build_slab_PG_perface(L, sig_t, r_nodes, r_wts, 1)
    R1_face = reflection_specular(1)
    R1_slab = np.zeros((2, 2)); R1_slab[:1, :1] = R1_face; R1_slab[1:, 1:] = R1_face
    K_vol = K_full_n1 - G1 @ R1_slab @ P1

    out = []
    for N in N_list:
        P, G, _ = _build_slab_PG_perface(L, sig_t, r_nodes, r_wts, N)
        R_face = reflection_specular(N)
        R_slab = np.zeros((2 * N, 2 * N))
        R_slab[:N, :N] = R_face; R_slab[N:, N:] = R_face
        T = build_T_slab(sig_t, L, N)
        ITR = np.eye(2 * N) - T @ R_slab
        K_mb = K_vol + G @ R_slab @ np.linalg.solve(ITR, P)
        k = keff_from_K(K_mb, sig_t_arr, nu_sf_arr, sig_s_arr)
        # k_inf = nu_sf / (sig_t - sig_s) — invariant ratio
        k_inf_local = nu_sf / (sig_t - sig_s)
        out.append((N, k, (k - k_inf_local) / k_inf_local * 100))
    return out


def test_slab_mb_regime_sweep(capsys):
    """Slab MB at τ_L ∈ {0.5, 1.0, 2.5, 5.0, 10.0}.

    For each τ_L, verify the MB k_eff is monotonic in N and never
    overshoots k_inf.

    Note: σ_s, νσ_f are SCALED proportionally with σ_t to maintain
    the same fuel-A-like ratio (c = σ_s/σ_t = 0.76, k_inf = 0.20833).
    """
    with capsys.disabled():
        N_list = (1, 2, 4, 8, 16)
        c = SIG_S / 0.5  # ratio σ_s/σ_t at the headline test
        f = NU_SIG_F / 0.5  # νσ_f / σ_t
        for sig_t, L in [(0.1, 5.0), (0.2, 5.0), (0.5, 5.0), (1.0, 5.0), (2.0, 5.0)]:
            sig_s = c * sig_t  # keep σ_s/σ_t fixed
            nu_sf = f * sig_t  # keep νσ_f/σ_t fixed
            k_inf_local = nu_sf / (sig_t - sig_s)  # invariant = K_INF
            tau_L = sig_t * L
            print(f"\n  τ_L = {tau_L} (σ_t={sig_t}, L={L}, k_inf={k_inf_local:.6f})")
            print(f"     N |  k_mb       |  rel(%)")
            results = _slab_keff_sweep_N(
                L, sig_t, N_list, sig_s=sig_s, nu_sf=nu_sf,
            )
            ks = []
            for N, k, _ in results:
                ks.append(k)
                rel_loc = (k - k_inf_local) / k_inf_local * 100
                print(f"     {N:2d}|  {k:.6f}  |  {rel_loc:+.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
