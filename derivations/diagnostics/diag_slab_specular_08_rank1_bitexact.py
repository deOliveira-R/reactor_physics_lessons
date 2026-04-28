"""Diagnostic: at rank-1, slab specular block-diag with divisor=1 should
bit-exactly equal Mark legacy K_bc.

Created by numerics-investigator on 2026-04-27.

The fix: per-face slab specular uses divisor = 1 (single face area).
At rank-1 with R_face = [[1]] = (1/2) M^-1 at N=1, the block-diagonal
R is diag(1, 1) and the K_bc reconstruction is

    K_bc = (G_outer / 1) * 1 * (P_outer * weights) +
           (G_inner / 1) * 1 * (P_inner * weights)

Mark legacy K_bc:
    P_combined = ½ E_2_outer + ½ E_2_inner
    G_combined = 2 E_2_outer + 2 E_2_inner
    K_bc_mark = (G_combined / 2) ⊗ (P_combined * weights)
              = ((G_outer + G_inner) / 2) ⊗ ((P_outer + P_inner) * weights)

Expand: K_bc_mark = (1/2) [G_o · P_o + G_o · P_i + G_i · P_o + G_i · P_i]

K_bc_specular_blockdiag (divisor=1) = G_o · P_o + G_i · P_i

These are NOT equal:
   K_bc_mark - K_bc_specular = (1/2)(G_o·P_o + G_o·P_i + G_i·P_o + G_i·P_i)
                             - (G_o·P_o + G_i·P_i)
                             = -(1/2) G_o·P_o + (1/2) G_o·P_i
                              + (1/2) G_i·P_o - (1/2) G_i·P_i
                             = (1/2)(G_o - G_i)(P_i - P_o)

For homogeneous slab with x_i symmetric around L/2: G_o(x_i) = G_i(L-x_i),
so the difference (G_o - G_i)(P_i - P_o) is anti-symmetric around L/2,
contributing zero net to the dominant eigenvalue (?).

But the DIAGNOSTIC at -1.60% rank-1 with divisor=1 specular MATCHED Mark
legacy at -1.60%. Something is hiding.

Let me actually compute K_bc element-by-element for rank-1 to see what's
going on.
"""
from __future__ import annotations

import numpy as np

from orpheus.derivations._xs_library import get_xs
from orpheus.derivations.peierls_geometry import (
    SLAB_POLAR_1D,
    _build_full_K_per_group,
    _slab_E2,
    _slab_tau_to_inner_face,
    _slab_tau_to_outer_face,
    build_volume_kernel,
    composite_gl_r,
    compute_G_bc,
    compute_P_esc,
)


L = 5.0
P_ORDER = 3
N_PANELS = 1
N_ANGULAR = 8
N_RHO = 8
N_SURF_QUAD = 8
DPS = 15


def _E_n(n, tau):
    import mpmath as mp
    if tau == 0.0:
        if n == 1:
            return float("inf")
        return 1.0 / (n - 1)
    return float(mp.expint(n, tau))


def main():
    xs = get_xs("A", "1g")
    sigt = float(xs["sig_t"][0])
    radii = np.array([L])
    sig_t_g = xs["sig_t"]
    r_nodes, r_wts, panels = composite_gl_r(
        radii, n_panels_per_region=N_PANELS, p_order=P_ORDER, dps=DPS,
    )
    print(f"r_nodes = {r_nodes}")
    print(f"r_wts   = {r_wts}")

    # Per-face mode-0 primitives (no_mu basis, E_2):
    P_o = np.array([0.5 * _E_n(2, _slab_tau_to_outer_face(float(x), radii, sig_t_g))
                    for x in r_nodes])
    P_i = np.array([0.5 * _E_n(2, _slab_tau_to_inner_face(float(x), radii, sig_t_g))
                    for x in r_nodes])
    G_o = np.array([2.0 * _E_n(2, _slab_tau_to_outer_face(float(x), radii, sig_t_g))
                    for x in r_nodes])
    G_i = np.array([2.0 * _E_n(2, _slab_tau_to_inner_face(float(x), radii, sig_t_g))
                    for x in r_nodes])
    print(f"\nP_outer = {P_o}, P_inner = {P_i}")
    print(f"G_outer = {G_o}, G_inner = {G_i}")

    rv = np.array([SLAB_POLAR_1D.radial_volume_weight(float(x)) for x in r_nodes])

    # ----- Variant A: legacy Mark via combined P+G with divisor=2 -----
    P_comb_w = rv * r_wts * (P_o + P_i)
    G_comb_div2 = sigt * (G_o + G_i) / 2.0
    K_bc_mark_legacy = np.outer(G_comb_div2, P_comb_w)
    print(f"\nK_bc_mark_legacy =\n{K_bc_mark_legacy}")

    # Reference via API:
    K_mark_full = _build_full_K_per_group(
        SLAB_POLAR_1D, r_nodes, r_wts, panels, radii, sig_t_g,
        "white_rank1_mark",
        n_angular=N_ANGULAR, n_rho=N_RHO, n_surf_quad=N_SURF_QUAD,
        n_bc_modes=1, dps=DPS,
    )
    K_vol = build_volume_kernel(
        SLAB_POLAR_1D, r_nodes, panels, radii, sig_t_g,
        n_angular=N_ANGULAR, n_rho=N_RHO, dps=DPS,
    )
    K_bc_mark_api = K_mark_full - K_vol
    print(f"\nK_bc_mark via API =\n{K_bc_mark_api}")
    print(f"max |my - api| = {np.max(np.abs(K_bc_mark_legacy - K_bc_mark_api)):.6e}")

    # ----- Variant B: per-face block-diag, R_face=[[1]], divisor=1 -----
    P_o_w = rv * r_wts * P_o
    P_i_w = rv * r_wts * P_i
    G_o_div1 = sigt * G_o / 1.0
    G_i_div1 = sigt * G_i / 1.0
    K_bc_pf_div1 = np.outer(G_o_div1, P_o_w) + np.outer(G_i_div1, P_i_w)
    print(f"\nK_bc_per_face (divisor=1, R=I, blockdiag) =\n{K_bc_pf_div1}")
    print(f"max |Kpf_div1 - Kmark| = {np.max(np.abs(K_bc_pf_div1 - K_bc_mark_legacy)):.6e}")

    # The difference should be:
    diff = K_bc_pf_div1 - K_bc_mark_legacy
    print(f"\ndiff =\n{diff}")

    # ----- Variant C: per-face block-diag, R_face=[[1]], divisor=2 (incorrect) -----
    G_o_div2 = sigt * G_o / 2.0
    G_i_div2 = sigt * G_i / 2.0
    K_bc_pf_div2 = np.outer(G_o_div2, P_o_w) + np.outer(G_i_div2, P_i_w)
    print(f"\nK_bc_per_face (divisor=2, R=I, blockdiag) =\n{K_bc_pf_div2}")

    # Predicted exact relation: Mark = pf_div2 + (1/2)(G_o P_i + G_i P_o)
    cross = 0.5 * (np.outer(G_o, P_i_w * sigt) + np.outer(G_i, P_o_w * sigt))
    print(f"\ncross (1/2)(G_o ⊗ P_i_w + G_i ⊗ P_o_w) * sigt =\n{cross}")
    pred = K_bc_pf_div2 + cross
    print(f"\npred = K_bc_pf_div2 + cross =\n{pred}")
    print(f"max |pred - Kmark| = {np.max(np.abs(pred - K_bc_mark_legacy)):.6e}")

    # Power iteration eigenvalue check
    sigs = float(xs["sig_s"][0, 0])
    nuf = float(xs["nu"][0] * xs["sig_f"][0])
    k_inf = nuf / (sigt - sigs)

    def kdom(K):
        A = sigt * np.eye(K.shape[0]) - K * sigs
        B = K * nuf
        M = np.linalg.solve(A, B)
        eigs = np.linalg.eigvals(M)
        return float(max(eigs.real))

    print(f"\n--- Eigenvalues ---")
    print(f"k_inf = {k_inf}")
    print(f"K_vol+K_mark_legacy: {kdom(K_vol + K_bc_mark_legacy):.10f}")
    print(f"K_vol+K_pf_div1:     {kdom(K_vol + K_bc_pf_div1):.10f}")
    print(f"K_vol+K_pf_div2:     {kdom(K_vol + K_bc_pf_div2):.10f}")


if __name__ == "__main__":
    main()
