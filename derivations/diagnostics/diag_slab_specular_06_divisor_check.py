"""Diagnostic: per-face slab specular with CORRECTED divisor (per-face area = 1, not 2).

Created by numerics-investigator on 2026-04-27.

The legacy slab Mark uses divisor = 2 (= total surface area for 2 unit-area
faces). When I build per-face K_bc by stacking outer & inner blocks, each
face has area 1 individually, so divisor should be 1.

Test rank-1 first: with R = block_diag(I, I) = I, divisor = 1, do we get
back to Mark legacy?

Then test rank-N convergence with corrected divisor.
"""
from __future__ import annotations

import numpy as np

from orpheus.derivations._xs_library import get_xs
from orpheus.derivations.peierls_geometry import (
    SLAB_POLAR_1D,
    _build_full_K_per_group,
    _shifted_legendre_monomial_coefs,
    _slab_tau_to_inner_face,
    _slab_tau_to_outer_face,
    build_volume_kernel,
    composite_gl_r,
    reflection_marshak,
    reflection_specular,
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


def _slab_per_face_PG(r_nodes, radii, sig_t_g, N, basis="no_mu"):
    N_r = len(r_nodes)
    P_o = np.zeros((N, N_r))
    P_i = np.zeros((N, N_r))
    G_o = np.zeros((N_r, N))
    G_i = np.zeros((N_r, N))
    P_offset = 3 if basis == "mu_weighted" else 2
    for i in range(N_r):
        x_i = float(r_nodes[i])
        tau_o = _slab_tau_to_outer_face(x_i, radii, sig_t_g)
        tau_n = _slab_tau_to_inner_face(x_i, radii, sig_t_g)
        for n in range(N):
            coefs = _shifted_legendre_monomial_coefs(n)
            Po = Pn = Go = Gn = 0.0
            for k, c in enumerate(coefs):
                if c == 0.0:
                    continue
                Po += 0.5 * c * _E_n(k + P_offset, tau_o)
                Pn += 0.5 * c * _E_n(k + P_offset, tau_n)
                Go += 2.0 * c * _E_n(k + 2, tau_o)
                Gn += 2.0 * c * _E_n(k + 2, tau_n)
            P_o[n, i] = Po
            P_i[n, i] = Pn
            G_o[i, n] = Go
            G_i[i, n] = Gn
    return P_o, P_i, G_o, G_i


def _build_K_with_per_face_R(N, R_2N_2N, divisor, basis="no_mu"):
    xs = get_xs("A", "1g")
    sig_t_g = xs["sig_t"]
    radii = np.array([L])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, n_panels_per_region=N_PANELS, p_order=P_ORDER, dps=DPS,
    )
    K_vol = build_volume_kernel(
        SLAB_POLAR_1D, r_nodes, panels, radii, sig_t_g,
        n_angular=N_ANGULAR, n_rho=N_RHO, dps=DPS,
    )
    rv = np.array([
        SLAB_POLAR_1D.radial_volume_weight(float(rj)) for rj in r_nodes
    ])
    sig_t_n = sig_t_g[0] * np.ones(len(r_nodes))

    P_o, P_i, G_o, G_i = _slab_per_face_PG(r_nodes, radii, sig_t_g, N, basis=basis)
    P_o_w = rv * r_wts * P_o
    P_i_w = rv * r_wts * P_i
    G_o_w = sig_t_n[:, None] * G_o / divisor
    G_i_w = sig_t_n[:, None] * G_i / divisor

    P_slab = np.vstack([P_o_w, P_i_w])
    G_slab = np.hstack([G_o_w, G_i_w])
    K_bc = G_slab @ R_2N_2N @ P_slab
    return K_vol + K_bc, r_nodes


def _solve_kdom(K, sig_t, sig_s_scalar, nuf_scalar):
    sigt = float(sig_t)
    A = sigt * np.eye(K.shape[0]) - K * sig_s_scalar
    B = K * nuf_scalar
    M = np.linalg.solve(A, B)
    eigs, vecs = np.linalg.eig(M)
    real_idx = np.argsort(eigs.real)[::-1]
    k_dom = float(eigs[real_idx[0]].real)
    v_dom = vecs[:, real_idx[0]].real
    if v_dom.sum() < 0:
        v_dom = -v_dom
    v_dom /= v_dom.mean()
    return k_dom, v_dom


def _block_diag(R_face):
    N = R_face.shape[0]
    R = np.zeros((2*N, 2*N))
    R[:N, :N] = R_face
    R[N:, N:] = R_face
    return R


def main():
    xs = get_xs("A", "1g")
    sigt = float(xs["sig_t"][0])
    sigs = float(xs["sig_s"][0, 0])
    nuf = float(xs["nu"][0] * xs["sig_f"][0])
    k_inf = nuf / (sigt - sigs)

    print(f"L={L}, k_inf={k_inf:.6f}")

    # Mark legacy reference
    radii = np.array([L])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, n_panels_per_region=N_PANELS, p_order=P_ORDER, dps=DPS,
    )
    K_mark = _build_full_K_per_group(
        SLAB_POLAR_1D, r_nodes, r_wts, panels, radii, np.array([sigt]),
        "white_rank1_mark",
        n_angular=N_ANGULAR, n_rho=N_RHO, n_surf_quad=N_SURF_QUAD,
        n_bc_modes=1, dps=DPS,
    )
    k_mark, v_mark = _solve_kdom(K_mark, sigt, sigs, nuf)
    print(f"\n--- Mark legacy ---")
    print(f"  k_eff = {k_mark:.10f}, rel = {(k_mark-k_inf)/k_inf*100:+.5f}%")

    # Per-face block-diag, vary divisor and N
    for divisor in (2.0, 1.0, 0.5):
        print(f"\n=== divisor = {divisor} ===")
        for basis in ("no_mu",):
            for N in (1, 2, 3, 4, 6, 8):
                R_face = reflection_specular(N)
                R = _block_diag(R_face)
                K, _ = _build_K_with_per_face_R(N, R, divisor, basis=basis)
                k, v = _solve_kdom(K, sigt, sigs, nuf)
                rel = (k - k_inf) / k_inf * 100
                print(f"  basis={basis:>9} N={N:>2} R=(1/2)M^-1 blkD : "
                      f"k={k:.8f}, rel={rel:+.5f}%, φ_max/min={v.max()/v.min():.4f}")
            for N in (1, 2, 3, 4, 6, 8):
                R_face = reflection_marshak(N)
                R = _block_diag(R_face)
                K, _ = _build_K_with_per_face_R(N, R, divisor, basis=basis)
                k, v = _solve_kdom(K, sigt, sigs, nuf)
                rel = (k - k_inf) / k_inf * 100
                print(f"  basis={basis:>9} N={N:>2} R=Marshak  blkD : "
                      f"k={k:.8f}, rel={rel:+.5f}%, φ_max/min={v.max()/v.min():.4f}")


if __name__ == "__main__":
    main()
