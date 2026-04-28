"""Diagnostic: compare bare-specular vs specular_mb eigvec at high N.

Created by numerics-investigator on 2026-04-27.

Bare specular at N=8 thin: -3.22% (under k_inf)
specular_mb at N=8 thin: +5.62% (over k_inf)

What's the eigvec shape difference? Where is K_bc^mb adding too much
flux? Compare K_bc^bare vs K_bc^mb element-wise to identify over-amplified
matrix entries.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _build_full_K_per_group,
    build_volume_kernel,
    composite_gl_r,
)


def _solve(K, sigt, sigs, nuf, return_vec=False):
    N = K.shape[0]
    A = sigt * np.eye(N) - sigs * K
    B = nuf * K
    M = np.linalg.solve(A, B)
    eigvals, eigvecs = np.linalg.eig(M)
    real_mask = np.abs(eigvals.imag) < 1e-10
    real_eigs = eigvals[real_mask].real
    real_vecs = eigvecs[:, real_mask].real
    idx = np.argmax(real_eigs)
    if return_vec:
        v = real_vecs[:, idx]
        sign = np.sign(v[np.argmax(np.abs(v))])
        v = v * sign
        v = v / np.max(np.abs(v))
        return float(real_eigs[idx]), v
    return float(real_eigs[idx])


def test_compare_bare_mb_at_N8(capsys):
    with capsys.disabled():
        R = 5.0
        sigt = 0.5
        sigs = 0.38
        nuf = 0.025
        k_inf = nuf / (sigt - sigs)
        radii = np.array([R])
        sig_t_g = np.array([sigt])
        r_nodes, r_wts, panels = composite_gl_r(radii, 2, 4, dps=20)
        K_vol = build_volume_kernel(
            SPHERE_1D, r_nodes, panels, radii, sig_t_g,
            n_angular=24, n_rho=24, dps=20,
        )
        N_r = len(r_nodes)
        N_modes = 8
        print(f"\n=== N_modes={N_modes} thin sphere comparison ===")
        print(f"  r_nodes = {r_nodes}")
        print(f"  k_inf = {k_inf:.6f}")

        K_bare = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels,
            radii, sig_t_g, "specular",
            n_angular=24, n_rho=24, n_surf_quad=24, dps=20, n_bc_modes=N_modes,
        )
        K_mb = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels,
            radii, sig_t_g, "specular_multibounce",
            n_angular=24, n_rho=24, n_surf_quad=24, dps=20, n_bc_modes=N_modes,
        )

        K_bc_bare = K_bare - K_vol
        K_bc_mb = K_mb - K_vol

        k_bare, v_bare = _solve(K_bare, sigt, sigs, nuf, return_vec=True)
        k_mb, v_mb = _solve(K_mb, sigt, sigs, nuf, return_vec=True)

        print(f"\n  Bare specular: k={k_bare:.6f} ({(k_bare-k_inf)/k_inf*100:+.3f}%)")
        print(f"  Bare eigvec:   {v_bare}")
        print(f"\n  Multi-bounce:  k={k_mb:.6f}  ({(k_mb-k_inf)/k_inf*100:+.3f}%)")
        print(f"  MB eigvec:     {v_mb}")

        # K_bc^mb / K_bc^bare element-wise (where bare nonzero)
        ratio = np.where(np.abs(K_bc_bare) > 1e-12, K_bc_mb / K_bc_bare,
                         np.nan)
        print(f"\n  K_bc^mb / K_bc^bare (element-wise) min/max/mean:")
        print(f"    min  = {np.nanmin(ratio):.4f}")
        print(f"    max  = {np.nanmax(ratio):.4f}")
        print(f"    mean = {np.nanmean(ratio):.4f}")
        print(f"  Per-row mean ratio (boundary contribution multiplier):")
        for i in range(N_r):
            print(f"    r_i={r_nodes[i]:.3f}: row mean ratio = "
                  f"{np.nanmean(ratio[i, :]):.4f}, "
                  f"K_bc_mb sum = {K_bc_mb[i,:].sum():.4e}, "
                  f"K_bc_bare sum = {K_bc_bare[i,:].sum():.4e}")

        # Apply uniform q to both; compare resulting flux
        q_uni = np.ones(N_r)
        phi_bare = K_bare @ q_uni
        phi_mb = K_mb @ q_uni
        print(f"\n  K · q_uniform (per cell):")
        print(f"  r_i  | bare φ  | MB φ    | MB/bare ratio")
        for i in range(N_r):
            print(f"  {r_nodes[i]:.3f} | {phi_bare[i]:.4f} | {phi_mb[i]:.4f} | "
                  f"{phi_mb[i]/phi_bare[i]:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
