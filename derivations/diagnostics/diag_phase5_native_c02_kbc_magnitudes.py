"""Phase 5 Front C — magnitudes of K_bc forms compared at thin sphere.

Probe the absolute scale of:
- K_bc from the existing Sanchez Eq. (A6) reference
- K_bc from white_hebert (rank-1 Hebert, working production)
- K_bc from specular_multibounce N=1 (matches white_hebert at rank-1)
- K_vol (volume kernel only)

This pins the Sanchez↔ORPHEUS conversion factor by looking at off-
diagonal entries (where the Sanchez singularity is mild) rather than
the diagonal (where 1/µ² dominates).
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _build_full_K_per_group,
    composite_gl_r,
    compute_K_bc_specular_continuous_mu_sphere,
)


SIG_T = 0.5
SIG_S = 0.38
NU_SIG_F = 0.025
R_THIN = 5.0


def test_kbc_magnitudes_thin_sphere(capsys):
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=0.0,
        )

        # K_vol only
        K_vac = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=15,
        )
        # K_full white_hebert
        K_heb = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "white_hebert",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=15,
        )
        # K_bc^Hebert = K_heb - K_vac
        K_bc_heb = K_heb - K_vac
        # K_bc^Sanchez at Q=64 (more representative)
        K_bc_san = compute_K_bc_specular_continuous_mu_sphere(
            r_nodes, radii, sig_t_g, n_quad=64,
        )
        # K_bc^Sanchez at Q=128
        K_bc_san_128 = compute_K_bc_specular_continuous_mu_sphere(
            r_nodes, radii, sig_t_g, n_quad=128,
        )

        print(f"\n=== Magnitudes at R={R_THIN}, σ_t={SIG_T} ===")
        print(f"r_nodes: {r_nodes}")
        print(f"\nK_vol max abs:        {np.max(np.abs(K_vac)):.6e}")
        print(f"K_bc^Hebert max abs:    {np.max(np.abs(K_bc_heb)):.6e}")
        print(f"K_bc^Sanchez Q=64 max:  {np.max(np.abs(K_bc_san)):.6e}")
        print(f"K_bc^Sanchez Q=128 max: {np.max(np.abs(K_bc_san_128)):.6e}")

        # Print K_bc^Hebert as a reference
        print(f"\nK_bc^Hebert (rank-1, ORPHEUS-native, working):")
        with np.printoptions(precision=4, suppress=False):
            print(K_bc_heb)
        print(f"\nK_bc^Sanchez (Q=64):")
        with np.printoptions(precision=4, suppress=False):
            print(K_bc_san)

        # Off-diagonal ratios (skip diagonal — Sanchez singular)
        print(f"\n=== Off-diagonal K_bc^Sanchez / K_bc^Hebert ratios ===")
        n = len(r_nodes)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # skip singular diag
                if abs(K_bc_heb[i, j]) > 1e-10:
                    r = K_bc_san[i, j] / K_bc_heb[i, j]
                    print(f"  i={i} j={j} r_i={r_nodes[i]:.3f} "
                          f"r_j={r_nodes[j]:.3f}  Hebert={K_bc_heb[i,j]:.4e}  "
                          f"Sanchez={K_bc_san[i,j]:.4e}  ratio={r:.4f}")
