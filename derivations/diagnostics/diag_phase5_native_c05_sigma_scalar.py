"""Phase 5 Front C — test scalar conversion K_orpheus = σ · K_sanchez.

The Sanchez 1986 form uses optical units (ρ = σr). The Peierls
integral in optical units is

    Σ·φ(ρ) = ∫ g(ρ' → ρ) q(ρ') dρ' / σ

After the σ Jacobian for dρ = σ dr, ORPHEUS K_ij = σ · g_h.

Test: at thin sphere (R=5, σ=0.5), do various ratios K_orpheus / K_sanchez
collapse to a constant when we exclude the diagonal Sanchez singularity?

If yes → simple scalar conversion → Phase 5 production wiring is trivial.
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
R_THIN = 5.0


def test_sanchez_kernel_intrinsic_scaling(capsys):
    """Compute K_sanchez at multiple Q. The OFF-DIAGONAL entries should
    converge as Q grows (the diagonal has the 1/µ² singularity but
    off-diagonal has only mild integrable behavior near µ_0).
    """
    with capsys.disabled():
        r_nodes = np.array([1.0, 2.0, 3.0, 4.0])  # interior nodes only
        radii = np.array([R_THIN])
        sig_t = np.array([SIG_T])

        print(f"\n=== Sanchez kernel Q-convergence on interior grid ===")
        print(f"r_nodes: {r_nodes}")
        print(f"\n  Q   | K[1,2]      | K[1,3]      | K[2,1]      | K[3,0]")
        for Q in (32, 64, 128, 256, 512):
            K = compute_K_bc_specular_continuous_mu_sphere(
                r_nodes, radii, sig_t, n_quad=Q,
            )
            print(
                f"  {Q:3d} | "
                f"{K[1,2]:.6e} | {K[1,3]:.6e} | "
                f"{K[2,1]:.6e} | {K[3,0]:.6e}"
            )
        print(f"\nDiagonal (singular):")
        for Q in (32, 64, 128, 256, 512):
            K = compute_K_bc_specular_continuous_mu_sphere(
                r_nodes, radii, sig_t, n_quad=Q,
            )
            print(f"  Q={Q}: K[1,1]={K[1,1]:.4e}, K[3,3]={K[3,3]:.4e}")


def test_compare_sanchez_to_phase4_low_N(capsys):
    """Compare K_sanchez (off-diagonal entries) to K_bc^Phase4 at N=1, 2, 3
    to discover the scalar conversion empirically.
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=0.0,
        )
        # K_vac
        K_vac = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=15,
        )
        # K_full Phase 4 multibounce N=1, 2, 3
        K_phase4 = {}
        for N in (1, 2, 3):
            K_full = _build_full_K_per_group(
                SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
                "specular_multibounce",
                n_angular=24, n_rho=24, n_surf_quad=24,
                n_bc_modes=N, dps=15,
            )
            K_phase4[N] = K_full - K_vac
        # K_sanchez at Q=128
        K_san = compute_K_bc_specular_continuous_mu_sphere(
            r_nodes, radii, sig_t_g, n_quad=128,
        )

        print(f"\n=== Phase 4 (N=1,2,3) vs Sanchez ratios ===")
        print(f"r_nodes: {r_nodes}")
        n = len(r_nodes)
        # Pick representative pairs
        pairs = [(1, 5), (2, 6), (3, 4), (5, 6), (4, 5)]
        print(f"\n  i j |   r_i,r_j   | K_p4(N=1)   | K_p4(N=2)   | "
              f"K_p4(N=3)   | K_san       | K_san/K_p4(1)")
        for i, j in pairs:
            r_p4_1 = K_phase4[1][i, j]
            r_p4_2 = K_phase4[2][i, j]
            r_p4_3 = K_phase4[3][i, j]
            r_san = K_san[i, j]
            ratio = r_san / r_p4_1 if abs(r_p4_1) > 1e-12 else float('nan')
            print(
                f"  {i} {j} | "
                f"{r_nodes[i]:.3f},{r_nodes[j]:.3f} | "
                f"{r_p4_1:.4e} | {r_p4_2:.4e} | "
                f"{r_p4_3:.4e} | {r_san:.4e} | {ratio:.4f}"
            )

        # Test scaled Sanchez (K_san * σ ?)
        print(f"\n  Hypothesized scaling: σ = {SIG_T}")
        print(f"  i j | K_p4(N=3)        | σ·K_san         | ratio")
        for i, j in pairs:
            r_p4_3 = K_phase4[3][i, j]
            r_scaled = SIG_T * K_san[i, j]
            ratio = r_scaled / r_p4_3 if abs(r_p4_3) > 1e-12 else float('nan')
            print(
                f"  {i} {j} | {r_p4_3:.6e} | {r_scaled:.6e} | {ratio:.4f}"
            )

        # Test the rv·r_wts·sig_t/divisor scaling
        rv = np.array([
            SPHERE_1D.radial_volume_weight(float(rj)) for rj in r_nodes
        ])
        sig_t_n = np.array([
            sig_t_g[SPHERE_1D.which_annulus(float(r_nodes[i]), radii)]
            for i in range(len(r_nodes))
        ])
        divisor = SPHERE_1D.rank1_surface_divisor(R_THIN)
        K_san_scaled = (
            (sig_t_n / divisor)[:, None]
            * (rv * r_wts)[None, :]
            * K_san
        )
        print(f"\n  Trying: K_san_scaled = (σ_t/divisor) · K_san · (rv·r_wts)")
        print(f"  i j | K_p4(N=3)        | scaled_K_san     | ratio")
        for i, j in pairs:
            r_p4_3 = K_phase4[3][i, j]
            r_sc = K_san_scaled[i, j]
            ratio = r_sc / r_p4_3 if abs(r_p4_3) > 1e-12 else float('nan')
            print(
                f"  {i} {j} | {r_p4_3:.6e} | {r_sc:.6e} | {ratio:.4f}"
            )

        # Hypothesis 3: K_orpheus[i,j] = α · K_san[i,j] / (r_j²) — i.e.,
        # the 4π r² Jacobian of the 3D radial-volume element.
        print(f"\n  Trying: K_orpheus[i,j] = α · K_san[i,j] / r_j²")
        ratios = []
        for i, j in pairs:
            r_p4_3 = K_phase4[3][i, j]
            r_san = K_san[i, j]
            r_j_val = r_nodes[j]
            if abs(r_p4_3) > 1e-12:
                a = r_p4_3 * r_j_val * r_j_val / r_san
                ratios.append(a)
                print(
                    f"  {i} {j} | r_j={r_j_val:.3f}  α = {a:.6f}"
                )
        if ratios:
            print(f"  α median = {np.median(ratios):.6f}, std/med = "
                  f"{np.std(ratios)/np.median(ratios):.4f}")

        # Hypothesis 4: K_orpheus[i,j] = α(r_i, r_j) · K_san[i,j]
        # with α(r_i, r_j) = w_i · w_j (separable)
        print(f"\n  Trying: K_orpheus[i,j] / K_san[i,j] decomposition test")
        for i_ref in (3, 5):
            print(f"  Holding j={i_ref} fixed, vary i:")
            for i in range(len(r_nodes)):
                if abs(K_san[i, i_ref]) > 1e-12 and abs(K_phase4[3][i, i_ref]) > 1e-12 and i != i_ref:
                    r = K_phase4[3][i, i_ref] / K_san[i, i_ref]
                    print(f"    i={i} r_i={r_nodes[i]:.3f} ratio={r:.4f}")
