"""Diagnostic: Sphere mode-0 P_esc/G_bc normalization.

From Issue #112 + reciprocity insight:

Sphere `G_bc^{(0)}(r_i) = 4 · P_esc^{(0)}(r_i)` numerically.
This is because the integrand ∫ sin θ exp(-τ) dθ is SHARED, but:
- G_bc uses prefactor 2
- P_esc uses prefactor 0.5
- Ratio = 4 = surface/volume ratio built into the primitives

In the rank-1 assembly:
    u[i] = Σ_t G_bc / R²   (R² = sphere surface divisor A_d = 4πR²/(4π))
    v[j] = r_j² · w_j · P_esc

    K_bc = u · v^T

So K_bc[i,j] = Σ_t G_bc(r_i) · r_j² w_j · P_esc(r_j) / R²
             = Σ_t · [4 P_esc(r_i)] · r_j² w_j · P_esc(r_j) / R²

Wait, but if G_bc = 4 P_esc for sphere, then the rank-1 formula can be rewritten
entirely in terms of P_esc.  Let me verify this claim by computing the rank-1
K_bc from a SYMMETRIC form and comparing to the production build.
"""
from __future__ import annotations
import sys

sys.path.insert(0, "/workspaces/ORPHEUS")
import numpy as np

from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D, SPHERE_1D,
    compute_G_bc, compute_G_bc_mode,
    compute_P_esc, compute_P_esc_mode,
    composite_gl_r,
    build_white_bc_correction,
)


def test_sphere_G_equals_4_P():
    """Sphere mode-0: G_bc(r) = 4 · P_esc(r) pointwise."""
    for R in [0.5, 1.0, 2.0, 5.0, 10.0]:
        radii = np.array([R])
        sig_t = np.array([1.0])
        r_nodes, _, _ = composite_gl_r(radii, 2, 5, dps=25)
        G = compute_G_bc(SPHERE_1D, r_nodes, radii, sig_t,
                         n_surf_quad=32, dps=25)
        P = compute_P_esc(SPHERE_1D, r_nodes, radii, sig_t,
                          n_angular=32, dps=25)
        ratio = G / P
        assert np.allclose(ratio, 4.0, rtol=1e-10), (
            f"R={R}: G/P ratio = {ratio} (expected constant 4.0)"
        )
        print(f"Sphere R={R}: G_bc/P_esc = 4.0  CONFIRMED")


def test_sphere_Gn_equals_4_Pn():
    """Sphere mode-n: G_bc^(n)(r) = 4 · P_esc^(n)(r) pointwise for all n."""
    for R in [1.0, 5.0]:
        radii = np.array([R])
        sig_t = np.array([1.0])
        r_nodes, _, _ = composite_gl_r(radii, 2, 5, dps=25)
        for n in range(1, 5):
            G = compute_G_bc_mode(SPHERE_1D, r_nodes, radii, sig_t, n,
                                  n_surf_quad=32, dps=25)
            P = compute_P_esc_mode(SPHERE_1D, r_nodes, radii, sig_t, n,
                                   n_angular=32, dps=25)
            # Ratio tricky because of sign changes
            good = np.abs(G - 4.0 * P) < 1e-8 * np.abs(P).max()
            all_good = np.all(good)
            print(f"Sphere R={R} n={n}: G_bc^(n) = 4·P_esc^(n)  "
                  f"{'CONFIRMED' if all_good else 'MISMATCH'}  "
                  f"max|G - 4P|/max|P| = {np.abs(G - 4*P).max() / np.abs(P).max():.2e}")


def test_cylinder_Gn_vs_Pn():
    """Cylinder: G_bc and P_esc use DIFFERENT integrands (surface-centered Ki_1/d vs
    observer-centered Ki_2).  So they should NOT be related by a simple constant.

    But if G and P were both observer-centered with Ki_2 and the 2D μ_s weighting,
    would they be related by reciprocity?

    Test numerically.
    """
    for R in [1.0, 5.0]:
        radii = np.array([R])
        sig_t = np.array([1.0])
        r_nodes, _, _ = composite_gl_r(radii, 2, 5, dps=25)
        G = compute_G_bc(CYLINDER_1D, r_nodes, radii, sig_t,
                         n_surf_quad=32, dps=25)
        P = compute_P_esc(CYLINDER_1D, r_nodes, radii, sig_t,
                          n_angular=32, dps=25)
        ratio = G / P
        print(f"\nCylinder R={R}: G_bc/P_esc ratio range: "
              f"[{ratio.min():.4f}, {ratio.max():.4f}]")
        for i in [0, len(r_nodes) // 2, -1]:
            print(f"  r={r_nodes[i]:.4f}  G={G[i]:.4e}  P={P[i]:.4e}  G/P={G[i]/P[i]:.4f}")


if __name__ == "__main__":
    test_sphere_G_equals_4_P()
    print()
    test_sphere_Gn_equals_4_Pn()
    print()
    test_cylinder_Gn_vs_Pn()
