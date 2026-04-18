"""Direct numerical check: is there a missing π factor in the rank-1 code?

Set up a simple fixed-source problem:
   source q = 1 uniform in a sphere of radius R with Σ_t = 1, Σ_s = 0 (pure absorber), white BC.
The Peierls equation gives φ_bc from the boundary contribution.

We compute φ_bc two ways:
1. Via the code's K_bc @ q
2. Via the manual derivation: φ_bc(r_i) = π · m^-_0 · g_0(r_i),
   m^-_0 = (1/A) ∫_V q P_esc^(0) dV

If they agree → my derivation's π is WRONG.
If they differ by π → code is missing π.

Since rank-1 is regression-tested, I expect my derivation to be off by π.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/workspaces/ORPHEUS")
import numpy as np

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, composite_gl_r,
    compute_G_bc, compute_P_esc,
    build_white_bc_correction,
)


def main():
    R = 2.0
    sig_t = np.array([1.0])
    radii = np.array([R])
    r_nodes, r_wts, _ = composite_gl_r(radii, 2, 5, dps=25)

    g_0 = compute_G_bc(SPHERE_1D, r_nodes, radii, sig_t, n_surf_quad=64, dps=25)
    P_0 = compute_P_esc(SPHERE_1D, r_nodes, radii, sig_t, n_angular=64, dps=25)

    K_bc = build_white_bc_correction(
        SPHERE_1D, r_nodes, r_wts, radii, sig_t,
        n_angular=64, n_surf_quad=64, dps=25,
    )

    # Uniform q = 1.
    q = np.ones_like(r_nodes)

    # Code computes K_bc · q = Σ_t · φ_bc_code (in the Peierls LHS-form).
    # Wait — actually K is the full kernel; but rank-1 K_bc specifically
    # represents the contribution from the boundary reflections.
    # K · q gives Σ_t(r_i) · φ(r_i) for the Peierls form.
    Kq_code = K_bc @ q

    # So φ_bc_code(r_i) = (K_bc q)_i / Σ_t(r_i).
    sig_t_n = np.full_like(r_nodes, sig_t[0])
    phi_bc_code = Kq_code / sig_t_n

    # Alternative: compute φ_bc from my derivation.
    # m^-_0 = ⟨m^+_0⟩ = (1/A) ∫_V q P_esc dV = (4π / A) · ∫_0^R q(r') P_esc(r') r'² dr'
    # For sphere, A = 4πR², so m^-_0 = (1/R²) · ∫_0^R q P_esc r'² dr'
    A = 4.0 * np.pi * R * R
    Vq_P = 4.0 * np.pi * np.sum(q * P_0 * r_nodes ** 2 * r_wts)
    m_minus_0 = Vq_P / A
    print(f"⟨m^+_0⟩ computed = {m_minus_0:.6e}")

    # My derivation: φ_bc(r_i) = π · m^-_0 · g_0(r_i)
    phi_bc_mine = np.pi * m_minus_0 * g_0
    print(f"\nr         φ_bc_code     φ_bc_mine     ratio  (mine/code)")
    for i in range(len(r_nodes)):
        print(f"{r_nodes[i]:.4f}  {phi_bc_code[i]:.4e}  {phi_bc_mine[i]:.4e}  "
              f"{phi_bc_mine[i]/phi_bc_code[i]:.4f}")


if __name__ == "__main__":
    main()
