"""Round 2 follow-up — Audit the per-bounce integrand smoothness.

Suspect: F_out(r,µ) · G_in(r',µ) has 1/cos(ω) Jacobians that
diverge at µ = µ_min(r) (grazing), independently of multi-bounce
factor T(µ). Plot/print the integrand's behavior across µ ∈ (0, 1)
to find the singular structure.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    composite_gl_r,
)

from derivations.diagnostics.diag_phase5_native_c01_orpheus_form import (
    F_out_mu_sphere,
    G_in_mu_sphere,
)


def test_integrand_audit(capsys):
    """Plot the per-bounce integrand at several (i,j) pairs to spot the
    singular structure.
    """
    with capsys.disabled():
        R = 5.0
        sigma = 0.5
        radii = np.array([R])
        sig_t = np.array([sigma])
        # Two interior nodes and one near-surface node
        r_nodes = np.array([1.0, 2.5, 4.5])

        # Dense µ grid
        mu_grid = np.linspace(1e-4, 1 - 1e-6, 2000)

        F = F_out_mu_sphere(r_nodes, radii, sig_t, mu_grid)  # (Q, 3)
        G = G_in_mu_sphere(r_nodes, radii, sig_t, mu_grid)  # (3, Q)

        print(f"\n=== Integrand audit at R={R}, σ={sigma} ===")
        print(f"r_nodes = {r_nodes.tolist()}")
        print(f"µ_min for each r: {[float(np.sqrt(max(0, 1-(R/r)**2))) if r > 0 else 0 for r in r_nodes]}")

        for i in range(3):
            for j in range(3):
                # Integrand for K_bc^(0)[i,j] = 2 ∫ G[i] F[j] dµ (no T, no µ)
                integrand = G[i, :] * F[:, j]
                # Find max
                idx = np.argmax(np.abs(integrand))
                # Tail behavior
                near_zero = integrand[:5]
                near_one = integrand[-5:]
                print(f"\n  (i={i}, r_i={r_nodes[i]}, j={j}, r_j={r_nodes[j]})")
                print(f"  max|integrand| = {np.abs(integrand).max():.4e} at µ={mu_grid[idx]:.4f}")
                print(f"  integrand at µ→0: {[f'{x:.3e}' for x in near_zero]}")
                print(f"  integrand at µ→1: {[f'{x:.3e}' for x in near_one]}")
                # Look for the location where integrand peaks; if it's at
                # µ_min interface, the integrand has a sqrt singularity.
                # ∫ 1/√(µ-µ_min) dµ exists but is hard for plain GL.
        # Also: integrand at fine µ grid near µ→0 to spot 1/µ blow-up
        mu_fine = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 0.99])
        F_fine = F_out_mu_sphere(r_nodes, radii, sig_t, mu_fine)
        G_fine = G_in_mu_sphere(r_nodes, radii, sig_t, mu_fine)
        i, j = 1, 1
        print(f"\n  Per-µ tabulation, (i,j)=(1,1) (mid-radius):")
        print(f"  µ          | F[µ,j]      | G[i,µ]      | F·G")
        for q in range(len(mu_fine)):
            mu = float(mu_fine[q])
            f = float(F_fine[q, j])
            g = float(G_fine[i, q])
            print(f"  {mu:.4e}  | {f:.4e}  | {g:.4e}  | {f*g:.4e}")
