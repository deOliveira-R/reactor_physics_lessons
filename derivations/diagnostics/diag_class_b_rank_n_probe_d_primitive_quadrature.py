"""Probe D: rank-N PRIMITIVE convergence under quadrature refinement.

Created by numerics-investigator on 2026-04-24.

Per probe-cascade SKILL §closed-form-detection: if a finite-N GL integral
plateaus above machine precision under refinement, that is the
fingerprint of a closed-form integral being silently approximated by
quadrature (Issue #131 anti-pattern).

Test: at sphere 1G/2R-Z (radii=[0.5, 1.0], σ_t=[1.0, 2.0]), evaluate
`compute_P_esc_mode(n=1)` and `compute_G_bc_mode(n=1)` per radial node
under increasing quadrature.

If the values converge cleanly under refinement → quadrature is fine, bug
is elsewhere (closure assembly, reflection operator, K_vol×K_bc coupling).

If the values plateau or oscillate → primitive itself has bad numerics.
"""

from __future__ import annotations

import numpy as np

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    composite_gl_r,
    compute_P_esc_mode,
    compute_G_bc_mode,
    compute_P_esc,
    compute_G_bc,
)


def _radial_nodes(radii, n_panels_per_region=2, p_order=3, dps=15):
    r_nodes, r_wts, _ = composite_gl_r(
        radii, n_panels_per_region, p_order, dps=dps, inner_radius=0.0,
    )
    return r_nodes, r_wts


def main():
    print("=" * 76)
    print("Probe D: rank-N primitive convergence under quadrature refinement")
    print("         sphere 2R, radii=[0.5,1.0], σ_t=[1.0, 2.0]")
    print("=" * 76)
    radii = np.array([0.5, 1.0])
    sig_t = np.array([1.0, 2.0])
    r_nodes, r_wts = _radial_nodes(radii, n_panels_per_region=2, p_order=3)
    print(f"\n  Radial nodes ({len(r_nodes)}): "
          + " ".join(f"{x:.4f}" for x in r_nodes))

    # Inspection: which nodes lie in which region
    inside = r_nodes < 0.5
    print(f"  Inner region (σ_t=1): nodes {np.where(inside)[0].tolist()}")
    print(f"  Outer region (σ_t=2): nodes {np.where(~inside)[0].tolist()}")

    print("\n--- compute_P_esc_mode (n=1) ---")
    print(f"  {'n_ang':>8}", " ".join(f"node_{i}    " for i in range(len(r_nodes))))
    P_history = []
    for n_ang in (24, 48, 96, 192):
        P = compute_P_esc_mode(SPHERE_1D, r_nodes, radii, sig_t, n_mode=1,
                               n_angular=n_ang, dps=20)
        P_history.append(P)
        print(f"  {n_ang:>8}", " ".join(f"{x:+10.6f}" for x in P))

    print("\n  Δ between successive refinements (should ~0):")
    for k in range(1, len(P_history)):
        diff = np.abs(P_history[k] - P_history[k-1])
        print(f"  {24*(2**k):>8}", " ".join(f"{x:10.2e}" for x in diff))

    print("\n--- compute_G_bc_mode (n=1) ---")
    print(f"  {'n_surf':>8}", " ".join(f"node_{i}    " for i in range(len(r_nodes))))
    G_history = []
    for n_surf in (24, 48, 96, 192):
        G = compute_G_bc_mode(SPHERE_1D, r_nodes, radii, sig_t, n_mode=1,
                              n_surf_quad=n_surf, dps=20)
        G_history.append(G)
        print(f"  {n_surf:>8}", " ".join(f"{x:+10.6f}" for x in G))

    print("\n  Δ between successive refinements (should ~0):")
    for k in range(1, len(G_history)):
        diff = np.abs(G_history[k] - G_history[k-1])
        print(f"  {24*(2**k):>8}", " ".join(f"{x:10.2e}" for x in diff))

    # Bonus: compare mode-0 (legacy compute_P_esc/G_bc) for sanity
    print("\n--- compute_P_esc (mode 0, legacy)  ---")
    P0_history = []
    for n_ang in (24, 48, 96):
        P0 = compute_P_esc(SPHERE_1D, r_nodes, radii, sig_t,
                           n_angular=n_ang, dps=20)
        P0_history.append(P0)
        print(f"  {n_ang:>8}", " ".join(f"{x:+10.6f}" for x in P0))

    print("\n--- compute_G_bc (mode 0, legacy) ---")
    G0_history = []
    for n_surf in (24, 48, 96):
        G0 = compute_G_bc(SPHERE_1D, r_nodes, radii, sig_t,
                          n_surf_quad=n_surf, dps=20)
        G0_history.append(G0)
        print(f"  {n_surf:>8}", " ".join(f"{x:+10.6f}" for x in G0))


if __name__ == "__main__":
    main()
