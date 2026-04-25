"""Probe E: per-node conservation defect for the failing case.

Created by numerics-investigator on 2026-04-24.

Per probe-cascade: localize the bug spatially using the per-node
fixed-source identity. The Peierls equation in pure-absorber form,
with φ ≡ 1, reduces to

    Σ_t(r_i) · 1 = Σ_j K_ij · Σ_t(r_j) · 1

i.e. (K · Σ_t)[i] = Σ_t[i] per node — the "K-collapses-Σ_t" identity.

This probe shows where the rank-N closure violates this identity, separately
for K_vol, K_bc, and K_total = K_vol + K_bc, at sphere 1G/2R-Z config.

Also, contrast with sphere 1G/1R-control (σ_t=1, radii=[1.0]) where
rank-2 is known to give -1.10 % k_eff — the conservation defect should be
small there.
"""

from __future__ import annotations

import numpy as np

from orpheus.derivations._xs_library import get_xs
from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    build_volume_kernel,
    build_white_bc_correction,
    build_white_bc_correction_rank_n,
    composite_gl_r,
)


def _conservation_table(label, radii, sig_t, n_bc_modes, *, quad):
    print(f"\n--- {label} (rank-{n_bc_modes}, BASE quadrature) ---")
    r_nodes, r_wts, panels = composite_gl_r(
        radii, quad["n_panels_per_region"], quad["p_order"],
        dps=quad["dps"], inner_radius=0.0,
    )
    K_vol = build_volume_kernel(
        SPHERE_1D, r_nodes, panels, radii, sig_t,
        n_angular=quad["n_angular"], n_rho=quad["n_rho"], dps=quad["dps"],
    )
    K_bc = build_white_bc_correction_rank_n(
        SPHERE_1D, r_nodes, r_wts, radii, sig_t,
        n_angular=quad["n_angular"], n_surf_quad=quad["n_surf_quad"],
        dps=quad["dps"], n_bc_modes=n_bc_modes,
    )
    sig_t_n = np.array([
        sig_t[SPHERE_1D.which_annulus(float(r_nodes[i]), radii)]
        for i in range(len(r_nodes))
    ])
    Kvol_st = K_vol @ sig_t_n
    Kbc_st = K_bc @ sig_t_n
    Ktot_st = Kvol_st + Kbc_st

    # Identity: K·σ_t should equal σ_t per node
    print(f"  {'i':>3} {'r_i':>8} {'σ_t,i':>8}"
          f"  {'(K_vol·σ)':>12} {'(K_bc·σ)':>12} {'sum':>12}"
          f"  {'rel_def':>10}")
    for i in range(len(r_nodes)):
        rel_def = (Ktot_st[i] - sig_t_n[i]) / sig_t_n[i]
        print(f"  {i:>3} {r_nodes[i]:>8.4f} {sig_t_n[i]:>8.4f}"
              f"  {Kvol_st[i]:>12.6f} {Kbc_st[i]:>12.6f} {Ktot_st[i]:>12.6f}"
              f"  {rel_def:>+10.3e}")
    rel_defs = (Ktot_st - sig_t_n) / sig_t_n
    print(f"\n  max |rel defect| = {np.max(np.abs(rel_defs)):.4e}")
    print(f"  rms  rel defect  = {np.sqrt(np.mean(rel_defs**2)):.4e}")


def main():
    quad = dict(n_panels_per_region=2, p_order=3,
                n_angular=24, n_rho=24, n_surf_quad=24, dps=15)
    print("=" * 76)
    print("Probe E: conservation defect (K·σ_t vs σ_t) per radial node, sphere")
    print("=" * 76)

    # Control: 1G/1R sphere — rank-2 known to give -1.10 % k_eff (good).
    _conservation_table(
        "control 1R: σ_t=[1.0], R=1.0",
        np.array([1.0]), np.array([1.0]),
        n_bc_modes=2, quad=quad,
    )
    # Suspect: 1G/2R sphere — rank-2 gives +57 % k_eff.
    _conservation_table(
        "suspect 2R: σ_t=[1.0, 2.0], R=[0.5, 1.0]",
        np.array([0.5, 1.0]), np.array([1.0, 2.0]),
        n_bc_modes=2, quad=quad,
    )
    # Compare rank-1 and rank-3 on the same suspect to see the trend.
    _conservation_table(
        "suspect 2R: σ_t=[1.0, 2.0], R=[0.5, 1.0]",
        np.array([0.5, 1.0]), np.array([1.0, 2.0]),
        n_bc_modes=1, quad=quad,
    )
    _conservation_table(
        "suspect 2R: σ_t=[1.0, 2.0], R=[0.5, 1.0]",
        np.array([0.5, 1.0]), np.array([1.0, 2.0]),
        n_bc_modes=3, quad=quad,
    )

    # Also: 2R-homog-σ control (=1.0 throughout, but radii=[0.5,1])
    _conservation_table(
        "homog-2R: σ_t=[1.0, 1.0], R=[0.5, 1.0]",
        np.array([0.5, 1.0]), np.array([1.0, 1.0]),
        n_bc_modes=2, quad=quad,
    )


if __name__ == "__main__":
    main()
