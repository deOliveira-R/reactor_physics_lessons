"""Probe: confirm the sphere 1G/2R rank-2 +57% gap at RICH quadrature.

If BASE→RICH stability holds (gap stays ~+50% within a few %), the
gap is structural, not Issue #114 ρ-subdivision noise → confirms the
H_B hypothesis (hidden bug in MR rank-N path).

Also runs the conservation row-sum defect check ((K_vol+K_bc)·1 vs Σ_t,i)
per radial node to surface where the closure breaks down spatially.

Run from repo root: ``python derivations/diagnostics/diag_class_b_rank_n_rich_check.py``
"""

from __future__ import annotations

import time

import numpy as np

from orpheus.derivations import cp_cylinder, cp_sphere
from orpheus.derivations._xs_library import LAYOUTS, get_xs
from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D,
    SPHERE_1D,
    build_volume_kernel,
    build_white_bc_correction_rank_n,
    composite_gl_r,
    solve_peierls_mg,
)

_QUAD_BASE = dict(
    n_panels_per_region=2, p_order=3,
    n_angular=24, n_rho=24, n_surf_quad=24, dps=15,
)
_QUAD_RICH = dict(
    n_panels_per_region=4, p_order=5,
    n_angular=64, n_rho=48, n_surf_quad=64, dps=20,
)


def _build_xs(ng_key: str, n_regions: int):
    layout = LAYOUTS[n_regions]
    xs_list = [get_xs(region, ng_key) for region in layout]
    return (
        np.vstack([xs["sig_t"] for xs in xs_list]),
        np.stack([xs["sig_s"] for xs in xs_list], axis=0),
        np.vstack([xs["nu"] * xs["sig_f"] for xs in xs_list]),
        np.vstack([xs["chi"] for xs in xs_list]),
    )


def _solve(geometry, ng_key, n_regions, n_bc_modes, *, quad, radii_key):
    cp_module = cp_cylinder if radii_key == "cyl" else cp_sphere
    sig_t, sig_s, nu_sig_f, chi = _build_xs(ng_key, n_regions)
    radii = np.array(cp_module._RADII[n_regions])
    sol = solve_peierls_mg(
        geometry, radii=radii, sig_t=sig_t, sig_s=sig_s,
        nu_sig_f=nu_sig_f, chi=chi,
        boundary="white_rank1_mark", n_bc_modes=n_bc_modes,
        **quad,
    )
    return sol.k_eff


def _conservation_defect(geometry, ng_key, n_regions, n_bc_modes, *, quad, radii_key, group_idx=0):
    """Per-node residual of  (K_vol + K_bc) · 1 ≈ 1/Σ_t for group g.

    The Peierls fixed-source identity in pure absorber form:
        Σ_t(r_i) φ(r_i) = sum_j K_ij Σ_t(r_j) φ(r_j) ⇒ K · 1 = 1/Σ_t · Σ_t = 1
    when φ ≡ 1 and Σ_t is constant. With piecewise Σ_t, the identity
    becomes K · Σ_t = 1 (where K · 1 = 1/Σ_t per node).

    Returns the absolute defect array per radial node.
    """
    cp_module = cp_cylinder if radii_key == "cyl" else cp_sphere
    sig_t, sig_s, nu_sig_f, chi = _build_xs(ng_key, n_regions)
    radii = np.array(cp_module._RADII[n_regions])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, quad["n_panels_per_region"], quad["p_order"],
        dps=quad["dps"],
        inner_radius=getattr(geometry, "inner_radius", 0.0) or 0.0,
    )
    sig_t_g = sig_t[:, group_idx]
    K_vol = build_volume_kernel(
        geometry, r_nodes, panels, radii, sig_t_g,
        n_angular=quad["n_angular"], n_rho=quad["n_rho"], dps=quad["dps"],
    )
    K_bc = build_white_bc_correction_rank_n(
        geometry, r_nodes, r_wts, radii, sig_t_g,
        n_angular=quad["n_angular"], n_surf_quad=quad["n_surf_quad"],
        dps=quad["dps"], n_bc_modes=n_bc_modes,
    )
    K = K_vol + K_bc
    # K · Σ_t  should be 1 per node (the fixed-source identity)
    sig_t_n = np.array([
        sig_t_g[geometry.which_annulus(float(r_nodes[i]), radii)]
        for i in range(len(r_nodes))
    ])
    Kq = K @ sig_t_n
    return r_nodes, sig_t_n, Kq, np.abs(Kq - 1.0)


def main():
    print("=" * 76)
    print("Class B rank-N MR×MG — RICH quadrature stability check (H_B confirmation)")
    print("=" * 76)

    suspects = [
        # (label, geometry, ng_key, n_regions, n_bc_modes)
        ("sph 1G/2R rank-1", SPHERE_1D, "1g", 2, 1),
        ("sph 1G/2R rank-2", SPHERE_1D, "1g", 2, 2),
        ("sph 1G/2R rank-3", SPHERE_1D, "1g", 2, 3),
        ("sph 1G/1R rank-2", SPHERE_1D, "1g", 1, 2),  # control: works well at BASE
        ("cyl 1G/2R rank-1", CYLINDER_1D, "1g", 2, 1),
        ("cyl 1G/2R rank-2", CYLINDER_1D, "1g", 2, 2),
        ("cyl 1G/1R rank-2", CYLINDER_1D, "1g", 1, 2),  # control
    ]

    print(f"\n{'case':<30} {'k_BASE':>14} {'k_RICH':>14} {'Δ_BR':>10} {'k_inf':>14} {'err_RICH%':>10}")
    print("-" * 100)
    for label, geom, ng, nr, n in suspects:
        radii_key = "cyl" if geom is CYLINDER_1D else "sph"
        cp_module = cp_cylinder if radii_key == "cyl" else cp_sphere
        kinf = cp_module._build_case(ng, nr).k_inf
        t0 = time.time()
        k_base = _solve(geom, ng, nr, n, quad=_QUAD_BASE, radii_key=radii_key)
        t_base = time.time() - t0
        t0 = time.time()
        k_rich = _solve(geom, ng, nr, n, quad=_QUAD_RICH, radii_key=radii_key)
        t_rich = time.time() - t0
        delta_br = abs(k_rich - k_base) / max(abs(k_rich), 1e-30) * 100
        err_rich_pct = (k_rich - kinf) / kinf * 100
        print(
            f"{label:<30} {k_base:>14.8f} {k_rich:>14.8f} {delta_br:>9.3f}% "
            f"{kinf:>14.8f} {err_rich_pct:>9.2f}%   ({t_base:.1f}s|{t_rich:.1f}s)"
        )

    print("\n=== Conservation defect (K·Σ_t vs 1) at sphere 1G/2R, group 0 ===")
    print("       (rank-2 should be more conservative than rank-1 per L18)")
    for n_bc in (1, 2, 3):
        r_nodes, sig_t_n, Kq, defect = _conservation_defect(
            SPHERE_1D, "1g", 2, n_bc, quad=_QUAD_BASE, radii_key="sph",
        )
        print(
            f"\n  rank-{n_bc} sph 1G/2R group-0:  "
            f"max defect = {defect.max():.4e}  median = {np.median(defect):.4e}"
        )
        # Show a few representative nodes
        idx = [0, len(r_nodes)//4, len(r_nodes)//2, 3*len(r_nodes)//4, -1]
        for i in idx:
            print(
                f"    r={r_nodes[i]:.4f}  Σ_t={sig_t_n[i]:.3f}  "
                f"K·Σ_t={Kq[i]:.6f}  |defect|={defect[i]:.4e}"
            )


if __name__ == "__main__":
    main()
