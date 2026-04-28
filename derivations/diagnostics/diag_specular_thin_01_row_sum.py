"""Diagnostic 01: K·1 row-sum profile for thin homogeneous sphere specular.

Created by numerics-investigator on 2026-04-27.

For a homogeneous infinite-medium-equivalent cell, k_inf recovery requires
that the Peierls operator K satisfies

    Σ_t · K[i, :] · 1 = 1

(within the "uniform-flux" subspace) for every spatial node i. We test
this for the thin-cell (τ_R = 2.5) specular sphere and compare against
the thick cell (τ_R = 5) — the working case.

We separately compute K_vol·1 and K_bc·1 to localise where the deviation
from the contract lives.

If this test catches a regression after the fix, promote to
``tests/derivations/test_peierls_specular_bc.py`` as
``test_specular_K1_row_sum_thin_homogeneous``.
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


def _build_K_components(
    geometry, R, sigt, *, n_bc_modes, closure="specular", p_order=4,
    n_panels=2, n_angular=24, n_rho=24, n_surf_quad=24, dps=20,
):
    radii = np.array([R])
    sig_t_g = np.array([sigt])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, n_panels, p_order, dps=dps,
    )
    K_vol = build_volume_kernel(
        geometry, r_nodes, panels, radii, sig_t_g,
        n_angular=n_angular, n_rho=n_rho, dps=dps,
    )
    K_full = _build_full_K_per_group(
        geometry, r_nodes, r_wts, panels, radii, sig_t_g, closure,
        n_angular=n_angular, n_rho=n_rho, n_surf_quad=n_surf_quad,
        dps=dps, n_bc_modes=n_bc_modes,
    )
    K_bc = K_full - K_vol
    return r_nodes, r_wts, K_vol, K_bc


def _print_row_sum_profile(label, r_nodes, K_vol, K_bc, sigt):
    """Print Σ_t · K · 1 vs r/R."""
    R = float(r_nodes[-1] if r_nodes[-1] > 0 else 1.0)
    Kvol_row = sigt * (K_vol @ np.ones(K_vol.shape[1]))
    Kbc_row = sigt * (K_bc @ np.ones(K_bc.shape[1]))
    K_row = Kvol_row + Kbc_row
    print(f"\n=== {label} ===")
    print(
        f"  {'i':>3s} {'r_i/R':>8s} {'σt·Kvol·1':>14s} "
        f"{'σt·Kbc·1':>14s} {'σt·K·1':>14s} {'err':>10s}"
    )
    for i in range(len(r_nodes)):
        ratio = r_nodes[i] / R if R > 0 else 0.0
        err = K_row[i] - 1.0
        print(
            f"  {i:>3d} {ratio:>8.4f} {Kvol_row[i]:>14.8f} "
            f"{Kbc_row[i]:>14.8f} {K_row[i]:>14.8f} {err:>+10.4e}"
        )
    print(
        f"  --- avg σt·K·1 = {K_row.mean():.10f}; "
        f"max |err| = {abs(K_row - 1.0).max():.4e}"
    )
    return K_row


@pytest.mark.parametrize(
    "tag,R,sigt,N_list",
    [
        ("thin sphere τR=2.5", 5.0, 0.5, [1, 2, 4]),
        ("thick sphere τR=5.0", 5.0, 1.0, [1, 2, 4]),
    ],
)
def test_specular_row_sum_profile(tag, R, sigt, N_list, capsys):
    """Print K·1 profile for thin vs thick homogeneous sphere specular."""
    with capsys.disabled():
        for N in N_list:
            r_nodes, r_wts, K_vol, K_bc = _build_K_components(
                SPHERE_1D, R, sigt, n_bc_modes=N,
            )
            _print_row_sum_profile(
                f"{tag}, N={N}", r_nodes, K_vol, K_bc, sigt,
            )


if __name__ == "__main__":
    import sys
    print("Running specular row-sum diagnostic standalone...")
    for tag, R, sigt, N_list in [
        ("thin sphere τR=2.5", 5.0, 0.5, [1, 2, 4, 6]),
        ("thick sphere τR=5.0", 5.0, 1.0, [1, 2, 4, 6]),
    ]:
        for N in N_list:
            r_nodes, r_wts, K_vol, K_bc = _build_K_components(
                SPHERE_1D, R, sigt, n_bc_modes=N,
            )
            _print_row_sum_profile(
                f"{tag}, N={N}", r_nodes, K_vol, K_bc, sigt,
            )
    sys.exit(0)
