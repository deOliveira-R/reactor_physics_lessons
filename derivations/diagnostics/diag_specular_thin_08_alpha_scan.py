"""Diagnostic 08: scan α in K = K_vol + α·K_bc^specular for thin sphere
to map out the relationship.

Created by numerics-investigator on 2026-04-27.

Diagnostic 07 showed brentq failures — k_eff(α) is not monotonic-passing-
through-k_inf. Scan α to see the actual shape.
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


def _build_K_components(geom, R, sigt, *, n_bc_modes):
    radii = np.array([R])
    sig_t_g = np.array([sigt])
    r_nodes, r_wts, panels = composite_gl_r(radii, 2, 4, dps=20)
    K_vol = build_volume_kernel(
        geom, r_nodes, panels, radii, sig_t_g,
        n_angular=24, n_rho=24, dps=20,
    )
    K_full = _build_full_K_per_group(
        geom, r_nodes, r_wts, panels, radii, sig_t_g, "specular",
        n_angular=24, n_rho=24, n_surf_quad=24, dps=20,
        n_bc_modes=n_bc_modes,
    )
    return K_vol, K_full - K_vol


def _solve(K, sigt, sigs, nuf):
    N = K.shape[0]
    A = sigt * np.eye(N) - sigs * K
    B = nuf * K
    eigvals = np.linalg.eigvals(np.linalg.solve(A, B))
    real_mask = np.abs(eigvals.imag) < 1e-10
    if not real_mask.any():
        return float("nan")
    return float(eigvals[real_mask].real.max())


def test_alpha_scan_thin_sphere(capsys):
    with capsys.disabled():
        R = 5.0
        sigt = 0.5
        sigs = 0.38
        nuf = 0.025
        k_inf = nuf / (sigt - sigs)
        print(f"\n=== thin τR=2.5, k_inf={k_inf:.6f} ===")

        for N in (1, 2, 4, 6):
            Kv, Kb = _build_K_components(SPHERE_1D, R, sigt, n_bc_modes=N)
            print(f"\n  N={N}:")
            print(f"    {'α':>8s} {'k_eff':>14s} {'err %':>10s}")
            for alpha in (0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0):
                K = Kv + alpha * Kb
                k = _solve(K, sigt, sigs, nuf)
                err = (k - k_inf) / k_inf * 100 if not np.isnan(k) else float("nan")
                print(f"    {alpha:>8.2f} {k:>14.8f} {err:>+10.4f}")


if __name__ == "__main__":
    R = 5.0; sigt = 0.5; sigs = 0.38; nuf = 0.025
    k_inf = nuf / (sigt - sigs)
    print(f"\n=== thin τR=2.5, k_inf={k_inf:.6f} ===")

    for N in (1, 2, 4, 6):
        Kv, Kb = _build_K_components(SPHERE_1D, R, sigt, n_bc_modes=N)
        print(f"\n  N={N}:")
        print(f"    {'α':>8s} {'k_eff':>14s} {'err %':>10s}")
        for alpha in (0.0, 0.5, 1.0, 1.083, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0):
            K = Kv + alpha * Kb
            k = _solve(K, sigt, sigs, nuf)
            err = (k - k_inf) / k_inf * 100 if not np.isnan(k) else float("nan")
            print(f"    {alpha:>8.3f} {k:>14.8f} {err:>+10.4f}")
