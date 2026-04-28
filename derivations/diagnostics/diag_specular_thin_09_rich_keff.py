"""Diagnostic 09: confirm RICH-quadrature plateau is structural for thin sphere.

Created by numerics-investigator on 2026-04-27.

User reports BASE quadrature plateau ~-5.2% at N=6, RICH plateau
~-6.3% at N=8. Confirm RICH numbers and extend to N=10, 12.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, CYLINDER_1D, SLAB_POLAR_1D,
    solve_peierls_1g,
)


def _solve(geom, R, sigt, sigs, nuf, *, n_bc_modes, boundary, level="BASE"):
    if level == "BASE":
        kw = dict(p_order=4, n_panels_per_region=2, n_angular=24, n_rho=24,
                  n_surf_quad=24, dps=20)
    elif level == "RICH":
        kw = dict(p_order=6, n_panels_per_region=4, n_angular=48, n_rho=48,
                  n_surf_quad=48, dps=30)
    elif level == "ULTRA":
        kw = dict(p_order=8, n_panels_per_region=6, n_angular=64, n_rho=64,
                  n_surf_quad=64, dps=40)
    radii = np.array([R])
    sig_t = np.array([sigt])
    sig_s_arr = np.array([[sigs]])
    nu_sig_f = np.array([nuf])
    return solve_peierls_1g(
        geom, radii, sig_t, sig_s_arr, nu_sig_f,
        boundary=boundary, n_bc_modes=n_bc_modes, tol=1e-10, **kw,
    )


@pytest.mark.parametrize("level", ["BASE", "RICH"])
def test_thin_sphere_quadrature_plateau(level, capsys):
    R = 5.0; sigt = 0.5; sigs = 0.38; nuf = 0.025
    k_inf = nuf / (sigt - sigs)
    with capsys.disabled():
        print(f"\n=== thin τR=2.5, k_inf={k_inf:.6f}, level={level} ===")
        for N in (1, 2, 4, 6, 8, 10):
            sol = _solve(SPHERE_1D, R, sigt, sigs, nuf,
                         n_bc_modes=N, boundary="specular", level=level)
            err = (sol.k_eff - k_inf) / k_inf
            print(f"  N={N}: k_eff={sol.k_eff:.8f} err={err*100:+.4f}%")


if __name__ == "__main__":
    R = 5.0; sigt = 0.5; sigs = 0.38; nuf = 0.025
    k_inf = nuf / (sigt - sigs)
    for level in ["BASE", "RICH"]:
        print(f"\n=== thin τR=2.5, k_inf={k_inf:.6f}, level={level} ===")
        for N in (1, 2, 4, 6, 8, 10):
            sol = _solve(SPHERE_1D, R, sigt, sigs, nuf,
                         n_bc_modes=N, boundary="specular", level=level)
            err = (sol.k_eff - k_inf) / k_inf
            print(f"  N={N}: k_eff={sol.k_eff:.8f} err={err*100:+.4f}%")
