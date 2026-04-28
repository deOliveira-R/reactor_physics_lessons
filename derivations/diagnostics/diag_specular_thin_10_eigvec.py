"""Diagnostic 10: actual eigenvector profile for specular vs Hébert
on thin sphere.

Created by numerics-investigator on 2026-04-27.

Both Hébert (k_eff = -0.27%) and specular (-5.2% to -8%) act on the
SAME K_vol (only K_bc differs). Compare the eigenvectors. If they
differ in shape, the redistribution within the cell differs.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, _build_full_K_per_group, build_volume_kernel,
    composite_gl_r,
)


def _build_K(R, sigt, *, n_bc_modes, closure):
    radii = np.array([R])
    sig_t_g = np.array([sigt])
    r_nodes, r_wts, panels = composite_gl_r(radii, 2, 4, dps=20)
    K_vol = build_volume_kernel(SPHERE_1D, r_nodes, panels, radii, sig_t_g,
                                n_angular=24, n_rho=24, dps=20)
    K_full = _build_full_K_per_group(SPHERE_1D, r_nodes, r_wts, panels,
        radii, sig_t_g, closure,
        n_angular=24, n_rho=24, n_surf_quad=24, dps=20,
        n_bc_modes=n_bc_modes)
    return r_nodes, K_vol, K_full


def _eig(K, sigt, sigs, nuf):
    A = sigt * np.eye(K.shape[0]) - sigs * K
    B = nuf * K
    eigvals, eigvecs = np.linalg.eig(np.linalg.solve(A, B))
    real_mask = np.abs(eigvals.imag) < 1e-10
    real_vals = eigvals[real_mask].real
    real_vecs = eigvecs[:, real_mask].real
    idx = np.argmax(real_vals)
    v = real_vecs[:, idx]
    if v.sum() < 0:
        v = -v
    return float(real_vals[idx]), v / v.max()


def test_eigvec_compare_thin(capsys):
    R = 5.0; sigt = 0.5; sigs = 0.38; nuf = 0.025
    k_inf = nuf / (sigt - sigs)
    with capsys.disabled():
        rn, Kv, K_heb = _build_K(R, sigt, n_bc_modes=1,
                                  closure="white_hebert")
        k_h, v_h = _eig(K_heb, sigt, sigs, nuf)
        for N in (1, 2, 4, 6):
            rn, _, K_spec = _build_K(R, sigt, n_bc_modes=N,
                                      closure="specular")
            k_s, v_s = _eig(K_spec, sigt, sigs, nuf)
            err_h = (k_h - k_inf) / k_inf * 100
            err_s = (k_s - k_inf) / k_inf * 100
            print(f"\n=== N={N}: Hebert k={k_h:.6f} ({err_h:+.4f}%) "
                  f"vs Specular k={k_s:.6f} ({err_s:+.4f}%) ===")
            print(f"  i  r_i/R     v_Hebert     v_Specular   "
                  f"v_S/v_H     diff")
            R_cell = float(rn[-1])
            for i in range(len(rn)):
                ratio_r = rn[i] / R_cell
                rat = v_s[i] / v_h[i] if v_h[i] != 0 else float("nan")
                print(f"  {i:>2d} {ratio_r:>6.4f}   {v_h[i]:>10.6f}  "
                      f"{v_s[i]:>10.6f}   {rat:>8.4f}   "
                      f"{v_s[i]-v_h[i]:>+8.4f}")


if __name__ == "__main__":
    R = 5.0; sigt = 0.5; sigs = 0.38; nuf = 0.025
    k_inf = nuf / (sigt - sigs)
    rn, Kv, K_heb = _build_K(R, sigt, n_bc_modes=1, closure="white_hebert")
    k_h, v_h = _eig(K_heb, sigt, sigs, nuf)
    for N in (1, 2, 4, 6):
        rn, _, K_spec = _build_K(R, sigt, n_bc_modes=N, closure="specular")
        k_s, v_s = _eig(K_spec, sigt, sigs, nuf)
        err_h = (k_h - k_inf) / k_inf * 100
        err_s = (k_s - k_inf) / k_inf * 100
        print(f"\n=== N={N}: Hebert k={k_h:.6f} ({err_h:+.4f}%) vs "
              f"Specular k={k_s:.6f} ({err_s:+.4f}%) ===")
        R_cell = float(rn[-1])
        print("  i  r_i/R     v_Hebert     v_Specular   v_S/v_H     diff")
        for i in range(len(rn)):
            ratio_r = rn[i] / R_cell
            rat = v_s[i] / v_h[i] if v_h[i] != 0 else float("nan")
            print(f"  {i:>2d} {ratio_r:>6.4f}   {v_h[i]:>10.6f}  "
                  f"{v_s[i]:>10.6f}   {rat:>8.4f}   "
                  f"{v_s[i]-v_h[i]:>+8.4f}")
