"""Diagnostic 04: Does specular need a Hébert-style (1 - P_ss)⁻¹ correction
on top of partial-current matching?

Created by numerics-investigator on 2026-04-27.

Diag 03 finding: Hébert recovers k_inf to -0.27% on thin sphere; specular
plateaus at -5.2% even at high N. The diagnostic 02 was misleading —
σ_t·K·1 ≈ 0.5 for all closures because the thin-cell eigenvector is
NOT uniform.

This probe: monkey-patch _build_full_K_per_group to ALSO multiply the
specular K_bc by 1/(1-P_ss) (the Hébert geometric-series factor). If
this single change makes specular converge to k_inf at thin cells, then:

  Specular as derived = "rank-N partial-current matching, single bounce"
  Specular = ?         = "rank-N partial-current + multi-bounce"

If yes, the fix is to wrap K_bc^specular in /(1 - P_ss) the same way
Hébert wraps Mark.

If NO, then there's a different structural defect — for example, the
partial-current matching ITSELF is incomplete on the rank-N shifted-
Legendre basis truncation (it preserves J^-_m = J^+_m at the truncated
moments, but not the full angular re-emission spectrum).
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, CYLINDER_1D,
    _build_full_K_per_group,
    build_volume_kernel,
    composite_gl_r,
    compute_P_ss_sphere,
    compute_P_ss_cylinder,
)


def _build_specular_K_with_factor(
    geom, R, sigt, *, n_bc_modes, factor_kbc,
):
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
    K_bc = K_full - K_vol
    return r_nodes, r_wts, K_vol + factor_kbc * K_bc


def _solve_with_K(K, r_wts, geometry, sigt, sigs, nuf):
    """Power-iterate K φ = (1/k) (Σ_t I - σ_s K)⁻¹ νΣ_f K φ.

    Equivalent to A φ = (1/k) B φ for A = Σ_t I - σ_s K, B = νΣ_f K.
    """
    N = K.shape[0]
    A = sigt * np.eye(N) - sigs * K
    B = nuf * K
    M = np.linalg.solve(A, B)
    eigvals, eigvecs = np.linalg.eig(M)
    # Take the largest real eigenvalue
    real_mask = np.abs(eigvals.imag) < 1e-10
    real_vals = eigvals[real_mask].real
    real_vecs = eigvecs[:, real_mask].real
    idx = np.argmax(real_vals)
    return float(real_vals[idx]), real_vecs[:, idx]


@pytest.mark.parametrize(
    "tag,R,sigt,sigs,nuf",
    [
        ("thin τR=2.5 sphere", 5.0, 0.5, 0.38, 0.025),
        ("thick τR=5.0 sphere fuelA", 5.0, 1.0, 0.5, 0.75),
    ],
)
def test_hebert_factor_on_specular(tag, R, sigt, sigs, nuf, capsys):
    with capsys.disabled():
        k_inf = nuf / (sigt - sigs)
        radii = np.array([R])
        sig_t_g = np.array([sigt])
        P_ss = compute_P_ss_sphere(
            radii, sig_t_g, n_quad=24, dps=20,
        )
        heb = 1.0 / (1.0 - P_ss)
        print(f"\n=== {tag}: σ_t={sigt}, k_inf={k_inf:.6f}, "
              f"P_ss={P_ss:.6f}, 1/(1-P_ss)={heb:.6f} ===")

        for N in (1, 2, 4, 6):
            for label, factor in [("baseline", 1.0), ("× Hébert", heb)]:
                rn, rw, K = _build_specular_K_with_factor(
                    SPHERE_1D, R, sigt, n_bc_modes=N, factor_kbc=factor,
                )
                k_eff, _ = _solve_with_K(K, rw, SPHERE_1D, sigt, sigs, nuf)
                err = (k_eff - k_inf) / k_inf
                print(f"  N={N} {label:10s}: k_eff={k_eff:.8f} "
                      f"err={err*100:+.4f}%")


if __name__ == "__main__":
    import sys
    for tag, R, sigt, sigs, nuf in [
        ("thin τR=2.5 sphere", 5.0, 0.5, 0.38, 0.025),
        ("thick τR=5.0 sphere fuelA", 5.0, 1.0, 0.5, 0.75),
    ]:
        k_inf = nuf / (sigt - sigs)
        radii = np.array([R])
        sig_t_g = np.array([sigt])
        P_ss = compute_P_ss_sphere(
            radii, sig_t_g, n_quad=24, dps=20,
        )
        heb = 1.0 / (1.0 - P_ss)
        print(f"\n=== {tag}: σ_t={sigt}, k_inf={k_inf:.6f}, "
              f"P_ss={P_ss:.6f}, 1/(1-P_ss)={heb:.6f} ===")

        for N in (1, 2, 4, 6):
            for label, factor in [("baseline", 1.0), ("× Hebert", heb)]:
                rn, rw, K = _build_specular_K_with_factor(
                    SPHERE_1D, R, sigt, n_bc_modes=N, factor_kbc=factor,
                )
                k_eff, _ = _solve_with_K(K, rw, SPHERE_1D, sigt, sigs, nuf)
                err = (k_eff - k_inf) / k_inf
                print(f"  N={N} {label:10s}: k_eff={k_eff:.8f} "
                      f"err={err*100:+.4f}%")
    sys.exit(0)
