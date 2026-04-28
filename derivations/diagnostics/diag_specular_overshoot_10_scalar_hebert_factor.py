"""Diagnostic: scalar Hebert factor on bare specular K_bc.

Created by numerics-investigator on 2026-04-27.

Hypothesis: the multi-bounce overshoot at high N is because (I-TR)⁻¹
amplifies different basis modes differently. The CORRECT multi-bounce
correction would be a SCALAR amplification by 1/(1-P_ss).

Test: K_bc^scalar_mb = K_bc^bare / (1 - P_ss). Check k_eff at various N.
If this gives k_inf to high precision, then we've identified the FIX:
use scalar Hebert factor (not matrix (I-TR)⁻¹) on the rank-N bare K_bc.

Note: at N=1, this is exactly the existing white_hebert closure.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _build_full_K_per_group,
    build_volume_kernel,
    composite_gl_r,
    compute_P_ss_sphere,
)


def _solve(K, sigt, sigs, nuf):
    N = K.shape[0]
    A = sigt * np.eye(N) - sigs * K
    B = nuf * K
    M = np.linalg.solve(A, B)
    eigvals = np.linalg.eigvals(M)
    real_mask = np.abs(eigvals.imag) < 1e-10
    return float(eigvals[real_mask].real.max())


@pytest.mark.parametrize(
    "tag,R,sigt,sigs,nuf",
    [
        ("thin τR=2.5", 5.0, 0.5, 0.38, 0.025),
        ("thick τR=5.0 fuelA", 5.0, 1.0, 0.5, 0.75),
        ("very-thin τR=1.0", 5.0, 0.2, 0.16, 0.01),
    ],
)
def test_scalar_hebert_factor_on_bare(tag, R, sigt, sigs, nuf, capsys):
    """Apply scalar 1/(1-P_ss) factor to bare K_bc and check k_eff."""
    with capsys.disabled():
        k_inf = nuf / (sigt - sigs)
        radii = np.array([R])
        sig_t_g = np.array([sigt])
        r_nodes, r_wts, panels = composite_gl_r(radii, 2, 4, dps=20)
        K_vol = build_volume_kernel(
            SPHERE_1D, r_nodes, panels, radii, sig_t_g,
            n_angular=24, n_rho=24, dps=20,
        )
        N_r = len(r_nodes)
        P_ss = compute_P_ss_sphere(radii, sig_t_g, n_quad=128, dps=20)
        scalar_factor = 1.0 / (1.0 - P_ss)

        print(f"\n=== {tag}: σ_t={sigt}, R={R}, k_inf={k_inf:.6f} ===")
        print(f"  P_ss = {P_ss:.6f}, 1/(1-P_ss) = {scalar_factor:.6f}")
        print(f"\n  N | bare k     | bare err    | scalar_hebert k | scalar err  | matrix mb k | mb err")
        print(f"  --|------------|-------------|-----------------|-------------|-------------|--------")

        for N in (1, 2, 3, 4, 6, 8, 10):
            K_bare = _build_full_K_per_group(
                SPHERE_1D, r_nodes, r_wts, panels,
                radii, sig_t_g, "specular",
                n_angular=24, n_rho=24, n_surf_quad=24, dps=20,
                n_bc_modes=N,
            )
            k_bare = _solve(K_bare, sigt, sigs, nuf)
            err_bare = (k_bare - k_inf) / k_inf

            # Scalar Hebert factor: K_total = K_vol + (K_bc_bare) * factor
            K_bc_bare = K_bare - K_vol
            K_scalar = K_vol + K_bc_bare * scalar_factor
            k_scalar = _solve(K_scalar, sigt, sigs, nuf)
            err_scalar = (k_scalar - k_inf) / k_inf

            # Matrix multi-bounce
            K_mb = _build_full_K_per_group(
                SPHERE_1D, r_nodes, r_wts, panels,
                radii, sig_t_g, "specular_multibounce",
                n_angular=24, n_rho=24, n_surf_quad=24, dps=20,
                n_bc_modes=N,
            )
            k_mb = _solve(K_mb, sigt, sigs, nuf)
            err_mb = (k_mb - k_inf) / k_inf

            print(f"  {N:2d}| {k_bare:.6f}   | {err_bare*100:+.3f}%    | "
                  f"{k_scalar:.6f}        | {err_scalar*100:+.3f}%    | "
                  f"{k_mb:.6f}    | {err_mb*100:+.3f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
