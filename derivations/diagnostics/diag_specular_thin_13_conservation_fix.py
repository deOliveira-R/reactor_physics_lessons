"""Diagnostic 13: enforce neutron conservation by rescaling K_bc.

Created by numerics-investigator on 2026-04-27.

Diag 12 showed rank-N specular K has implied leakage 26-44% on thin
cell. Hébert's Mark + (1-P_ss)⁻¹ has only 1.3% leakage. Hypothesis:
the rank-N specular K_bc is missing the contribution from truncated
high-N angular modes. We can compensate by rescaling K_bc so that
the resulting K satisfies P_leak = 0 for the dominant eigenvector.

Strategy: find scalar α such that K = K_vol + α·K_bc gives the
dominant eigenvector v with leak = 0. Equivalently, k_eff = k_inf for
the homogeneous case. This is the "implied factor" we tried in diag 07
but now searching with k_inf as the target instead of constant root.

Diag 07 showed the search fails for some N (k_eff(α) is not monotonic
through k_inf). That's because at high N the K_bc structure is wrong
in shape (not just magnitude). A scalar fix can't compensate for the
shape error.

Diagnostic alternative: rescale K_bc per ROW (i.e., scale K_bc[i, :] by
factor f_i chosen so that K[i, :]·1·Σ_t = 1 for the row OR so that
P_leak per cell volume is 0).

This isn't a clean closure but it WILL diagnose whether the per-row
K_bc magnitude (vs the shape across columns) is the dominant issue.
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
    return r_nodes, r_wts, K_vol, K_full


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
    return float(real_vals[idx]), v


def test_per_row_renormalization(capsys):
    """Force row sum σ_t·K·1 = 1 by per-row rescaling K_bc."""
    R = 5.0; sigt = 0.5; sigs = 0.38; nuf = 0.025
    sig_a = sigt - sigs
    k_inf = nuf / sig_a
    with capsys.disabled():
        print(f"\n=== thin sphere τR=2.5, k_inf={k_inf:.6f} ===")
        for N in (1, 2, 4, 6):
            rn, rw, Kv, Kfull = _build_K(R, sigt, n_bc_modes=N,
                                          closure="specular")
            Kb = Kfull - Kv

            # Per-row factor needed so σ_t·(K_vol + f_i·K_bc)·1 [i] = 1
            #   → σ_t·K_vol·1 [i] + f_i · σ_t·K_bc·1 [i] = 1
            #   → f_i = (1 - σ_t·K_vol·1 [i]) / (σ_t·K_bc·1 [i])
            row_v = sigt * (Kv @ np.ones(Kv.shape[1]))
            row_b = sigt * (Kb @ np.ones(Kb.shape[1]))
            f_per_row = (1.0 - row_v) / row_b
            print(f"\n  N={N}: per-row K_bc factors needed: "
                  f"min={f_per_row.min():.4f}, max={f_per_row.max():.4f}, "
                  f"mean={f_per_row.mean():.4f}")

            # Apply per-row rescale to K_bc
            Kb_scaled = f_per_row[:, None] * Kb
            K_fixed = Kv + Kb_scaled
            k, v = _eig(K_fixed, sigt, sigs, nuf)
            err = (k - k_inf) / k_inf
            print(f"  N={N}: per-row-renormalized k_eff = {k:.6f} "
                  f"({err*100:+.4f}%)")


if __name__ == "__main__":
    R = 5.0; sigt = 0.5; sigs = 0.38; nuf = 0.025
    sig_a = sigt - sigs
    k_inf = nuf / sig_a
    print(f"\n=== thin sphere τR=2.5, k_inf={k_inf:.6f} ===")
    for N in (1, 2, 4, 6):
        rn, rw, Kv, Kfull = _build_K(R, sigt, n_bc_modes=N,
                                      closure="specular")
        Kb = Kfull - Kv
        row_v = sigt * (Kv @ np.ones(Kv.shape[1]))
        row_b = sigt * (Kb @ np.ones(Kb.shape[1]))
        f_per_row = (1.0 - row_v) / row_b
        Kb_scaled = f_per_row[:, None] * Kb
        K_fixed = Kv + Kb_scaled
        k, v = _eig(K_fixed, sigt, sigs, nuf)
        err = (k - k_inf) / k_inf
        print(f"  N={N}: per-row factors min/max/mean = "
              f"{f_per_row.min():.3f}/{f_per_row.max():.3f}/{f_per_row.mean():.3f}, "
              f"k_eff = {k:.6f} ({err*100:+.4f}%)")
