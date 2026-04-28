"""Diagnostic: how does the eigenvector shape depend on N for specular_mb.

Created by numerics-investigator on 2026-04-27.

Per-bounce <φ_total> stays clean as N grows but k_eff overshoots more.
This means the eigenvector SHAPE is changing — it must be picking up
modes that contribute to a larger k_eff than the physically correct
"uniform" shape.

For homogeneous sphere, the physical eigenvector should be UNIFORM
(constant in r). Let's see how far the multi-bounce eigvec deviates,
and what causes the deviation.
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


def _solve(K, sigt, sigs, nuf, return_vec=False):
    N = K.shape[0]
    A = sigt * np.eye(N) - sigs * K
    B = nuf * K
    M = np.linalg.solve(A, B)
    eigvals, eigvecs = np.linalg.eig(M)
    real_mask = np.abs(eigvals.imag) < 1e-10
    real_eigs = eigvals[real_mask].real
    real_vecs = eigvecs[:, real_mask].real
    idx = np.argmax(real_eigs)
    if return_vec:
        v = real_vecs[:, idx]
        v = np.abs(v) / np.max(np.abs(v))
        return float(real_eigs[idx]), v
    return float(real_eigs[idx])


def test_eigenvector_shape_vs_N(capsys):
    """Track how the eigvec deviates from uniform as N grows."""
    with capsys.disabled():
        R = 5.0
        sigt = 0.5
        sigs = 0.38
        nuf = 0.025
        k_inf = nuf / (sigt - sigs)
        radii = np.array([R])
        sig_t_g = np.array([sigt])
        r_nodes, r_wts, panels = composite_gl_r(radii, 2, 4, dps=20)
        K_vol = build_volume_kernel(
            SPHERE_1D, r_nodes, panels, radii, sig_t_g,
            n_angular=24, n_rho=24, dps=20,
        )
        N_r = len(r_nodes)
        print(f"\n=== Eigvec shape for thin sphere multi-bounce specular ===")
        print(f"  N_r = {N_r}, r_nodes range [{r_nodes.min():.3f}, {r_nodes.max():.3f}]")

        for N in (1, 2, 4, 6, 8):
            K = _build_full_K_per_group(
                SPHERE_1D, r_nodes, r_wts, panels,
                radii, sig_t_g,
                "specular_multibounce",
                n_angular=24, n_rho=24, n_surf_quad=24, dps=20,
                n_bc_modes=N,
            )
            k_mb, v_mb = _solve(K, sigt, sigs, nuf, return_vec=True)
            err = (k_mb - k_inf) / k_inf

            # Compute eigvec deviation from uniform
            v_uniform = np.ones(N_r)
            v_mb_norm = v_mb / np.linalg.norm(v_mb)
            v_uni_norm = v_uniform / np.linalg.norm(v_uniform)
            angle_deg = np.degrees(np.arccos(min(abs(v_mb_norm @ v_uni_norm), 1.0)))
            ratio = float(v_mb.max() / v_mb.min())
            print(f"  N={N}: k_eff={k_mb:.6f} ({err*100:+.3f}%), "
                  f"eigvec ratio max/min={ratio:.3f}, deviation_from_uniform={angle_deg:.3f}°")
            print(f"    v_mb = {v_mb}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
