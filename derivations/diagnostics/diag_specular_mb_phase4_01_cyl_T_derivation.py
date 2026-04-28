"""Diagnostic: Cylinder T matrix derivation + rank-1 identity check.

Created by numerics-investigator on 2026-04-28.

DERIVATION
----------
For sphere, ``compute_T_specular_sphere`` builds:

    T_mn^sph = 2 ∫_0^1 µ P̃_m(µ) P̃_n(µ) e^{-τ(µ)} dµ,   τ(µ) = Σ_t · 2Rµ

by composing the surface-to-surface transit with the µ-weighted partial-
current measure. The corresponding cylinder primitives (P/G) use the
Knyazev expansion:

    P_esc^(n,3d)(r_i) = (1/π) ∫_0^π Σ_k c_n^k µ_2D(ω)^k Ki_(k+2)(τ_2D(ω)) dω

for the in-plane chord τ_2D and exit cosine µ_2D. The 3-D direction
cosine is µ_3D = sin θ_p · µ_2D after polar integration.

For the cylinder T_mn (surface-to-surface partial current), a ray exits
at in-plane angle α ∈ [-π/2, π/2] from the inward normal and polar
angle θ_p ∈ [0, π]. The chord across the cell (homogeneous):

    in-plane chord d_2D(α) = 2R cos α       (homogeneous; for multi-region
                                              piecewise via shell intersection)
    3-D chord d_3D = d_2D / sin θ_p
    optical depth τ_3D = τ_2D(α) / sin θ_p
    direction cosine  µ_3D = sin θ_p · cos α

The µ-weighted partial-current measure on inward directions (per inverse
cylinder area, mirroring the (1/π) of P_esc^cyl) gives:

    T_mn^cyl = (4/π) ∫_0^(π/2) cos α · Σ_{k_m,k_n} c_m^{k_m} c_n^{k_n}
                · (cos α)^(k_m+k_n) · Ki_(3+k_m+k_n)(τ_2D(α)) dα

Note: Ki_(3+k_m+k_n), one order HIGHER than the P_esc^cyl Ki_(2+k_n) —
this extra Ki order comes from the additional µ_3D = sin θ_p factor
(partial-current weight) compared to the flux-only measure used in
P_esc.

RANK-1 CHECK
------------
At m=n=0, c_0^0 = 1, so the only term is (4/π) ∫ cos α · Ki_3(τ_2D(α)) dα.
This is exactly compute_P_ss_cylinder.

If T_00^cyl == P_ss^cyl to ~1e-12 the derivation is correct.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    _shifted_legendre_monomial_coefs,
    compute_P_ss_cylinder,
    gl_float,
)
from orpheus.derivations._kernels import ki_n_mp


def build_T_specular_cylinder(
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_modes: int,
    *,
    n_quad: int = 64,
    dps: int = 25,
) -> np.ndarray:
    """Reference cylinder T matrix (homogeneous + multi-region) per
    derivation in the module docstring."""
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    radii_inner = np.concatenate([[0.0], radii[:-1]])
    radii_outer = radii

    alpha_pts, alpha_wts = gl_float(n_quad, 0.0, np.pi / 2.0, dps)

    # Multi-region τ_2D along antipodal in-plane chord at each α.
    tau_arr = np.zeros(n_quad)
    for k in range(n_quad):
        a = float(alpha_pts[k])
        sa = float(np.sin(a))
        h = R * sa
        tau = 0.0
        for n_reg in range(len(radii)):
            r_in = float(radii_inner[n_reg])
            r_out = float(radii_outer[n_reg])
            if h >= r_out:
                continue
            seg_outer = float(np.sqrt(max(r_out * r_out - h * h, 0.0)))
            seg_inner = (
                float(np.sqrt(max(r_in * r_in - h * h, 0.0)))
                if h < r_in else 0.0
            )
            chord_in_annulus = 2.0 * (seg_outer - seg_inner)
            tau += float(sig_t[n_reg]) * chord_in_annulus
        tau_arr[k] = tau

    cos_alpha = np.cos(np.array([float(a) for a in alpha_pts]))

    T = np.zeros((n_modes, n_modes))
    for m in range(n_modes):
        cm = _shifted_legendre_monomial_coefs(m)
        for n in range(n_modes):
            cn = _shifted_legendre_monomial_coefs(n)
            total = 0.0
            for k_q in range(n_quad):
                ca = float(cos_alpha[k_q])
                kernel = 0.0
                for k_m, c_m in enumerate(cm):
                    if c_m == 0.0:
                        continue
                    for k_n, c_n in enumerate(cn):
                        if c_n == 0.0:
                            continue
                        kk = k_m + k_n
                        ki = float(ki_n_mp(kk + 3, float(tau_arr[k_q]), dps))
                        kernel += c_m * c_n * (ca ** kk) * ki
                total += float(alpha_wts[k_q]) * ca * kernel
            T[m, n] = (4.0 / np.pi) * total
    return T


def test_cyl_T_rank1_equals_pss(capsys):
    """T_00^cyl should equal P_ss^cyl exactly at rank-1."""
    with capsys.disabled():
        cases = [
            ("thin", np.array([5.0]), np.array([0.5])),    # τ_R=2.5
            ("thick", np.array([5.0]), np.array([1.0])),   # τ_R=5
            ("very-thin", np.array([5.0]), np.array([0.2])),
            ("MR", np.array([2.0, 5.0]), np.array([0.6, 0.4])),
        ]
        print(f"\n{'case':<12} {'T_00':<15} {'P_ss^cyl':<15} {'rel_err':<10}")
        for name, radii, sig_t in cases:
            T = build_T_specular_cylinder(radii, sig_t, 1, n_quad=128, dps=30)
            Pss = compute_P_ss_cylinder(radii, sig_t, n_quad=128, dps=30)
            rel = abs(T[0, 0] - Pss) / Pss if Pss != 0 else float('nan')
            print(f"{name:<12} {T[0,0]:<15.10f} {Pss:<15.10f} {rel:.2e}")
            assert rel < 1e-10, (
                f"T_00^cyl != P_ss^cyl for {name}: rel_err={rel:.3e}"
            )


def test_cyl_T_smoke(capsys):
    """T matrix should be symmetric (T_mn = T_nm) and decreasing
    in trace as N grows."""
    with capsys.disabled():
        radii = np.array([5.0])
        sig_t = np.array([0.5])

        for N in (1, 2, 3, 4, 6, 8):
            T = build_T_specular_cylinder(radii, sig_t, N, n_quad=128)
            # Symmetry
            sym_err = float(np.max(np.abs(T - T.T)))
            print(f"N={N}: ‖T-Tᵀ‖_max={sym_err:.2e}, trace(T)={np.trace(T):.4e}, "
                  f"T[0,0]={T[0,0]:.4e}")
            assert sym_err < 1e-12, f"T not symmetric at N={N}: max diff {sym_err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
