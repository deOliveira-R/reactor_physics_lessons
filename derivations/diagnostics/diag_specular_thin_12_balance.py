"""Diagnostic 12: neutron balance for specular vs Hébert thin sphere.

Created by numerics-investigator on 2026-04-27.

Counter to my "physical plateau" hypothesis: for a HOMOGENEOUS cell
with PERFECT specular BC, neutrons cannot escape. So:

    rate of neutron production / rate of neutron loss
       = νΣ_f ∫φ dV / Σ_a ∫φ dV
       = νΣ_f / Σ_a
       = k_inf  (regardless of φ distribution, as long as φ uniform
                 in COMPOSITION — for homogeneous cell σ_t etc are
                 constants so the integrals factor)

If specular gives k_eff ≠ k_inf for a homogeneous cell, then either:
  (a) there is residual numerical leakage (the truncated specular BC
      lets some neutrons escape that perfect specular wouldn't);
  (b) the eigenvector is wrong (not the dominant eigenvalue);
  (c) my derivation is wrong somewhere.

This diagnostic computes:
  P_fis = νΣ_f · ∫φ dV  (rate of neutron production)
  P_abs = (Σ_t - σ_s) · ∫φ dV = Σ_a · ∫φ dV  (rate of net loss)
  P_leak = something — for the K formulation, leak is implicit if
           k_eff < νΣ_f/Σ_a.

For perfect specular, P_leak = 0, so k_eff = P_fis/P_abs = k_inf.
If specular k_eff = 0.195 vs k_inf = 0.208 (-6%), there's effectively
6% leakage — verify by computing the implied leakage from the
operator equation.
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
    # Normalize so ∫v dV = 1
    return float(real_vals[idx]), v


def shell_volume(r, w):
    """∫ r^2 dr in spherical shell."""
    return 4.0 * np.pi * np.sum(r ** 2 * w)


def shell_integral(r, w, f):
    return 4.0 * np.pi * np.sum(r ** 2 * w * f)


def test_neutron_balance_specular(capsys):
    R = 5.0; sigt = 0.5; sigs = 0.38; nuf = 0.025
    sig_a = sigt - sigs
    k_inf = nuf / sig_a
    with capsys.disabled():
        print(f"\n=== thin sphere τR=2.5, σ_t={sigt}, σ_a={sig_a}, "
              f"k_inf={k_inf:.6f} ===")

        for closure_name, n_modes in [
            ("white_hebert", 1),
            ("specular", 1),
            ("specular", 2),
            ("specular", 4),
            ("specular", 6),
        ]:
            rn, rw, Kv, K = _build_K(R, sigt, n_bc_modes=n_modes,
                                      closure=closure_name)
            k, v = _eig(K, sigt, sigs, nuf)
            err = (k - k_inf) / k_inf

            # Volume integrals
            Vphi = shell_integral(rn, rw, v)
            P_fis = nuf * Vphi
            P_abs = sig_a * Vphi

            # Implied leakage: balance is (P_fis/k - P_abs - P_leak) = 0
            # → P_leak = P_fis/k - P_abs
            P_leak = P_fis / k - P_abs
            implied_leak_frac = P_leak / P_fis  # leak as fraction of fission
            print(f"  {closure_name:14s} N={n_modes}: k={k:.6f} "
                  f"({err*100:+.4f}%); ∫φdV={Vphi:.4f}; "
                  f"P_fis={P_fis:.4f}, P_abs={P_abs:.4f}, "
                  f"P_leak/P_fis={implied_leak_frac*100:.4f}%")


if __name__ == "__main__":
    R = 5.0; sigt = 0.5; sigs = 0.38; nuf = 0.025
    sig_a = sigt - sigs
    k_inf = nuf / sig_a
    print(f"\n=== thin sphere τR=2.5, σ_t={sigt}, σ_a={sig_a}, "
          f"k_inf={k_inf:.6f} ===")
    for closure_name, n_modes in [
        ("white_hebert", 1),
        ("specular", 1), ("specular", 2), ("specular", 4),
        ("specular", 6),
    ]:
        rn, rw, Kv, K = _build_K(R, sigt, n_bc_modes=n_modes,
                                  closure=closure_name)
        k, v = _eig(K, sigt, sigs, nuf)
        err = (k - k_inf) / k_inf
        Vphi = shell_integral(rn, rw, v)
        P_fis = nuf * Vphi
        P_abs = sig_a * Vphi
        P_leak = P_fis / k - P_abs
        implied_leak_frac = P_leak / P_fis
        print(f"  {closure_name:14s} N={n_modes}: k={k:.6f} "
              f"({err*100:+.4f}%); ∫φdV={Vphi:.4f}; "
              f"P_leak/P_fis={implied_leak_frac*100:.4f}%")
