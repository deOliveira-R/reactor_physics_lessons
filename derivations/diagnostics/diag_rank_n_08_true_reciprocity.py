"""Compute the TRUE ⟨m^+_n⟩ = (1/A) · ∫_A m^+_n(r_s) dA_s via the 1/μ_s-weighted
formula, and compare to P_esc^(n)(r')/A.

Rigorously:
   ⟨m^+_n⟩ = (1/(2A)) ∫_0^π sin θ P̃_n(μ_s) exp(-τ) / μ_s dθ

where μ_s = (ρ_max + r' cos θ)/R for sphere, and the 1/μ_s comes from
the Jacobian dA_s/dΩ_{r'} = |r_s-r'|²/cos(θ_s) where cos(θ_s) = μ_s.

This is the reciprocity for the n-th SURFACE-AVERAGED LEGENDRE MOMENT of
the outgoing flux due to a unit point source at r'.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/workspaces/ORPHEUS")
import numpy as np

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, gl_float, compute_P_esc_mode, composite_gl_r,
)
from orpheus.derivations._kernels import _shifted_legendre_eval


def brute_m_plus_n(R, sig_t, r_src, n_mode, n_theta=128, dps=25):
    """True ⟨m^+_n⟩ via 1/μ_s-weighted integral."""
    theta_pts, theta_wts = gl_float(n_theta, 0.0, np.pi, dps)
    A = 4 * np.pi * R * R
    total = 0.0
    for k in range(n_theta):
        ct = np.cos(theta_pts[k])
        st = np.sin(theta_pts[k])
        rho_max = -r_src * ct + np.sqrt(max(r_src**2 * ct**2 + R**2 - r_src**2, 0.0))
        tau = sig_t * rho_max
        mu_s = (rho_max + r_src * ct) / R
        if mu_s <= 0.0:
            continue
        p_tilde = float(_shifted_legendre_eval(n_mode, np.array([mu_s]))[0])
        total += theta_wts[k] * st * p_tilde * np.exp(-tau) / mu_s
    return total / (2 * A)


def main():
    R = 2.0
    sig_t = 1.0
    print("⟨m^+_n⟩ (true, with 1/μ_s) vs P_esc^(n)(r')/A")
    print(f"{'r_src':>8} | " + " | ".join(f"n={n}: m+/P_esc·A ratio" for n in range(5)))
    for r_src in [0.01, 0.5, 1.0, 1.5, 1.9]:
        row = [f"{r_src:>8.2f}"]
        for n_mode in range(5):
            m_true = brute_m_plus_n(R, sig_t, r_src, n_mode)
            r_nodes = np.array([r_src])
            radii = np.array([R])
            sig_t_arr = np.array([sig_t])
            P_n = compute_P_esc_mode(SPHERE_1D, r_nodes, radii, sig_t_arr, n_mode,
                                     n_angular=128, dps=25)[0]
            A = 4 * np.pi * R * R
            if P_n != 0:
                # Compare to P_esc/A
                ratio = m_true / (P_n / A)
                row.append(f"  {ratio:+.4f}")
            else:
                row.append(f"   (P_n=0)")
        print(" | ".join(row))


if __name__ == "__main__":
    main()
