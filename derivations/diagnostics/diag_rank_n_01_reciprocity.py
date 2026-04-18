"""Diagnostic: test whether G_bc^(n) and P_esc^(n) are related by reciprocity.

Created by numerics-investigator on 2026-04-18 for Issue #112.

Sanchez & McCormick 1982 Eq. (167) asserts per-mode reciprocity relations
between the surface-to-volume and volume-to-surface probabilities. The
observer-centred forms should yield the SAME angular integrand:

    G_bc^(n)(r_i) = ∫_{4π} P̃_n(μ_s) exp(-τ(r_i→r_s)) dΩ_{r_i}
    P_esc^(n)(r_i) = ∫_{4π} P̃_n(μ_exit(r_i,Ω)) K_esc(τ) dΩ_{r_i}

For SPHERE, K_esc = exp(-τ), and μ_exit(r_i, Ω, R) = (rho_max + r_i cos Ω)/R
equals μ_s at the exit point (same cosine with the outward normal at the ray
exit). So these two integrals should be IDENTICAL for the sphere.

For CYLINDER, the existing G_bc uses a surface-centered (phi) integration
with Ki_1(τ)/d instead of the observer-centered (Ω) form. These may or may
not match at n=0 only, with n≥1 diverging — that's the H1 hypothesis.

This diagnostic:
1. Compares G_bc^(0) from observer-centered vs surface-centered forms → must match.
2. Compares G_bc^(n) to P_esc^(n) at SAME nodes to find the right normalization.
3. Diagnoses which normalization factor is missing.
"""
from __future__ import annotations

import numpy as np
import sys

sys.path.insert(0, "/workspaces/ORPHEUS")

from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D, SPHERE_1D,
    compute_G_bc, compute_G_bc_mode,
    compute_P_esc, compute_P_esc_mode,
    composite_gl_r, gl_float,
)
from orpheus.derivations._kernels import _shifted_legendre_eval, ki_n_mp


def compute_G_bc_observer(geometry, r_nodes, radii, sig_t, n_mode,
                          n_angular=32, dps=25):
    """Observer-centered G_bc^(n) - analogue of compute_P_esc_mode with
    K_esc replaced by exp(-τ) for both sphere and cylinder.

    For SPHERE, K_esc = exp(-τ) already, so this MATCHES compute_G_bc_mode.
    For CYLINDER, K_esc = Ki_2(τ) while G_bc uses exp(-τ) in the volume kernel
    — so we override with a direct 3D angular integration here.
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])

    # For observer-centred form: 2D in (theta, phi_azim).
    # Cylinder: theta in [0, pi] weighted by sin(theta), then radial direction
    # becomes like (ρ_max in 2D = R*sin something).
    # For a 1D radial cylinder with azimuthal symmetry of neutron field:
    #   φ(r_i) = (1/(4π)) ∫_{4π} dΩ · ψ_in(r_i, -Ω_hat) · exp(-τ)
    # With boundary source ψ_in corresponding to Marshak mode n.
    # After the r-symmetry fold, we parametrize by polar angle θ (from cyl axis)
    # and azimuthal angle β (in x-y plane). μ_s = sin(θ) * (R - r cos β)/d
    # with d = distance to surface in the (x,y) plane.
    #
    # For mode n, the contribution to scalar flux at r_i is:
    #   G_bc^(n)(r_i) = (prefactor) ∫_0^π dθ sin θ ∫_0^π dβ P̃_n(μ_s)·exp(-τ)
    # where prefactor includes the 1/π from 2π azim / 2π / (hemi) / etc.
    #
    # Going the purely observer-centered route (bypassing the Ki_1 form):

    N = len(r_nodes)
    G = np.zeros(N)

    if geometry.kind == "sphere-1d":
        # Same as compute_G_bc_mode — should match exactly.
        return compute_G_bc_mode(
            geometry, r_nodes, radii, sig_t, n_mode,
            n_surf_quad=n_angular, dps=dps,
        )

    # CYLINDER — direct observer-centered 3D quadrature.
    # θ ∈ [0, π] (polar angle from z-axis), β ∈ [0, π] (azimuth in x-y,
    # folded by symmetry from 2π). The full 3D exit cosine μ_s satisfies:
    # μ_s = sin(θ) · cos(angle of ray in x-y with outward surface normal)
    # For the 2D geometry, if the in-plane ray exits at angular β_surf
    # measured from r_i, the exit point is at R with the outward normal
    # along the radial direction at that point. The cosine is (R - r_i cos β)/d_2D.

    # Let's parametrize differently: cylinder is 2D radial geometry,
    # so a ray from r_i in direction (θ, φ) where φ is in the x-y plane:
    #   - 2D ray direction: (cos φ, sin φ)
    #   - 2D path length: d_2D(r_i, φ) = -r_i cos φ + sqrt(R² - r_i² sin² φ)
    #     (assuming φ = 0 points towards +x and r_i is on +x axis)
    #     Actually: with r_i = (r_i, 0, 0), direction (cos φ, sin φ, 0) wrapped with sin θ:
    #   - 3D ray direction: (sin θ cos φ, sin θ sin φ, cos θ)
    #   - Ray hits cyl surface when (r_i + sin θ cos φ · s)² + (sin θ sin φ · s)² = R²
    #     → s · sin θ = -r_i cos φ + sqrt(R² - r_i² sin² φ) = d_2D
    #     → s_3D = d_2D / sin θ (physical path length)
    #   - τ = Σ_t · s_3D = Σ_t · d_2D / sin θ
    #   - Exit point (same 2D coords): (R cos α, R sin α, *)
    #   - Outward normal in cyl: radial only = (cos α, sin α, 0)
    #   - Full 3D cosine: μ_s = sin θ · cos(angle between (cos φ, sin φ) and (cos α, sin α))
    #   - That 2D angle: cos(α - φ) where tan α = (r_i sin 0 + s_3D sin θ sin φ) / (r_i + s_3D sin θ cos φ) ...

    # Use the 2D exit cosine already: |μ_s_2D| = (R - r_i cos β)/d where β is
    # the polar angle in 2D of the ray direction (NOT of the surface point).
    # Actually: in 2D, β is the angle between the ray direction and the
    # positive x-axis. The in-plane exit cosine:
    #   μ_s_2D = (R - r_i cos β)/d_2D  (validated by the existing code)
    # Full 3D μ_s = sin θ · μ_s_2D.

    # Quadrature in (β, θ):
    beta_pts, beta_wts = gl_float(n_angular, 0.0, np.pi, dps)  # ray direction angle
    theta_pts, theta_wts = gl_float(n_angular, 0.0, np.pi, dps)  # polar from z

    cos_betas = np.cos(beta_pts)
    sin_thetas = np.sin(theta_pts)

    # Prefactor derivation for cylinder G_bc:
    # φ_bc(r_i) via 3D point kernel integration, fold β∈[0,2π]→[0,π] (mirror),
    # θ∈[0,π]. On a unit sphere at r_i, the kernel is (1/(4π)) · exp(-τ) per
    # solid angle.  But for an INWARD isotropic partial current at the lateral
    # surface, the angular flux at r_s is isotropic over inward hemisphere,
    # ψ_in = J^-/π.  Then φ_bc(r_i) = (J^-/π) ∫_{inward} dΩ_{r_i} exp(-τ).
    # Per unit J^-: G_bc(r_i) = (1/π) · ∫_{inward} dΩ exp(-τ).
    # Expressed in (β, θ): dΩ = sin θ dθ dβ, β folded gives factor 2.
    # G_bc^(n)(r_i) = (2/π) ∫_0^π dβ ∫_0^π dθ sin θ · P̃_n(sin θ · μ_s_2D) · exp(-τ)

    pref_3d = 2.0 / np.pi

    for i in range(N):
        r_i = r_nodes[i]
        total = 0.0
        for kb in range(n_angular):
            cb = cos_betas[kb]
            d_sq = r_i * r_i + R * R - 2.0 * r_i * R * cb
            d_2d = np.sqrt(max(d_sq, 0.0))
            if d_2d <= 0.0:
                continue
            mu_s_2d = (R - r_i * cb) / d_2d
            for kt in range(n_angular):
                st = sin_thetas[kt]
                if st <= 0.0:
                    continue
                mu_s_3d = st * mu_s_2d
                tau = sig_t[0] * d_2d / st
                p_tilde = float(
                    _shifted_legendre_eval(n_mode, np.array([mu_s_3d]))[0]
                )
                total += (beta_wts[kb] * theta_wts[kt]
                          * st * p_tilde * np.exp(-tau))
        G[i] = pref_3d * total
    return G


def test_sphere_G_bc_mode_equals_P_esc_mode():
    """Sphere: G_bc^(n) should equal P_esc^(n) numerically."""
    R = 5.0
    radii = np.array([R])
    sig_t = np.array([1.0])
    r_nodes, r_wts, _ = composite_gl_r(radii, 2, 5, dps=25)

    for n in range(0, 4):
        G = compute_G_bc_mode(SPHERE_1D, r_nodes, radii, sig_t, n,
                              n_surf_quad=32, dps=25)
        P = compute_P_esc_mode(SPHERE_1D, r_nodes, radii, sig_t, n,
                               n_angular=32, dps=25)
        # Normalized by ratio at r=0 for direct comparison
        print(f"\nSphere R={R}, n={n}:")
        print(f"  G_bc[0]   = {G[0]:.6e}, G_bc[-1]  = {G[-1]:.6e}")
        print(f"  P_esc[0]  = {P[0]:.6e}, P_esc[-1] = {P[-1]:.6e}")
        # Look at profile shape via ratio
        ratio = G / P
        print(f"  G/P range: [{ratio.min():.4f}, {ratio.max():.4f}]  (constant = reciprocity up to normalization)")


def test_cyl_observer_vs_existing_n0():
    """Cylinder n=0: observer-centered 3D form should match existing surface-centered form."""
    R = 2.0
    radii = np.array([R])
    sig_t = np.array([1.0])
    r_nodes, _, _ = composite_gl_r(radii, 2, 5, dps=25)

    # Existing surface-centered
    G_existing = compute_G_bc_mode(CYLINDER_1D, r_nodes, radii, sig_t, 0,
                                   n_surf_quad=32, dps=25)
    # Observer-centered 3D
    G_obs = compute_G_bc_observer(CYLINDER_1D, r_nodes, radii, sig_t, 0,
                                  n_angular=32, dps=25)

    print(f"\nCylinder n=0, R={R}:")
    for i in range(len(r_nodes)):
        print(f"  r={r_nodes[i]:.4f}  surface-centered={G_existing[i]:.6e}  "
              f"observer-centered={G_obs[i]:.6e}  ratio={G_obs[i]/G_existing[i]:.6f}")


def test_cyl_observer_vs_existing_n1():
    """Cylinder n=1: observer-centered 3D form (H1 fix) should DIFFER from the 2D-projected form."""
    R = 2.0
    radii = np.array([R])
    sig_t = np.array([1.0])
    r_nodes, _, _ = composite_gl_r(radii, 2, 5, dps=25)

    for n in [1, 2, 3]:
        G_existing = compute_G_bc_mode(CYLINDER_1D, r_nodes, radii, sig_t, n,
                                       n_surf_quad=32, dps=25)
        G_obs = compute_G_bc_observer(CYLINDER_1D, r_nodes, radii, sig_t, n,
                                      n_angular=32, dps=25)

        print(f"\nCylinder n={n}, R={R}:")
        for i in [0, len(r_nodes) // 2, -1]:
            print(f"  r={r_nodes[i]:.4f}  surface(2D)={G_existing[i]:+.4e}  "
                  f"observer(3D)={G_obs[i]:+.4e}  ratio={G_obs[i]/G_existing[i] if G_existing[i] != 0 else float('nan'):+.4f}")


if __name__ == "__main__":
    test_sphere_G_bc_mode_equals_P_esc_mode()
    test_cyl_observer_vs_existing_n0()
    test_cyl_observer_vs_existing_n1()
