"""Diagnostic: calibrate rank-N normalization via numerical reciprocity.

Goal: find the correct coefficient α_n in the rank-N closure

    K_bc[i,j] = α_n · Σ_t · g_n(r_i) · P_esc^(n)(r_j) · r_j² w_j / R²

where α_0 = 1 (the code's rank-1 value).  We test multiple α_n for n ≥ 1
against a HIGH-FIDELITY brute-force computation: explicitly compute the
mode-n Legendre moment of the outgoing flux due to a unit point source,
and invert the Marshak condition.

For a SPHERE, we can do this directly:
1. Place a source q at node r_j.
2. Compute the full outgoing ψ^+(r_s, μ) over the surface.
3. Extract its n-th half-range moment ⟨m^+_n⟩.
4. Set m^-_n = ⟨m^+_n⟩ and compute φ_bc at any observer r_i.
5. Compare to the (u_n, v_n) assembly.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/workspaces/ORPHEUS")
import numpy as np

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, composite_gl_r, gl_float,
    compute_G_bc_mode, compute_P_esc_mode,
)
from orpheus.derivations._kernels import _shifted_legendre_eval


def compute_m_plus_n_brute(R, sig_t, r_source, n_mode,
                           n_theta_src=64, n_theta_surf=128, dps=25):
    """Brute-force compute the surface-averaged n-th half-range Legendre moment
    of the outgoing flux due to a unit point (shell) source at r_source.

    For a sphere with 1D radial symmetry, the outgoing flux at surface point r_s
    in direction (μ, φ) depends only on μ (polar angle from outward normal) and
    the distance from r_source to r_s along the ray.

    By reciprocity, the contribution to ⟨m^+_n⟩ from a ISOTROPIC point source at
    r' can be computed by ray-tracing from the source outward:

    For an isotropic source at r' emitting Q_total = 1 neutron per unit time,
    the angular distribution per unit solid angle is 1/(4π).  Neutron emitted
    in direction Ω hits surface at r_s with cosine μ_s = (ρ_max + r'·cos θ)/R,
    carrying attenuated intensity exp(-τ).

    ⟨m^+_n⟩ = (1/A) ∫_A m^+_n(r_s) dA_s
           = (1/A) ∫_A dA_s ∫_0^1 dμ P̃_n(μ) ψ^+(r_s, μ, φ)

    Express in terms of the ISOTROPIC source at r': for each emitted direction
    Ω at r', the neutron arrives at the surface at r_s = r' + ρ·Ω with outgoing
    cosine μ_s.  The angular flux at r_s in direction Ω is (contribution from
    the point source in solid-angle space):

        dψ^+/d(source) = (1/(4π)) · exp(-τ) · δ-in-Ω-at-surface
    """
    # Observer-centered angular quadrature at r_source.
    # dΩ = sin θ dθ dφ.  With axial symmetry (source is a shell), fold φ → 1 since everything is symmetric.
    # Actually, for r_source on the axis (WLOG) and axisymmetric, we just integrate θ.
    theta_pts, theta_wts = gl_float(n_theta_src, 0.0, np.pi, dps)
    cos_thetas = np.cos(theta_pts)
    sin_thetas = np.sin(theta_pts)

    # For each θ, the ray from r' = (r_source, 0, 0) in direction
    # (cos θ, sin θ) (2D; or really (cos θ, sin θ, 0) with θ from z-axis).
    # Exit point at radius R reached at distance ρ_max.
    # μ_s at exit = (ρ_max + r_source · cos θ)/R (this is the standard formula).
    total_mn = 0.0
    for k in range(n_theta_src):
        ct = cos_thetas[k]
        st = sin_thetas[k]
        rho_max_val = -r_source * ct + np.sqrt(max(
            r_source ** 2 * ct * ct + R * R - r_source ** 2, 0.0
        ))
        tau = sig_t * rho_max_val
        mu_s = (rho_max_val + r_source * ct) / R
        p_tilde = float(_shifted_legendre_eval(n_mode, np.array([mu_s]))[0])
        # The neutron emitted in direction θ (with azimuthal integration giving
        # 2π factor) contributes exp(-τ) · p_tilde to the n-th moment at its
        # exit point.
        total_mn += theta_wts[k] * st * p_tilde * np.exp(-tau)
    # Azimuthal fold: 2π factor.
    total_mn *= 2.0 * np.pi
    # Source emits 1/(4π) per solid angle (isotropic normalized to 1 neutron/s).
    total_mn /= (4.0 * np.pi)
    # Each neutron exits through SOME surface point; its contribution to
    # the surface integral of m^+_n(r_s) is exp(-τ) · P̃_n(μ_s) — this is what
    # total_mn integrates.  To get ⟨m^+_n⟩ (surface-averaged), divide by A = 4πR².
    # Actually, the integral ∫_A m^+_n(r_s) dA_s = total over all exit points.
    # total_mn above is precisely that (before divide by A) — per unit source emission
    # of 1 neutron per unit solid angle steradian? Wait.
    # Let me redo: source emits 1 neutron total (integrated over 4π).
    # Emission rate in direction Ω per solid angle: 1/(4π).
    # So the contribution to "flux arriving at surface in direction Ω at r_s, per unit time"
    # is (1/4π) · exp(-τ) per steradian at r_s (directed outward).
    # When integrated over 4π solid angle at r' and over r_s on the surface:
    # ∫_A dA_s ∫_0^1 dμ_s 2π dφ · P̃_n(μ_s) ψ^+(r_s, μ_s)
    # = ∫_{4π at r'} dΩ (1/(4π)) · P̃_n(μ_s(Ω)) · exp(-τ)
    # = (1/(4π)) · 2π · ∫_0^π sin θ P̃_n(μ_s) exp(-τ) dθ
    # = (1/2) · ∫_0^π sin θ P̃_n(μ_s) exp(-τ) dθ

    # So the UNSCALED integral is ∫_A m^+_n dA_s (summed over surface) =
    #   = (1/2) · ∫ sin θ P̃_n exp(-τ) dθ
    # for a unit (total) isotropic source at r'.
    # The surface-averaged:
    # ⟨m^+_n⟩ = (1/A) · [unscaled integral] = [unscaled] / (4πR²)
    #
    # Redo: forget the above numerical, just use this derived form:
    I_n = 0.0
    for k in range(n_theta_src):
        ct = cos_thetas[k]
        st = sin_thetas[k]
        rho_max_val = -r_source * ct + np.sqrt(max(
            r_source ** 2 * ct * ct + R * R - r_source ** 2, 0.0
        ))
        tau = sig_t * rho_max_val
        mu_s = (rho_max_val + r_source * ct) / R
        p_tilde = float(_shifted_legendre_eval(n_mode, np.array([mu_s]))[0])
        I_n += theta_wts[k] * st * p_tilde * np.exp(-tau)
    # Per unit point source at r' emitting 1 neutron/sec:
    #   ∫_A m^+_n(r_s) dA_s  =  (1/2) · I_n  (derived above)
    integral_mn_over_surface = 0.5 * I_n
    # ⟨m^+_n⟩ per UNIT source = integral_mn_over_surface / A
    A_surface = 4.0 * np.pi * R * R
    return integral_mn_over_surface / A_surface


def test_m_plus_0_sphere():
    """For a point source at r', ⟨m^+_0⟩ should equal J^+_total/(A·π) by
    Lambertian interpretation (assuming ψ^+ isotropic over emerging hemi = J^+/π)."""
    R = 2.0
    sig_t = 1.0
    n_mode = 0

    r_sources = [0.1, 0.5, 1.0, 1.5, 1.9]
    for r_src in r_sources:
        m0_brute = compute_m_plus_n_brute(R, sig_t, r_src, n_mode)

        # Compare with: P_esc^(0)(r') · (something)
        # compute_P_esc_mode for this single point requires an r_nodes array
        r_nodes = np.array([r_src])
        radii = np.array([R])
        sig_t_arr = np.array([sig_t])
        P_n = compute_P_esc_mode(SPHERE_1D, r_nodes, radii, sig_t_arr, n_mode,
                                 n_angular=64, dps=25)[0]

        # Hypothesis 1: ⟨m^+_n⟩ = (2/A) · P_esc^(n) · 4π  =  8π P_esc / A
        # (because ∫_V q P_esc dV from a point source with unit Q = P_esc(r')· ... )
        # Actually for a UNIT POINT SOURCE at r', ∫_V q P_esc dV = P_esc(r') · ??
        # q = δ(r-r'), so ∫_V δ(r-r') P_esc(r) dV = P_esc(r') · (1-dim integration over the angular part)
        # For SHELL-DISTRIBUTED unit source: q(r) = δ(r-r')/(4π r'²), then ∫ q · 4π r² P_esc dr = P_esc(r').
        # OK for a "shell source" with unit strength (one neutron per second spread uniformly over the shell):
        #   ∫_V q P_esc dV = P_esc(r')

        # So: ⟨m^+_0⟩_brute should be compared with P_esc(r') via our formula:
        # If ⟨m^+_0⟩ = (2/A) ∫ q P_esc dV = (2/A) · P_esc(r') = 2 P_esc / A
        hyp_coefficient = 2.0 / (4 * np.pi * R * R)
        expected = hyp_coefficient * P_n
        print(f"r'={r_src:.2f}: ⟨m^+_0⟩_brute = {m0_brute:.6e}  "
              f"P_esc={P_n:.4e}  "
              f"ratio ⟨m^+_0⟩/P_esc = {m0_brute/P_n:.6e}  "
              f"hyp=2/A={hyp_coefficient:.6e}  "
              f"match_ratio={m0_brute/expected:.6f}")


def test_m_plus_n_sphere():
    """Same brute-force calibration for n≥1."""
    R = 2.0
    sig_t = 1.0

    for n_mode in range(5):
        print(f"\n=== mode n={n_mode} ===")
        for r_src in [0.1, 0.5, 1.0, 1.5, 1.9]:
            m_brute = compute_m_plus_n_brute(R, sig_t, r_src, n_mode)
            r_nodes = np.array([r_src])
            radii = np.array([R])
            sig_t_arr = np.array([sig_t])
            P_n = compute_P_esc_mode(SPHERE_1D, r_nodes, radii, sig_t_arr, n_mode,
                                     n_angular=64, dps=25)[0]
            if P_n != 0:
                ratio = m_brute / P_n
                hyp = 2.0 / (4 * np.pi * R * R)  # 2/A
                print(f"  r'={r_src:.2f}: m^+_n/P_esc^(n) = {ratio:.6e}  "
                      f"(vs 2/A = {hyp:.6e}; m_brute/(2/A · P_n) = {m_brute/(hyp*P_n):.4f})")
            else:
                print(f"  r'={r_src:.2f}: P_n = 0 (skip)")


if __name__ == "__main__":
    test_m_plus_0_sphere()
    print()
    test_m_plus_n_sphere()
