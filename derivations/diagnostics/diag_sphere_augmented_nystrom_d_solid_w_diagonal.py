"""Diagnostic D — augmented-Nyström for SOLID sphere with proper W(µ).

Issue #132 augmented-Nyström investigation.

Key derivation
--------------
For a solid homogeneous sphere of radius R with Σ_t, a neutron that
LEAVES the surface at angle µ to the outward normal travels along a
chord of length 2 R µ before hitting the surface AGAIN at the
antipodal-symmetric point with the SAME µ (cosine to the inward normal
at the new point). The angular-resolved surface-to-surface
transmission probability is therefore DIAGONAL in µ:

    W(µ_out → µ_in) = exp(-Σ_t · 2 R µ_in) · δ(µ_in − µ_out)

In the discrete µ-basis on M Nyström nodes:
    W[m, m] = exp(-2 Σ_t R µ_m)
    W[m, n] = 0 for m ≠ n

White BC J^-(µ) = J^+(µ) gives the angular-resolved reflection
    R_eff = (I_M − W)^{-1}     (DIAGONAL, R_eff[m,m] = 1/(1−W_mm))

The closure operator K_bc = G · R_eff · P at M ≥ 2 then captures the
ANGLE-DEPENDENT chord weighting that the rank-1 Mark closure
(R = e_0 e_0^T, single mode) cannot.

Hypothesis check
----------------
- M = 1 with Mark mode: should reproduce existing rank-1 Mark.
- M = 1 with Hébert (1−P_ss)^-1: should reproduce shipped Hébert
  closure (since P_ss is the µ-weighted scalar version of W_mm).
- M = 2, 4, 8, 16: should converge to cp k_inf if the angular
  resolution genuinely captures the surface partial-current
  distribution that the rank-1 cannot.

Critical caveat from Diagnostic A/B
-----------------------------------
The per-angle kernel K(r_i, µ) has an inverse-square-root endpoint
singularity at µ = µ_min(r_i) = sqrt(1−(r_i/R)^2). Naive
Gauss-Legendre on µ ∈ [0, 1] does not capture this — even at M=512
the kernel evaluation has 1-10% relative error against the
analytical G_bc(r_i) Lambertian collapse (Diagnostic B).

Hence even if W is correct (this diagnostic), the per-angle kernel
representation is **structurally low-fidelity** at any practical M,
which CAPS the augmented-Nyström convergence rate from above. The
diagnostic still reports k_eff(M) so we can see what the cap is.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from orpheus.derivations._xs_library import LAYOUTS, get_xs
from orpheus.derivations import cp_sphere
from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, build_volume_kernel, compute_P_esc, compute_G_bc,
    composite_gl_r,
)
from orpheus.derivations._eigenvalue import kinf_from_cp
from orpheus.derivations.cp_sphere import _sphere_cp_matrix

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_sphere_augmented_nystrom_a_kernel_derivation import (  # noqa: E402
    K_per_angle_via_observer,
)


def _per_angle_G_block(r_nodes, R, sig_t_n_per_node, mu_pts, mu_wts):
    """G[i, m] = w_m · K(r_i, µ_m), the per-angle response density.

    Note: r-DEPENDENT support — K = 0 for µ < µ_min(r_i) — handled
    inside K_per_angle_via_observer.

    For a single-region SOLID sphere only.
    """
    N = len(r_nodes)
    M = len(mu_pts)
    G = np.zeros((N, M))
    for i, r_i in enumerate(r_nodes):
        if r_i == 0.0:
            # Centre: µ_exit = 1 always. K is a delta at µ=1; treat as
            # G[i, m] = (i = M-1 contribution).
            # Use a smoothed limit: integrate the analytical G_bc and
            # assign all weight to µ_m closest to 1.
            G_centre = compute_G_bc(SPHERE_1D, np.array([0.0]),
                                    np.array([R]), np.array([sig_t_n_per_node[i]]),
                                    n_surf_quad=64, dps=15)[0]
            # Distribute by scaling so that ψ^- = 1/π collapses to G_centre
            # M-mode-uniform isn't right; just put all weight on the µ=1
            # node — the augmented eigensolve will tolerate a single
            # discrepancy at r=0.
            G[i, -1] = G_centre / (mu_wts[-1] / np.pi)
            continue
        for m, (mu, w) in enumerate(zip(mu_pts, mu_wts)):
            G[i, m] = w * K_per_angle_via_observer(r_i, mu, R, sig_t_n_per_node[i])
    return G


def _per_angle_P_block(r_nodes, r_wts, R, sig_t_n_per_node, mu_pts, mu_wts):
    """P[m, j] = J^+_m response per unit interior source at r_j.

    By reciprocity G_bc(r, µ) ∝ µ · P_esc(r, µ) but the precise factor
    depends on normalisation. We derive directly:
        P_esc(r_j, µ) = ∫_solid-angle response of µ-tagged outgoing
                        partial current at the surface to a unit
                        isotropic source at r_j.

    For the sphere with isotropic emission, the surface µ_exit
    distribution from a source at r_j is

        P_esc(r_j, µ) dµ = (1/4π) · K(r_j, µ) · exp(-τ) · dµ

    Since K already includes the exp(-τ) and the 2π·sin θ·|dθ/dµ|
    Jacobian (from the change of variables in the observer integral),
    and noting that the source emits isotropically into 4π steradians
    while we want the partial current per unit surface area at the
    exit point, the relation is:

        P_esc(r_j, µ) = K(r_j, µ) / (4π · R²)

    but normalisation conventions vary. For diagnostic purposes use
    reciprocity: P[m, j] = (volume weight at j) · K(r_j, µ_m) / (4π R²)
    so that ∫ P · 1 over j gives the existing P_esc.
    """
    N = len(r_nodes)
    M = len(mu_pts)
    P = np.zeros((M, N))
    rv = np.array([SPHERE_1D.radial_volume_weight(rj) for rj in r_nodes])
    for m, mu in enumerate(mu_pts):
        for j, r_j in enumerate(r_nodes):
            if r_j == 0.0:
                continue
            K_val = K_per_angle_via_observer(r_j, mu, R, sig_t_n_per_node[j])
            # Reciprocity-consistent normalisation: P · 1 over m·w_m
            # should give P_esc(r_j) for the Lambertian collapse.
            # We absorb the 1/π factor into the normalisation here.
            P[m, j] = rv[j] * r_wts[j] * K_val / (4.0 * np.pi * R * R)
    return P


def solve_augmented_solid_sphere_diag_W(
    sig_t: float, sig_s: float, nu_sig_f: float, R: float,
    M: int,
    *, n_panels=4, p_order=5, n_angular=64, n_rho=64,
    n_surf_quad=64, dps=25,
):
    """Solve the (N+M) augmented eigenvalue problem with diagonal W."""
    radii = np.array([R])
    sig_t_arr = np.array([sig_t])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, n_panels_per_region=n_panels, p_order=p_order, dps=dps,
    )
    K_vol = build_volume_kernel(
        SPHERE_1D, r_nodes, panels, radii, sig_t_arr,
        n_angular=n_angular, n_rho=n_rho, dps=dps,
    )
    sig_t_n = sig_t * np.ones(len(r_nodes))

    # M-node Gauss-Legendre on µ ∈ (0, 1]
    pts, wts = np.polynomial.legendre.leggauss(M)
    mu_pts = 0.5 * (pts + 1.0)
    mu_wts = 0.5 * wts

    # Diagonal transmission W[m,m] = exp(-2 Σ_t R µ_m)
    W = np.diag(np.exp(-2.0 * sig_t * R * mu_pts))
    R_eff = np.linalg.inv(np.eye(M) - W)

    G = _per_angle_G_block(r_nodes, R, sig_t_n, mu_pts, mu_wts)
    P = _per_angle_P_block(r_nodes, r_wts, R, sig_t_n, mu_pts, mu_wts)

    K_bc = G @ R_eff @ P
    K_total = K_vol + K_bc

    # 1-G eigenvalue: (Σ_t I − K_total · σ_s) φ = (1/k) K_total · νΣ_f φ
    A = np.diag(sig_t_n) - K_total * sig_s
    B_op = K_total * nu_sig_f
    eigvals = la.eigvals(np.linalg.solve(A, B_op))
    return float(np.max(eigvals.real))


def main():
    print("=" * 78)
    print("Augmented-Nyström solid-sphere k_eff convergence vs M")
    print("=" * 78)
    print()

    # Region A 1G as the canonical solid-sphere benchmark
    xs = get_xs("A", "1g")
    sig_t = float(xs["sig_t"][0])
    sig_s = float(xs["sig_s"][0, 0])
    nu_sig_f = float(xs["nu"][0] * xs["sig_f"][0])
    R = 1.0
    cp_kinf = nu_sig_f / (sig_t - sig_s)  # for 1-G/1-R, k_inf = νΣ_f/Σ_a
    print(f"  Region A 1G:  σ_t={sig_t:.3f} σ_s={sig_s:.3f} "
          f"νΣ_f={nu_sig_f:.3f}  cp k_inf = {cp_kinf:.6f}")
    print()
    print(f"  {'M':>4} {'k_aug(M)':>14} {'rel_err vs cp':>15}")
    for M in [1, 2, 4, 8, 16, 32, 64]:
        k_M = solve_augmented_solid_sphere_diag_W(
            sig_t, sig_s, nu_sig_f, R, M,
            n_panels=2, p_order=3, n_angular=24, n_rho=24,
            n_surf_quad=24, dps=15,
        )
        rel = (k_M - cp_kinf) / cp_kinf * 100
        print(f"  {M:>4d} {k_M:>14.10f} {rel:>+14.3f}%")

    print()
    print("  Now try the chi-dependent 2G/2R (the limitation case):")
    print()


if __name__ == "__main__":
    main()
