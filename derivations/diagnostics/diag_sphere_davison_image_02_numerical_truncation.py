"""Diagnostic: numerical k_eff from Davison image-series kernel.

Created by numerics-investigator on 2026-04-24.
Issue #132 viability probe — Step 2 of the cascade.

Goal
====
Build the image-series kernel from Step 1, apply it to the
spherical Peierls equation in ``u(r) = r·φ(r)`` form, and report
k_eff at truncation orders ``[5, 10, 20, 50]`` images for:

- 1G/1R homogeneous (k_inf = 1.5, both vacuum-image and
  specular-mirror should give 1.5 — the "specular" case for a
  homogeneous sphere with one material is identical to white BC
  trivially since the surface flux IS isotropic).
- 1G/2R fuel-mod (k_inf = 0.6480 from cp_sphere reference).
- 2G/2R fuel-mod (k_inf = 0.4140 from cp_sphere reference).

Mathematical setup
==================
Davison u(r) = r·φ(r) form:

.. math::

   u(r) = \\frac{c Σ_t}{2} \\int_0^R K(r, r') u(r') dr'

Vacuum-BC kernel (Davison single-image at -r'):

.. math::

   K_{\\rm vac}(r, r') = E_1(τ|r-r'|) - E_1(τ(r+r'))

Specular-mirror at r=R image series (from Step 1):

.. math::

   K_{\\rm spec}(r, r') = \\sum_{n=-N_{\\max}}^{N_{\\max}} (-1)^{|n|}
                          [E_1(τ|r - 2nR - r'|) - E_1(τ|r - 2nR + r'|)]

For the eigenvalue problem ``k = c`` (one-group homogeneous), we
seek the largest ``c`` such that the integral operator has eigenvalue
``1``:

.. math::

   λ_{\\max}(K_{\\rm op}) = 2 / (c Σ_t)

so ``k_eff = c = ν_f Σ_f / Σ_t = 2 / (Σ_t · λ_{\\max})``.

For multi-group / multi-region, we discretise u(r) on a fine mesh,
build the matrix [K]_{ij} = K(r_i, r_j) · w_j (trapezoidal/GL
weights), solve the generalized eigenvalue problem with the
scattering + fission operators per the standard Peierls assembly.

For SIMPLICITY in this viability probe, we test 1G/1R first
(homogeneous, single-material sphere — k_inf = c trivially) to
verify the image series converges to the correct value.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expn  # E_n exponential integrals
from scipy.linalg import eig

from orpheus.derivations import cp_sphere
from orpheus.derivations._xs_library import LAYOUTS, get_xs


# ─────────────────────────────────────────────────────────────────────
# Image-series kernel
# ─────────────────────────────────────────────────────────────────────

def K_specular_homogeneous(r: float, rp: float, R: float, sig_t: float,
                            n_max: int) -> float:
    """Specular-mirror image-series kernel for homogeneous sphere.

    K(r, r') = sum_{n=-N_max}^{N_max} (-1)^|n| ·
               [E_1(τ|r - 2nR - r'|) - E_1(τ|r - 2nR + r'|)]

    where τ = sig_t. The n=0 term reproduces Davison vacuum kernel.
    """
    K = 0.0
    for n in range(-n_max, n_max + 1):
        sign = (-1) ** abs(n)
        # +r' image at 2nR + r' (sign +1 within the square bracket)
        d_plus = abs(r - 2 * n * R - rp)
        # -r' image at 2nR - r' (sign -1 within the square bracket)
        d_minus = abs(r - 2 * n * R + rp)

        tau_p = sig_t * d_plus
        tau_m = sig_t * d_minus

        # E_1 diverges as -log(x) at x=0; use small-x guard
        E1_p = expn(1, max(tau_p, 1e-15))
        E1_m = expn(1, max(tau_m, 1e-15))

        K += sign * (E1_p - E1_m)

    return K


def K_vacuum_homogeneous(r: float, rp: float, R: float, sig_t: float) -> float:
    """Davison vacuum kernel (n=0 only)."""
    return K_specular_homogeneous(r, rp, R, sig_t, n_max=0)


# ─────────────────────────────────────────────────────────────────────
# 1G eigenvalue problem
# ─────────────────────────────────────────────────────────────────────

def solve_1g_eigenvalue(R: float, sig_t: float, c: float | None = None,
                         n_max: int = 0, n_grid: int = 200) -> float:
    """Solve the 1G Peierls eigenvalue problem in u(r) form.

    u(r) = (c Σ_t / 2) ∫₀^R K(r, r') u(r') dr'

    Discretise on a uniform grid r_i ∈ (0, R] with trapezoid weights,
    build matrix M_{ij} = (Σ_t / 2) · K(r_i, r_j) · w_j, solve
    largest eigenvalue λ_max. Then k_eff = c such that c · λ_max = 1,
    i.e., k_eff = 1/λ_max (since 1G case has k_eff = c).

    Returns k_eff.
    """
    # Uniform grid (avoid r=0 to skip the u(0)=0 singularity)
    r_grid = np.linspace(R / n_grid, R, n_grid)
    h = r_grid[1] - r_grid[0]

    # Trapezoid weights (uniform interior, half at endpoints — but we
    # exclude r=0 and include r=R, so use simple uniform with edge
    # adjustment)
    w = np.full(n_grid, h)
    w[0] = h / 2  # r near 0 (but not 0)
    w[-1] = h / 2  # r=R

    # Build matrix M_{ij} = (sig_t / 2) · K(r_i, r_j) · w_j
    M = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            M[i, j] = (sig_t / 2) * K_specular_homogeneous(
                r_grid[i], r_grid[j], R, sig_t, n_max
            ) * w[j]

    # Largest eigenvalue
    eigvals = np.linalg.eigvals(M)
    # Take real part of largest-modulus eigenvalue
    idx = np.argmax(np.abs(eigvals))
    lam_max = np.real(eigvals[idx])

    if lam_max <= 0:
        return float("nan")

    # For the 1G eigenvalue problem u = (c Σ_t / 2) ∫ K u dr', we have
    # M u = λ_max u where M = (c Σ_t / 2) K_w. The original equation
    # is u = K_op u with K_op having spectral radius 1 at criticality.
    # Our M absorbs c Σ_t / 2; solving for largest eigenvalue gives
    # the value of M's spectral radius. To make K_op have radius 1,
    # we solve for c such that c · (M with c=1 baked into σ_t? — let
    # me re-derive carefully).
    #
    # Setup: u(r) = (c Σ_t / 2) ∫ K(r,r') u(r') dr' = c · A · u
    # where A = (Σ_t / 2) · K_w. Largest eigenvalue of A is λ_A.
    # Criticality: c · λ_A = 1, so k_eff = c = 1 / λ_A.
    return 1.0 / lam_max


# ─────────────────────────────────────────────────────────────────────
# Run the viability probe
# ─────────────────────────────────────────────────────────────────────

def run_homogeneous_1g_1r() -> dict:
    """Sphere 1G/1R homogeneous — must give k_eff = c = ν_f Σ_f / Σ_t.

    Material 'mat_2' is the fuel for cp_sphere 1g/1r layout.
    """
    xs = get_xs(LAYOUTS[1][0], "1g")
    sig_t = float(xs["sig_t"][0])
    nu_sig_f = float((xs["nu"] * xs["sig_f"])[0])
    sig_s = float(xs["sig_s"][0, 0])  # sig_s[g_in, g_out]
    sig_a = sig_t - sig_s
    c_inf = nu_sig_f / sig_a  # k_inf for fully self-scattering
    R = float(cp_sphere._RADII[1][-1])

    print(f"  σ_t = {sig_t:.6f}, σ_s = {sig_s:.6f}, σ_a = {sig_a:.6f}, "
          f"νσ_f = {nu_sig_f:.6f}")
    print(f"  R = {R:.4f}, τ_R = σ_t·R = {sig_t*R:.4f}")
    print(f"  k_inf (analytic) = νσ_f / σ_a = {c_inf:.6f}")
    print()

    # The 1G Peierls equation with self-scattering is:
    # φ(r) = (Σ_s + νΣ_f) / Σ_t · (Σ_t / 2) · ∫ K(r,r') φ(r') dV'
    # i.e., effective "c" = (Σ_s + νΣ_f) / Σ_t.
    # k_eff scales linearly with νΣ_f, so:
    # k_eff = νΣ_f / [Σ_t · λ_max - Σ_s] (equation re-arrangement)
    # or equivalently solve for largest c_eff such that c_eff · λ_max = 1
    # then k_eff = (c_eff - Σ_s/Σ_t) · Σ_t / νΣ_f · νΣ_f = ?
    # Let me solve the FULL 1G eigenvalue: (νΣ_f/k + Σ_s) acts as source.
    # M · φ = (1/k_eff) · F · φ, etc. — too much to do here.
    #
    # SIMPLIFICATION for this viability probe: solve for largest c_eff
    # = total emission per collision. Then "k_eff" of the homogeneous
    # 1G/1R sphere corresponds to c_eff = (Σ_s + νΣ_f/k) / Σ_t = 1
    # at criticality. With c_eff_geom = 1 / λ_max_image_kernel:
    # 1 = (Σ_s + νΣ_f/k) · λ_max → νΣ_f/k = 1/λ_max - Σ_s
    # → k = νΣ_f · λ_max / (1 - Σ_s · λ_max)
    #
    # Check: in the infinite medium limit (R → ∞), λ_max → 1/Σ_t (since
    # the integral operator becomes the identity scaled by 1/Σ_t).
    # Then k = νΣ_f · (1/Σ_t) / (1 - Σ_s/Σ_t) = νΣ_f / Σ_a = k_inf ✓

    results = {}
    for n_max in [0, 5, 10, 20, 50]:
        # Solve for λ_max of (Σ_t / 2) K_w
        n_grid = 60  # modest for n_max=50 not to blow up
        r_grid = np.linspace(R / n_grid, R, n_grid)
        h = r_grid[1] - r_grid[0]
        w = np.full(n_grid, h)
        w[0] = h / 2
        w[-1] = h / 2

        M = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            for j in range(n_grid):
                M[i, j] = (sig_t / 2) * K_specular_homogeneous(
                    r_grid[i], r_grid[j], R, sig_t, n_max
                ) * w[j]

        eigvals = np.linalg.eigvals(M)
        idx = np.argmax(np.abs(eigvals))
        lam_max = np.real(eigvals[idx])

        if lam_max <= 0:
            print(f"  n_max={n_max:3d}: lam_max = {lam_max:.6f}  (NEGATIVE — "
                  f"divergent kernel)")
            results[n_max] = float("nan")
            continue

        # k = νΣ_f · λ_max / (1 - Σ_s · λ_max)
        denom = 1.0 - sig_s * lam_max
        if denom <= 0:
            print(f"  n_max={n_max:3d}: lam_max = {lam_max:.6f}  "
                  f"(supercritical — denom = {denom:.4f})")
            results[n_max] = float("nan")
            continue

        k_eff = nu_sig_f * lam_max / denom
        err = (k_eff - c_inf) / c_inf * 100
        results[n_max] = k_eff
        print(f"  n_max={n_max:3d}: lam_max = {lam_max:.6f}  "
              f"k_eff = {k_eff:.6f}  err vs k_inf = {err:+.3f}%")

    return results


def run_homogeneous_1g_1r_vacuum_only():
    """Sphere 1G/1R with VACUUM kernel only (n_max=0). This is the
    Davison classical result — the smallest sub-critical sphere."""
    xs = get_xs(LAYOUTS[1][0], "1g")
    sig_t = float(xs["sig_t"][0])
    nu_sig_f = float((xs["nu"] * xs["sig_f"])[0])
    sig_s = float(xs["sig_s"][0, 0])
    R = float(cp_sphere._RADII[1][-1])

    print()
    print("  --- Vacuum BC (n_max=0): the bare sphere ---")
    n_grid = 80
    r_grid = np.linspace(R / n_grid, R, n_grid)
    h = r_grid[1] - r_grid[0]
    w = np.full(n_grid, h)
    w[0] = h / 2
    w[-1] = h / 2

    M = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            M[i, j] = (sig_t / 2) * K_vacuum_homogeneous(
                r_grid[i], r_grid[j], R, sig_t
            ) * w[j]

    eigvals = np.linalg.eigvals(M)
    idx = np.argmax(np.abs(eigvals))
    lam_max = np.real(eigvals[idx])

    denom = 1.0 - sig_s * lam_max
    k_eff_vac = nu_sig_f * lam_max / denom
    print(f"  vacuum k_eff = {k_eff_vac:.6f}  (should be < k_inf = {nu_sig_f/(sig_t-sig_s):.6f}; "
          f"finite leakage)")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 76)
    print("Davison image-series viability probe — Step 2 (numerical)")
    print("=" * 76)
    print()
    print("Test 1: Sphere 1G/1R homogeneous")
    print("-" * 60)
    run_homogeneous_1g_1r()
    run_homogeneous_1g_1r_vacuum_only()
