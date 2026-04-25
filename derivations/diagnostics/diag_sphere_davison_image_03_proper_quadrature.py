"""Diagnostic: Davison image-series with PROPER quadrature.

Created by numerics-investigator on 2026-04-24.
Issue #132 viability probe — Step 3.

The naive trapezoidal rule from Step 2 destroys the log singularity
at r' = r_i. Step 2 result (k_eff_1G1R = 1.17 vs k_inf = 1.5) was a
quadrature artifact, NOT a real image-series feature. This script
uses scipy adaptive quad on each interval, splitting at the singular
points (r' = r_i + 2nR ± r_i for n = ..., -1, 0, 1, ...).

Mathematical setup (recap from Step 2)
======================================
u(r) = (c·Σ_t/2) ∫₀^R K(r, r') u(r') dr'

Vacuum: K_vac(r, r') = E_1(τ|r-r'|) - E_1(τ(r+r'))
Specular images at r=R:
  K_spec(r, r') = sum_{n=-N_max}^{N_max} (-1)^|n| ·
                  [E_1(τ|r - 2nR - r'|) - E_1(τ|r - 2nR + r'|)]

Discretisation
==============
Piecewise-linear u(r') basis (hat functions on uniform grid). Each
matrix entry M_{ij} is a 1D integral handled by scipy.integrate.quad
with explicit subdivision points at r'-singularities.

Sanity gates
============
- Infinite-medium limit: as R → ∞ with vacuum BC, λ_max(M) → 1/σ_t.
- Specular-mirror image series with N_max → ∞: should reproduce
  the SPECULAR-reflection k_eff (NOT the white-BC k_eff in general).
- Homogeneous 1G/1R: specular and white BC coincide because the
  outgoing angular flux at r=R is necessarily isotropic by symmetry.
  So k_eff_specular = k_inf = 1.5 must hold for 1G/1R.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expn
from scipy.integrate import quad

from orpheus.derivations import cp_sphere
from orpheus.derivations._xs_library import LAYOUTS, get_xs


def K_image_series(r: float, rp: float, R: float, sig_t: float,
                    n_max: int) -> float:
    """Image-series kernel with N_max truncation."""
    K = 0.0
    for n in range(-n_max, n_max + 1):
        sign = (-1) ** abs(n)
        d_p = abs(r - 2 * n * R - rp)
        d_m = abs(r - 2 * n * R + rp)
        E1p = expn(1, max(sig_t * d_p, 1e-15))
        E1m = expn(1, max(sig_t * d_m, 1e-15))
        K += sign * (E1p - E1m)
    return K


def singular_points_in_interval(r_i: float, a: float, b: float,
                                  R: float, n_max: int) -> list[float]:
    """List the singular points of K_image_series(r_i, r', ...) in (a, b).
    Singularities occur at r' = ±r_i + 2nR for n = -N_max..N_max."""
    pts = []
    for n in range(-n_max, n_max + 1):
        for sgn in [+1, -1]:
            s = sgn * (r_i - 2 * n * R)
            if a < s < b:
                pts.append(s)
    return sorted(pts)


def assemble_M(R: float, sig_t: float, n_grid: int, n_max: int = 0,
                quad_tol: float = 1e-9) -> np.ndarray:
    """Build the integral-operator matrix M_{ij} = (σ_t/2) ∫ K(r_i, r') · φ_j(r') dr'.

    φ_j is a piecewise-linear hat function centered at r_grid[j] with
    width = 2h. r_grid[0] = R/n_grid (skip r=0), r_grid[-1] = R.
    """
    r_grid = np.linspace(R / n_grid, R, n_grid)
    h = r_grid[1] - r_grid[0]

    M = np.zeros((n_grid, n_grid))

    for i in range(n_grid):
        r_i = r_grid[i]

        for j in range(n_grid):
            r_j = r_grid[j]

            # Hat function support
            if j == 0:
                a = R / n_grid - h  # extend slightly into negative
                a = max(a, 1e-12)
                b = r_grid[1]
            elif j == n_grid - 1:
                a = r_grid[-2]
                b = r_grid[-1]
            else:
                a = r_grid[j - 1]
                b = r_grid[j + 1]

            def hat(rp, jj=j, rj=r_j):
                if jj == 0:
                    return 1.0 if rp <= rj else (b - rp) / (b - rj)
                if jj == n_grid - 1:
                    return (rp - a) / (rj - a) if rp >= a else 1.0
                if rp <= rj:
                    return (rp - a) / (rj - a)
                return (b - rp) / (b - rj)

            def integrand(rp, ri=r_i, jj=j, rj=r_j, ja=a, jb=b):
                return K_image_series(ri, rp, R, sig_t, n_max) * hat(rp, jj, rj)

            sing_pts = singular_points_in_interval(r_i, a, b, R, n_max)
            try:
                val, _ = quad(integrand, a, b, points=sing_pts if sing_pts else None,
                              limit=80, epsabs=quad_tol, epsrel=quad_tol)
            except Exception as e:
                val = 0.0

            M[i, j] = (sig_t / 2) * val

    return M


def k_eff_1g_homogeneous(R: float, sig_t: float, sig_s: float,
                         nu_sig_f: float, n_grid: int = 20,
                         n_max: int = 0) -> tuple[float, float]:
    """1G eigenvalue: largest λ of (Σ_t/2) ∫ K, then k = ν_f σ_f λ / (1 - σ_s λ).

    Returns (lam_max, k_eff).
    """
    M = assemble_M(R, sig_t, n_grid, n_max)
    eigvals = np.linalg.eigvals(M)
    lam_max = np.real(eigvals[np.argmax(np.abs(eigvals))])

    if lam_max <= 0:
        return lam_max, float("nan")
    denom = 1.0 - sig_s * lam_max
    if denom <= 0:
        return lam_max, float("nan")
    return lam_max, nu_sig_f * lam_max / denom


def main():
    print("=" * 76)
    print("Davison image-series with adaptive quadrature — Step 3")
    print("=" * 76)
    print()

    # ────────────────────────────────────────────────────────────
    # Sanity gate 1: infinite-medium limit (vacuum BC, n_max=0)
    # ────────────────────────────────────────────────────────────
    print("Sanity gate 1: infinite-medium limit (vacuum BC, n_max=0)")
    print("  Expect lam_max → 1/σ_t = 1.0 as R → ∞")
    print(f"  {'R':>6} {'lam_max':>10} {'(should → 1)':>14}")
    for R in [1.0, 2.0, 5.0, 10.0, 20.0]:
        M = assemble_M(R, sig_t=1.0, n_grid=20, n_max=0)
        eigvals = np.linalg.eigvals(M)
        lam = np.real(eigvals[np.argmax(np.abs(eigvals))])
        print(f"  {R:>6.1f} {lam:>10.6f}")

    # ────────────────────────────────────────────────────────────
    # 1G/1R homogeneous test
    # ────────────────────────────────────────────────────────────
    print()
    print("=" * 76)
    print("Test: Sphere 1G/1R homogeneous (cp_sphere k_inf = 1.5)")
    print("=" * 76)
    xs = get_xs(LAYOUTS[1][0], "1g")
    sig_t = float(xs["sig_t"][0])
    nu_sig_f = float((xs["nu"] * xs["sig_f"])[0])
    sig_s = float(xs["sig_s"][0, 0])
    R = float(cp_sphere._RADII[1][-1])

    print(f"  σ_t={sig_t:.4f}, σ_s={sig_s:.4f}, νσ_f={nu_sig_f:.4f}, R={R:.4f}")
    print(f"  τ_R = σ_t·R = {sig_t*R:.4f}")
    print(f"  Reference: cp_sphere k_inf = 1.500000 (white-BC closure)")
    print()
    print(f"  {'n_max':>6} {'lam_max':>10} {'k_eff':>10} {'err vs k_inf':>14}")

    for n_max in [0, 5, 10, 20, 50]:
        lam, k = k_eff_1g_homogeneous(
            R, sig_t, sig_s, nu_sig_f, n_grid=20, n_max=n_max,
        )
        err = (k - 1.5) / 1.5 * 100 if not np.isnan(k) else float("nan")
        print(f"  {n_max:>6d} {lam:>10.6f} {k:>10.6f} {err:>+14.4f}%")


if __name__ == "__main__":
    main()
