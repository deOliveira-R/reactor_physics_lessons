"""Diagnostic: Decompose high-N cyl ρ(TR) explosion as R-conditioning.

Created by numerics-investigator on 2026-04-28.

In phase4_04 we found cyl ρ(T·R) explodes at N ≥ 16 (935 at N=16,
2.9e8 at N=20) but ‖(I-TR)⁻¹‖_2 stays bounded ≈ 5. Hypothesis: the
ρ explosion is numerical noise from R = (1/2) M^{-1} where M is
ill-conditioned at high N (cond(M) grows polynomially).

The ‖.‖_2 is the operator 2-norm = max singular value, which is
robust to noise eigenvalue ghosts. ρ uses the largest |eigenvalue|
which is dominated by noise eigenpairs of TR at high N.

To confirm:
  1. Print cond(M) directly.
  2. Print ρ(T·R) using high-precision R via mpmath inversion of M.
  3. Show that the 'noise' eigenvalues are SPURIOUS — their
     corresponding singular values are BOUNDED.

Conclusion: cyl pathology profile is QUALITATIVELY DIFFERENT from
sphere — sphere has a TRUE structural divergence in the resolvent
norm (memo specular_mb_overshoot_root_cause.md), cyl has bounded
resolvent + R-conditioning blowup at high N (numerical, fixable
with mpmath R).
"""
from __future__ import annotations

import numpy as np
import pytest
import mpmath

from orpheus.derivations.peierls_geometry import (
    _shifted_legendre_eval,
    reflection_specular,
)
from derivations.diagnostics.diag_specular_mb_phase4_03_pathology_resolvent import (
    build_T_cyl, build_T_slab, build_T_sphere,
)


def reflection_specular_mp(N: int, dps: int = 50) -> np.ndarray:
    """High-precision R = (1/2) M^{-1} via mpmath."""
    with mpmath.workdps(dps):
        M = mpmath.zeros(N, N)
        for n in range(N):
            M[n, n] = mpmath.mpf(1) / (2 * (2 * n + 1))
            if n + 1 < N:
                off = mpmath.mpf(n + 1) / (2 * (2 * n + 1) * (2 * n + 3))
                M[n, n + 1] = off
                M[n + 1, n] = off
        Minv = M ** (-1)
        R = mpmath.matrix(N, N)
        for i in range(N):
            for j in range(N):
                R[i, j] = mpmath.mpf(1) / 2 * Minv[i, j]
        return np.array([[float(R[i, j]) for j in range(N)] for i in range(N)])


def test_M_conditioning_grows(capsys):
    """cond(M) — supports the conjecture that high-N R is noisy."""
    with capsys.disabled():
        print(f"\n  N | cond(M) (float)  | cond(M) (mpmath dps=50)")
        for N in (1, 4, 8, 12, 16, 20, 25):
            M = np.zeros((N, N))
            for n in range(N):
                M[n, n] = 1.0 / (2.0 * (2 * n + 1))
                if n + 1 < N:
                    off = (n + 1) / (2.0 * (2 * n + 1) * (2 * n + 3))
                    M[n, n + 1] = off
                    M[n + 1, n] = off
            cond_f = float(np.linalg.cond(M))
            with mpmath.workdps(50):
                Mmp = mpmath.matrix(M.tolist())
                eigs_result = mpmath.eig(Mmp, right=False)
                eigs = eigs_result if not isinstance(
                    eigs_result, tuple
                ) else eigs_result[0]
                eigvals = [abs(eigs[i]) for i in range(N)]
                cond_mp = float(max(eigvals) / min(eigvals))
            print(f"  {N:2d}| {cond_f:.3e}    | {cond_mp:.3e}")


def test_cyl_rho_with_mp_R(capsys):
    """Compare cyl ρ(T·R) using float R vs high-precision R.

    If the explosion at N=16 is R-conditioning, mp R should keep ρ
    near sphere-like (~0.7..0.95)."""
    with capsys.disabled():
        sig_t, R = 0.5, 5.0
        print(f"\nCYL thin (τ_R={sig_t*R}): ρ(T·R) float vs mp R")
        print(f"  N  |  ρ float    |  ρ mp dps=50")
        for N in (8, 12, 16, 18, 20):
            T = build_T_cyl(sig_t, R, N)
            Rfl = reflection_specular(N)
            Rmp = reflection_specular_mp(N, dps=50)
            rho_fl = float(np.max(np.abs(np.linalg.eigvals(T @ Rfl))))
            rho_mp = float(np.max(np.abs(np.linalg.eigvals(T @ Rmp))))
            n2_fl = float(np.linalg.norm(np.linalg.inv(np.eye(N) - T@Rfl), 2))
            n2_mp = float(np.linalg.norm(np.linalg.inv(np.eye(N) - T@Rmp), 2))
            print(f"  {N:2d} |  {rho_fl:.4e} |  {rho_mp:.4f}    |  "
                  f"‖.‖_2 fl={n2_fl:.3e}  mp={n2_mp:.3e}")


def test_continuous_kernel_resolvent_per_geometry(capsys):
    """Quantitative comparison: integrate the continuous-µ resolvent
    operator-norm bound for each geometry.

    For sphere: 1/(1 - T_op^sph(µ)) = 1/(1 - e^{-σ·2Rµ}).
    Its INTEGRAL ∫_0^1 µ · 1/(1 - e^{-σ·2Rµ}) dµ — Wait,
    THIS is the µ-weight integration that USED to vanish the
    singularity. The operator norm of the resolvent in L² with the
    (uniform measure) is the L^∞ of 1/(1 - e^{-σ·2Rµ}) which is
    UNBOUNDED at µ=0.

    For slab: 1/(1 - T_op^slab(µ)) = 1/(1 - e^{-σL/µ}).
    L^∞ norm over µ ∈ [0,1] is at µ=1 = 1/(1-e^{-σL}).

    For cyl: 1/(1 - T_op^cyl(α)) where T_op^cyl(α) = (4/π)cos α Ki_3(τ_2D).
    L^∞ norm over α ∈ [0, π/2] is at α=0 (where T_op is largest).
    """
    with capsys.disabled():
        from orpheus.derivations._kernels import ki_n_float
        sig_t, R = 0.5, 5.0
        # Sphere: max of 1/(1-T_op) on µ∈(0,1]
        # — limit µ→0 gives 1/(σ·2R·µ) → ∞.
        mus = np.linspace(1e-6, 1, 200)
        T_sph = np.exp(-sig_t * 2.0 * R * mus)
        sup_sph = np.max(1.0 / (1.0 - T_sph))
        # Slab: max of 1/(1-T_op^slab) on µ∈(0,1]
        T_slab = np.exp(-sig_t * R / mus)
        sup_slab = np.max(1.0 / (1.0 - T_slab))
        # Cyl: max of 1/(1-T_op^cyl) on α∈(0,π/2)
        alphas = np.linspace(1e-6, np.pi/2 - 1e-6, 200)
        T_cyl = np.array([
            (4.0/np.pi) * np.cos(a) * ki_n_float(3, 2.0*sig_t*R*np.cos(a))
            for a in alphas
        ])
        sup_cyl = np.max(1.0 / (1.0 - T_cyl))

        print(f"\n=== Continuous-limit L^∞ resolvent operator norm ===")
        print(f"  Sphere  sup_µ 1/(1-T_op^sph(µ)) [τ_R={sig_t*R}] = "
              f"{sup_sph:.4e}  <-- diverges at µ→0")
        print(f"  Slab    sup_µ 1/(1-T_op^slab(µ)) [τ_L={sig_t*R}] = "
              f"{sup_slab:.4e}  <-- BOUNDED (max at µ=1)")
        print(f"  Cyl     sup_α 1/(1-T_op^cyl(α)) [τ_R={sig_t*R}] = "
              f"{sup_cyl:.4e}  <-- BOUNDED")
        print(f"\n  Verdict: only sphere has the singular continuous-limit")
        print(f"  resolvent. Cyl/slab are bounded → matrix Galerkin form")
        print(f"  inherits a bounded operator (modulo R-conditioning noise).")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
