"""Diagnostic: resolvent ‖(I-T·R)⁻¹‖_2 for sphere vs cyl vs slab as N grows.

Created by numerics-investigator on 2026-04-28.

Sphere has been shown (memo specular_mb_overshoot_root_cause.md, diag
specular_overshoot_13) to have UNBOUNDED resolvent norm growth with N
because the continuous limit operator 1/(1 - e^{-σ·2Rµ}) blows up at
µ → 0 (grazing rays have chord 2Rµ → 0 so transmission → 1).

This diagnostic answers: does cyl have the same pathology? Does slab?

GEOMETRY-SPECIFIC GRAZING ANALYSIS
----------------------------------
Sphere: chord(µ) = 2Rµ. As µ → 0 (grazing), chord → 0, so
        transmission e^{-2σRµ} → 1 → resolvent diverges.

Cylinder: in-plane chord d_2D(α) = 2R cos α. As α → π/2 (in-plane
          grazing), chord → 0, transmission → 1. The polar integration
          over θ_p ∈ [0, π] (sin² θ_p weight in Ki_3) provides
          smoothing but does NOT regularize: at α = π/2, Ki_3(0) = π/4
          (FINITE) — but this finite value combined with the
          µ_3D = sin θ_p · cos α weight integrating against the
          partial-current measure means the cylinder T_oi(α) at grazing
          α STILL gives a non-vanishing transmission floor that
          (I-TR)⁻¹ amplifies.

          RH: cylinder pathology is qualitatively SAME as sphere
          (grazing in-plane chord → 0 ⇒ Ki_3(0) finite ⇒ continuous
          T(α) does not vanish at α=π/2 ⇒ matrix inverse blows up).

Slab: chord(µ) = L/µ. As µ → 0 (grazing), chord → ∞, so
      transmission e^{-σL/µ} → 0 → bounded! Slab grazing rays have
      INFINITE optical depth, NOT zero. The integrand
      µ · e^{-σL/µ} is SMOOTH AND BOUNDED on [0,1] (vanishes at both
      endpoints). The continuous limit operator on slab single-transit
      T_oi has spectrum well-separated from 1; (I - T·R)⁻¹ should
      stay bounded as N → ∞ for slab.

      RH: slab matrix-form (I - T·R)⁻¹ converges as N → ∞.

EXPERIMENT
----------
Build sphere/cyl/slab T at a thin cell (τ ≈ 2.5) for N = 1..16 and
report:
  - ρ(T·R)
  - ‖(I-T·R)⁻¹‖_2
  - condition(I - T·R)

Verdict: per-geometry the answer to "matrix form will fundamentally
diverge" or "matrix form will converge".
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    _shifted_legendre_eval,
    _shifted_legendre_monomial_coefs,
    reflection_specular,
)
from orpheus.derivations._kernels import ki_n_float


# ─── Sphere T (matches compute_T_specular_sphere; pure-µ form) ────────


def build_T_sphere(sig_t: float, R: float, N: int, n_quad: int = 256):
    nodes, wts = np.polynomial.legendre.leggauss(n_quad)
    mu = 0.5 * (nodes + 1.0)
    w = 0.5 * wts
    decay = np.exp(-sig_t * 2.0 * R * mu)
    T = np.zeros((N, N))
    for m in range(N):
        Pm = _shifted_legendre_eval(m, mu)
        for n in range(N):
            Pn = _shifted_legendre_eval(n, mu)
            T[m, n] = 2.0 * np.sum(w * mu * Pm * Pn * decay)
    return T


# ─── Cylinder T (homogeneous, fast float ki_n) ────────────────────────


def build_T_cyl(sig_t: float, R: float, N: int, n_quad: int = 128):
    """Cylinder T per derivation in diag_phase4_01.

    T_mn = (4/π) ∫_0^(π/2) cos α Σ_{k_m,k_n} c_m^{k_m} c_n^{k_n}
                 (cos α)^(k_m+k_n) Ki_(k_m+k_n+3)(τ_2D(α)) dα
    with τ_2D(α) = 2 σ R cos α (homogeneous).
    """
    nodes, wts = np.polynomial.legendre.leggauss(n_quad)
    alpha = 0.5 * (nodes + 1.0) * (np.pi / 2.0)
    aw = wts * (np.pi / 4.0)
    cos_a = np.cos(alpha)

    # Pre-evaluate Ki_(j+3) for j = 0 .. 2N-2 (max k_m + k_n).
    max_kk = 2 * (N - 1)
    Ki_arr = np.zeros((max_kk + 1, n_quad))
    for j in range(max_kk + 1):
        for k in range(n_quad):
            tau = 2.0 * sig_t * R * float(cos_a[k])
            Ki_arr[j, k] = ki_n_float(j + 3, tau)

    coef_list = [_shifted_legendre_monomial_coefs(m) for m in range(N)]

    T = np.zeros((N, N))
    for m in range(N):
        cm = coef_list[m]
        for n in range(N):
            cn = coef_list[n]
            kernel = np.zeros(n_quad)
            for k_m, c_m in enumerate(cm):
                if c_m == 0.0:
                    continue
                for k_n, c_n in enumerate(cn):
                    if c_n == 0.0:
                        continue
                    kk = k_m + k_n
                    kernel += c_m * c_n * (cos_a ** kk) * Ki_arr[kk]
            T[m, n] = (4.0 / np.pi) * np.sum(aw * cos_a * kernel)
    return T


# ─── Slab T (per-face block off-diagonal) ──────────────────────────────


def build_T_slab(sig_t: float, L: float, N: int, n_quad: int = 256):
    """Slab T per derivation in diag_phase4_02. Returns (2N, 2N) with
    only off-diagonal blocks T_oi = T_io populated."""
    nodes, wts = np.polynomial.legendre.leggauss(n_quad)
    mu = 0.5 * (nodes + 1.0)
    w = 0.5 * wts

    decay = np.exp(-sig_t * L / mu)  # NOTE: L/µ, not 2Rµ
    mw = w * mu

    T_oi = np.zeros((N, N))
    for m in range(N):
        Pm = _shifted_legendre_eval(m, mu)
        for n in range(N):
            Pn = _shifted_legendre_eval(n, mu)
            T_oi[m, n] = 2.0 * np.sum(mw * Pm * Pn * decay)

    T = np.zeros((2 * N, 2 * N))
    T[:N, N:] = T_oi
    T[N:, :N] = T_oi
    return T


# ─── Pathology probe ───────────────────────────────────────────────────


def _resolvent_metrics(T, R_op):
    """Return (ρ(TR), ‖(I-TR)^-1‖_2, cond(I-TR))."""
    TR = T @ R_op
    M = np.eye(T.shape[0]) - TR
    try:
        Minv = np.linalg.inv(M)
        rho = float(np.max(np.abs(np.linalg.eigvals(TR))))
        n2 = float(np.linalg.norm(Minv, 2))
        cond = float(np.linalg.cond(M))
        return rho, n2, cond
    except np.linalg.LinAlgError:
        return float('nan'), float('inf'), float('inf')


def test_sphere_resolvent_grows(capsys):
    """Replicate the sphere baseline pathology (sanity check)."""
    with capsys.disabled():
        sig_t, R = 0.5, 5.0  # τ_R = 2.5
        print(f"\n=== SPHERE thin (τ_R = {sig_t * R}) ===")
        print(f"  N | ρ(TR)   | ‖(I-TR)⁻¹‖_2 | cond(I-TR)")
        norms = []
        for N in (1, 2, 3, 4, 6, 8, 12, 16):
            T = build_T_sphere(sig_t, R, N)
            R_op = reflection_specular(N)
            rho, n2, cond = _resolvent_metrics(T, R_op)
            norms.append(n2)
            print(f"  {N:2d}| {rho:.4f} | {n2:>10.3e} | {cond:>10.3e}")
        # Sphere baseline pathology: ‖.‖_2 grows monotonically by >5x
        # over the N range explored.
        assert norms[-1] > 5.0 * norms[0], (
            f"sphere resolvent did NOT grow: {norms[0]:.2f} → {norms[-1]:.2f}"
        )


def test_cyl_resolvent(capsys):
    """Pathology probe for cylinder."""
    with capsys.disabled():
        sig_t, R = 0.5, 5.0  # τ_R = 2.5 (in-plane)
        print(f"\n=== CYLINDER thin (τ_R = {sig_t * R}) ===")
        print(f"  N | ρ(TR)   | ‖(I-TR)⁻¹‖_2 | cond(I-TR)")
        norms = []
        for N in (1, 2, 3, 4, 6, 8, 12):
            T = build_T_cyl(sig_t, R, N)
            R_op = reflection_specular(N)
            rho, n2, cond = _resolvent_metrics(T, R_op)
            norms.append(n2)
            print(f"  {N:2d}| {rho:.4f} | {n2:>10.3e} | {cond:>10.3e}")
        # Print verdict:
        ratio = norms[-1] / norms[0]
        if ratio > 5.0:
            print(f"  CYL VERDICT: resolvent DIVERGES "
                  f"(grew {ratio:.2f}x over N=1..12)")
        else:
            print(f"  CYL VERDICT: resolvent BOUNDED "
                  f"(grew only {ratio:.2f}x over N=1..12)")


def test_slab_resolvent(capsys):
    """Pathology probe for slab (face-block transit kernel)."""
    with capsys.disabled():
        sig_t, L = 0.5, 5.0  # τ_L = 2.5 (matched optical thickness)
        print(f"\n=== SLAB thin (τ_L = {sig_t * L}) ===")
        print(f"  N | ρ(TR)   | ‖(I-TR)⁻¹‖_2 | cond(I-TR)")
        norms = []
        for N in (1, 2, 3, 4, 6, 8, 12, 16):
            T = build_T_slab(sig_t, L, N)
            R_op_face = reflection_specular(N)
            R_op = np.zeros((2 * N, 2 * N))
            R_op[:N, :N] = R_op_face
            R_op[N:, N:] = R_op_face
            rho, n2, cond = _resolvent_metrics(T, R_op)
            norms.append(n2)
            print(f"  {N:2d}| {rho:.4f} | {n2:>10.3e} | {cond:>10.3e}")
        ratio = norms[-1] / norms[0]
        if ratio > 5.0:
            print(f"  SLAB VERDICT: resolvent DIVERGES "
                  f"(grew {ratio:.2f}x over N=1..16)")
        else:
            print(f"  SLAB VERDICT: resolvent BOUNDED "
                  f"(grew only {ratio:.2f}x over N=1..16)")


def test_grazing_floor_continuous_limit(capsys):
    """Compute the continuous-limit operator value at µ → 0 (or
    α → π/2 for cyl) to confirm the geometry-specific grazing
    behaviour."""
    with capsys.disabled():
        # Sphere: T_op(µ) = e^{-2σRµ} → e^0 = 1 at µ=0. So
        # 1/(1-T_op(µ)) → 1/(1-1) = ∞ at grazing.
        sig_t, R = 0.5, 5.0
        # Cyl: at α → π/2, τ_2D → 0, so Ki_3(0) = π/4 ≈ 0.785 (FINITE).
        ki3_at_zero = ki_n_float(3, 0.0)
        print(f"\nGRAZING-LIMIT CONTINUOUS OPERATOR FLOOR:")
        print(f"  Sphere  e^{{-σ·2R·0}}      = {math.exp(0.0):.4f}  (grazing limit µ→0)")
        print(f"  Cyl     Ki_3(0)             = {ki3_at_zero:.4f}  (grazing limit α→π/2)")
        # Slab: e^{-σL/µ} → e^{-∞} = 0 at µ → 0 (grazing has INFINITE chord).
        # Probe at a small µ:
        for mu in (1e-2, 1e-3, 1e-4):
            slab_v = math.exp(-sig_t * R / mu)
            print(f"  Slab    e^{{-σL/{mu}}}    = {slab_v:.4e}  (chord = L/µ)")
        # Sphere's continuous integrand 1/(1 - e^{-σ·2R·µ}) at µ=0 is ∞,
        # but at µ = 1e-3:
        sph_inv = 1.0 / (1.0 - math.exp(-sig_t * 2.0 * R * 1e-3))
        print(f"\n  Sphere 1/(1 - e^{{-σ·2R·1e-3}}) = {sph_inv:.4e}  (grazing pole)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
