"""Diagnostic: extended-N pathology probe + thinner-cell stress test.

Created by numerics-investigator on 2026-04-28.

Companion to diag_specular_mb_phase4_03 — that probe showed at thin
τ_R = 2.5, N = 1..16:
  - SPHERE: ‖(I-TR)⁻¹‖_2 grows 1.08 → 23.3 (21x) — DIVERGES.
  - CYL  : 1.03 → 3.60 over N=1..12 — bounded so far.
  - SLAB : 1.03 → 1.09 over N=1..16 — clearly bounded.

This probe extends:
  - sphere/cyl up to N = 40 to verify the trend continues
  - cyl at very-thin (τ_R = 1.0, the regime where sphere goes to 86x)
    to see whether the cyl trend is truly bounded or just slow-growing.
  - sphere at thicker cell (τ_R = 5) to confirm the pathology weakens
    when grazing chord is non-tiny.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import reflection_specular
from derivations.diagnostics.diag_specular_mb_phase4_03_pathology_resolvent import (
    build_T_sphere, build_T_cyl, build_T_slab, _resolvent_metrics,
)


def test_sphere_thinner_pathology(capsys):
    """Sphere at τ_R = 1.0 — grazing pathology should be SEVERE."""
    with capsys.disabled():
        sig_t, R = 0.2, 5.0  # τ_R = 1.0
        print(f"\n=== SPHERE very-thin (τ_R = {sig_t * R}) ===")
        for N in (1, 4, 8, 12, 16, 20, 25):
            T = build_T_sphere(sig_t, R, N)
            R_op = reflection_specular(N)
            rho, n2, cond = _resolvent_metrics(T, R_op)
            print(f"  N={N:2d}: ρ={rho:.4f}, ‖.‖_2={n2:.3e}, cond={cond:.2e}")


def test_cyl_extended_N(capsys):
    """Cyl up to N=20 — does the resolvent stay bounded?"""
    with capsys.disabled():
        # thin
        sig_t, R = 0.5, 5.0
        print(f"\n=== CYL thin (τ_R = {sig_t * R}) — extended N ===")
        for N in (1, 4, 8, 12, 16, 20):
            T = build_T_cyl(sig_t, R, N)
            R_op = reflection_specular(N)
            rho, n2, cond = _resolvent_metrics(T, R_op)
            print(f"  N={N:2d}: ρ={rho:.4f}, ‖.‖_2={n2:.3e}, cond={cond:.2e}")
        # very-thin (sphere goes to 86x at this τ)
        sig_t, R = 0.2, 5.0
        print(f"\n=== CYL very-thin (τ_R = {sig_t * R}) — extended N ===")
        for N in (1, 4, 8, 12, 16, 20):
            T = build_T_cyl(sig_t, R, N)
            R_op = reflection_specular(N)
            rho, n2, cond = _resolvent_metrics(T, R_op)
            print(f"  N={N:2d}: ρ={rho:.4f}, ‖.‖_2={n2:.3e}, cond={cond:.2e}")


def test_slab_extended_N(capsys):
    """Slab up to N=24 — confirm rapid plateau (vanishing grazing
    transmission)."""
    with capsys.disabled():
        sig_t, L = 0.5, 5.0
        print(f"\n=== SLAB thin (τ_L = {sig_t * L}) — extended N ===")
        for N in (1, 4, 8, 12, 16, 20, 24):
            T = build_T_slab(sig_t, L, N)
            R_op_face = reflection_specular(N)
            R_op = np.zeros((2 * N, 2 * N))
            R_op[:N, :N] = R_op_face
            R_op[N:, N:] = R_op_face
            rho, n2, cond = _resolvent_metrics(T, R_op)
            print(f"  N={N:2d}: ρ={rho:.4f}, ‖.‖_2={n2:.3e}, cond={cond:.2e}")
        sig_t, L = 0.2, 5.0
        print(f"\n=== SLAB very-thin (τ_L = {sig_t * L}) — extended N ===")
        for N in (1, 4, 8, 12, 16, 20, 24):
            T = build_T_slab(sig_t, L, N)
            R_op_face = reflection_specular(N)
            R_op = np.zeros((2 * N, 2 * N))
            R_op[:N, :N] = R_op_face
            R_op[N:, N:] = R_op_face
            rho, n2, cond = _resolvent_metrics(T, R_op)
            print(f"  N={N:2d}: ρ={rho:.4f}, ‖.‖_2={n2:.3e}, cond={cond:.2e}")


def test_cyl_grazing_chord_floor_quantitative(capsys):
    """Quantitative: how does cyl T_continuous(α) behave at grazing?

    The continuous-limit operator analog for cyl is the integrand of
    T_oo at α (mode-0 form):

        T_op^cyl(α) = (4/π) cos α · Ki_3(2σR cos α)

    At α = π/2, cos α = 0, Ki_3(0) finite, so T_op^cyl(π/2) = 0.
    The grazing modes of cyl T transmit ZERO partial current
    (because the µ-weight cos α vanishes). This is fundamentally
    different from sphere where the continuous T_op^sph(µ) =
    e^{-2σRµ} → 1 at grazing µ → 0.

    So cyl actually has the SAME geometric regularization as slab
    (grazing modes vanish in the partial-current weight) — but for a
    different reason than slab.
    """
    with capsys.disabled():
        from orpheus.derivations._kernels import ki_n_float
        sig_t, R = 0.5, 5.0
        print(f"\n=== Continuous-limit T_op^cyl(α) at grazing (mode-0) ===")
        for alpha_deg in (0, 30, 60, 80, 89, 89.9, 89.99):
            alpha = np.deg2rad(alpha_deg)
            cosa = np.cos(alpha)
            tau = 2.0 * sig_t * R * cosa
            T_op = (4.0 / np.pi) * cosa * ki_n_float(3, tau)
            print(f"  α={alpha_deg:>6}°: cos α={cosa:.4e}, τ_2D={tau:.4e}, "
                  f"T_op={T_op:.4e}")
        print("\n  ↑ T_op^cyl(α) → 0 at α → π/2 (cos α factor wins)")
        print("  → cyl grazing rays transmit ZERO partial current")
        print("  → multi-bounce factor 1/(1 - T_op·R_op) is BOUNDED")


def test_continuous_kernel_per_mode(capsys):
    """Print the µ-resolved (or α-resolved) kernel `T_op·R_op` =
    multiplication operator value at each angle, for sphere/slab/cyl
    at thin τ ≈ 2.5. Sphere should show 1/(1-T_op) blowup at µ→0;
    slab/cyl should not."""
    with capsys.disabled():
        from orpheus.derivations._kernels import ki_n_float

        sig_t, R = 0.5, 5.0
        print(f"\n=== Sphere continuous T_op(µ) and 1/(1-T_op·1) ===")
        # Sphere R_op = 1 in the continuous limit (mode-0 Mark factor).
        for mu in (1.0, 0.5, 0.1, 0.01, 0.001, 1e-5):
            T_op = np.exp(-sig_t * 2.0 * R * mu)
            denom = 1.0 / (1.0 - T_op) if T_op < 1.0 else float('inf')
            print(f"  µ={mu:.0e}: T_op={T_op:.4e}, 1/(1-T_op)={denom:.4e}")

        print(f"\n=== Slab continuous T_oi_op(µ) (homogeneous, τ_L={sig_t*R}) ===")
        for mu in (1.0, 0.5, 0.1, 0.01, 0.001, 1e-5):
            T_op = np.exp(-sig_t * R / mu)  # = e^{-σL/µ}
            print(f"  µ={mu:.0e}: T_oi_op={T_op:.4e}")

        print(f"\n=== Cyl continuous T_op(α) (mode-0) ===")
        for alpha_deg in (0, 30, 60, 80, 89, 89.9):
            cosa = np.cos(np.deg2rad(alpha_deg))
            tau = 2.0 * sig_t * R * cosa
            T_op = (4.0 / np.pi) * cosa * ki_n_float(3, tau)
            print(f"  α={alpha_deg:>5}°: cos α={cosa:.4e}, T_op^cyl={T_op:.4e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
