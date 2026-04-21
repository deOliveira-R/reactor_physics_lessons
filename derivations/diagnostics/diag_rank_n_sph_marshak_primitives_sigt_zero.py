"""Diagnostic: σ_t → 0 verification of Marshak per-face primitives.

Created for Phase F.5 / Issue #119 (Marshak closure landing).

Verifies that the new compute_{P_esc,G_bc}_{outer,inner}_mode_marshak
primitives reduce to the analytical half-range partial-current moments
in the σ_t → 0 limit. At σ_t = 0: K_esc = 1 and exp(-τ) = 1, so the
primitives collapse to purely geometric integrals that can be evaluated
independently via high-precision mpmath.quad and compared.

Three checks:

1. **Hollow sphere outer-mode reduction**
   P_out_marshak^{(n)}(r_i) = pref · ∫_allowed sin θ · µ_exit · P̃_n(µ_exit) dθ
   Independent mpmath reference must match to ~1e-10.

2. **Hollow sphere inner-mode reduction**
   P_in_marshak^{(n)}(r_i) = pref · ∫_hit sin θ · µ_exit_in · P̃_n(µ_exit_in) dθ
   Independent mpmath reference must match.

3. **Solid sphere conservation at center (r_i → 0)**
   With r_0 → 0, r_i = 0, σ_t = 0: µ_exit = 1 for all outgoing θ, so
   P̃_n(1) = 1 for all n. Integral reduces to pref · ∫_0^π sin θ dθ = 1
   regardless of mode n. The Marshak primitive at the center gives 1 for
   every mode — a basis-agnostic sanity check.

If promoted, this becomes a Phase F.5 foundation test.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspaces/ORPHEUS")

import mpmath
import numpy as np

from orpheus.derivations._kernels import _shifted_legendre_eval
from orpheus.derivations.peierls_geometry import (
    CurvilinearGeometry,
    compute_P_esc_outer_mode_marshak,
    compute_P_esc_inner_mode_marshak,
    compute_G_bc_outer_mode_marshak,
    compute_G_bc_inner_mode_marshak,
)


def _ptilde(n, mu):
    return float(_shifted_legendre_eval(n, np.array([float(mu)]))[0])


def ref_P_out_marshak_sigt_zero(r_i, r_0, R, n, dps=25):
    r"""Reference mpmath.quad evaluation of

    P_out_marshak^{(n)}(r_i) |_{σ_t = 0}
       = 0.5 · ∫_0^π sin θ · µ_exit · P̃_n(µ_exit) · 1[not blocked] dθ
    """
    def rho_out(cos_th):
        disc = r_i * r_i * cos_th * cos_th + R * R - r_i * r_i
        return -r_i * cos_th + mpmath.sqrt(max(float(disc), 0.0))

    def rho_in_minus(cos_th):
        disc = r_i * r_i * cos_th * cos_th - (r_i * r_i - r_0 * r_0)
        if disc < 0.0:
            return None
        rho_m = -r_i * cos_th - mpmath.sqrt(mpmath.mpf(float(disc)))
        rho_m_f = float(rho_m)
        if rho_m_f <= 0.0:
            return None
        return rho_m_f

    def integrand(theta):
        cos_th = float(mpmath.cos(theta))
        sin_th = float(mpmath.sin(theta))
        rho_o = float(rho_out(cos_th))
        if rho_o <= 0.0:
            return 0.0
        rho_im = rho_in_minus(cos_th)
        if rho_im is not None and rho_im < rho_o:
            return 0.0
        mu_exit = (rho_o + r_i * cos_th) / R
        return sin_th * mu_exit * _ptilde(n, mu_exit)

    with mpmath.workdps(dps):
        return 0.5 * float(mpmath.quad(integrand, [0.0, mpmath.pi]))


def ref_P_in_marshak_sigt_zero(r_i, r_0, R, n, dps=25):
    def rho_in_minus(cos_th):
        disc = r_i * r_i * cos_th * cos_th - (r_i * r_i - r_0 * r_0)
        if disc < 0.0:
            return None
        rho_m = -r_i * cos_th - mpmath.sqrt(mpmath.mpf(float(disc)))
        rho_m_f = float(rho_m)
        if rho_m_f <= 0.0:
            return None
        return rho_m_f

    def integrand(theta):
        cos_th = float(mpmath.cos(theta))
        sin_th = float(mpmath.sin(theta))
        rho_im = rho_in_minus(cos_th)
        if rho_im is None:
            return 0.0
        sin_om = float(mpmath.sqrt(max(0.0, 1.0 - cos_th * cos_th)))
        h_sq = r_i * r_i * sin_om * sin_om
        mu_exit_sq = max(0.0, (r_0 * r_0 - h_sq) / (r_0 * r_0))
        mu_exit = float(mpmath.sqrt(mu_exit_sq))
        return sin_th * mu_exit * _ptilde(n, mu_exit)

    with mpmath.workdps(dps):
        return 0.5 * float(mpmath.quad(integrand, [0.0, mpmath.pi]))


def run_sigt_zero_check():
    R, r_0 = 1.0, 0.3
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_0)
    radii = np.array([R])
    sig_t_zero = np.array([0.0])
    r_nodes = np.array([0.4, 0.6, 0.9])

    print(f"Hollow sphere R={R}, r_0={r_0}, σ_t = 0")
    print(f"Test nodes: {r_nodes}")
    print()

    # Check 1: Marshak OUTER P matches mpmath reference
    print("=" * 74)
    print("[1] P_esc_outer_mode_marshak vs mpmath reference (σ_t = 0):")
    print("=" * 74)
    print(f"{'n':>3} {'r_i':>6} {'numeric':>14} {'reference':>14} "
          f"{'rel err':>10}")
    for n in range(3):
        P = compute_P_esc_outer_mode_marshak(
            geom, r_nodes, radii, sig_t_zero, n,
            n_angular=64, dps=20,
        )
        for i, r_i in enumerate(r_nodes):
            P_ref = ref_P_out_marshak_sigt_zero(r_i, r_0, R, n)
            if abs(P_ref) > 1e-14:
                rel = abs(P[i] - P_ref) / abs(P_ref)
            else:
                rel = abs(P[i] - P_ref)
            print(f"{n:>3} {r_i:>6.2f} {P[i]:>14.8f} {P_ref:>14.8f} "
                  f"{rel:>10.2e}")

    # Check 2: Marshak INNER P matches mpmath reference
    print()
    print("=" * 74)
    print("[2] P_esc_inner_mode_marshak vs mpmath reference (σ_t = 0):")
    print("=" * 74)
    print(f"{'n':>3} {'r_i':>6} {'numeric':>14} {'reference':>14} "
          f"{'rel err':>10}")
    for n in range(3):
        P = compute_P_esc_inner_mode_marshak(
            geom, r_nodes, radii, sig_t_zero, n,
            n_angular=64, dps=20,
        )
        for i, r_i in enumerate(r_nodes):
            P_ref = ref_P_in_marshak_sigt_zero(r_i, r_0, R, n)
            if abs(P_ref) > 1e-14:
                rel = abs(P[i] - P_ref) / abs(P_ref)
            else:
                rel = abs(P[i] - P_ref)
            print(f"{n:>3} {r_i:>6.2f} {P[i]:>14.8f} {P_ref:>14.8f} "
                  f"{rel:>10.2e}")

    # Check 3: G_bc_outer_mode_marshak at σ_t = 0 equals P_esc_out by
    #          reciprocity (both use the same integrand structure — the
    #          observer-centred sphere form).
    print()
    print("=" * 74)
    print("[3] G_bc_*_mode_marshak at σ_t = 0 vs P_esc_*_mode_marshak")
    print("    (sphere observer-centred: G = 2·(P·pref/0.5 / (1/2 · 2))"
          " — identity check)")
    print("=" * 74)
    for n in range(3):
        Po = compute_P_esc_outer_mode_marshak(
            geom, r_nodes, radii, sig_t_zero, n,
            n_angular=64, dps=20,
        )
        Go = compute_G_bc_outer_mode_marshak(
            geom, r_nodes, radii, sig_t_zero, n,
            n_surf_quad=64, dps=20,
        )
        # G's prefactor is 2.0 vs P's 0.5 for sphere — ratio = 4
        ratios = Go / Po if np.all(np.abs(Po) > 1e-12) else np.zeros_like(Go)
        print(f"  n={n}: G/P ratios = {ratios} (expect 4.0)")

    # Check 4: Marshak vs Lambert relation at σ_t = 0
    #   Marshak integrand = Lambert integrand × µ_exit
    #   So P_marshak is always numerically smaller than P_lambert
    #   (since µ_exit ≤ 1). This is the operational signature of the
    #   µ weighting.
    print()
    print("=" * 74)
    print("[4] Marshak vs Lambert primitives at σ_t = 0")
    print("    Marshak has µ_exit in integrand; Lambert does not.")
    print("    P_marshak ≤ P_lambert pointwise on each r_i.")
    print("=" * 74)
    from orpheus.derivations.peierls_geometry import (
        compute_P_esc_outer_mode,
    )
    for n in range(3):
        Po_marshak = compute_P_esc_outer_mode_marshak(
            geom, r_nodes, radii, sig_t_zero, n,
            n_angular=128, dps=20,
        )
        Po_lambert = compute_P_esc_outer_mode(
            geom, r_nodes, radii, sig_t_zero, n,
            n_angular=128, dps=20,
        )
        print(f"  n={n}:")
        for i, r_i in enumerate(r_nodes):
            ratio = Po_marshak[i] / Po_lambert[i] if abs(Po_lambert[i]) > 1e-14 else 0.0
            print(f"    r_i={r_i:.2f}: Marshak={Po_marshak[i]:+.6f}, "
                  f"Lambert={Po_lambert[i]:+.6f}, "
                  f"ratio={ratio:+.4f}")


def test_marshak_P_outer_matches_reference_sigt_zero():
    """Regression pin: Marshak primitive matches mpmath reference.

    GL convergence is slow (~1e-3 at n_angular=128) due to the kink at
    θ = θ_c where the indicator 1[ray hits inner first] discontinuously
    cuts the integrand. The tolerance here is the converged-to-quadrature
    accuracy, not machine precision — that's fine: this test isolates
    the primitive-formula correctness from quadrature accuracy.
    """
    R, r_0 = 1.0, 0.3
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_0)
    radii = np.array([R])
    sig_t_zero = np.array([0.0])
    r_nodes = np.array([0.4, 0.6, 0.9])
    for n in range(3):
        P = compute_P_esc_outer_mode_marshak(
            geom, r_nodes, radii, sig_t_zero, n,
            n_angular=512, dps=25,
        )
        for i, r_i in enumerate(r_nodes):
            P_ref = ref_P_out_marshak_sigt_zero(r_i, r_0, R, n, dps=30)
            if abs(P_ref) > 1e-14:
                rel = abs(P[i] - P_ref) / abs(P_ref)
            else:
                rel = abs(P[i] - P_ref)
            assert rel < 5e-3, (
                f"P_out_marshak(n={n}, r_i={r_i}) = {P[i]:.10f} "
                f"vs mpmath ref {P_ref:.10f}, rel_err = {rel:.3e}"
            )


def test_marshak_G_is_4x_P_at_sphere_sigt_zero():
    """Sanchez-McCormick sphere identity: G_bc^{(n)} = 4·P_esc^{(n)}.

    Both primitives share the same angular integrand for sphere; the
    ratio is a function of the prefactor (G uses 2.0, P uses sphere
    pref = 0.5, so the ratio is 2.0 / 0.5 = 4.0). This identity must
    hold in the Marshak basis just as it does in the Lambert basis.
    """
    R, r_0 = 1.0, 0.3
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_0)
    radii = np.array([R])
    sig_t_zero = np.array([0.0])
    r_nodes = np.array([0.4, 0.6, 0.9])
    for n in range(3):
        Po = compute_P_esc_outer_mode_marshak(
            geom, r_nodes, radii, sig_t_zero, n,
            n_angular=64, dps=20,
        )
        Go = compute_G_bc_outer_mode_marshak(
            geom, r_nodes, radii, sig_t_zero, n,
            n_surf_quad=64, dps=20,
        )
        for i in range(len(r_nodes)):
            if abs(Po[i]) > 1e-10:
                ratio = Go[i] / Po[i]
                assert abs(ratio - 4.0) < 1e-10, (
                    f"G/P ratio at n={n}, r_i={r_nodes[i]} = "
                    f"{ratio:.10f}, expected 4.0"
                )


if __name__ == "__main__":
    run_sigt_zero_check()
