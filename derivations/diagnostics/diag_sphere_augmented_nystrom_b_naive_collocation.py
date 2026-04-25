"""Diagnostic B — naive Nyström-collocation on µ ∈ [0,1] for sphere G_bc.

Issue #132 augmented-Nyström investigation.

Hypothesis tested
-----------------
Build G_bc[i, m] = w_m · K(r_i, µ_m) where {µ_m, w_m} are M-node
Gauss-Legendre on µ ∈ [0, 1] (the user's explicit suggestion). Verify
that

    Σ_m G_bc[i, m] · ψ^-(µ_m) → exact scalar flux

as M → ∞ for the canonical Lambertian ψ^-(µ) ≡ 1/π (which gives back
the existing G_bc).

Predicted failure mode
----------------------
The per-angle kernel K(r_i, µ) has an inverse-square-root endpoint
singularity at µ = µ_min(r_i) = sqrt(1 - (r_i/R)²). This is INTRINSIC
to the geometry (rays grazing the surface tangentially carry a 1/|cos θ|
weight in the change of variables θ → µ). Plain Gauss-Legendre on
µ ∈ [0, 1] (an r-INDEPENDENT grid that the augmented system requires)
converges only as O(1/M) instead of spectrally — confirmed in
diag_sphere_augmented_nystrom_a, where 1024 nodes still gave 4×10⁻⁴
relative error against G_bc_ref.

This diagnostic quantifies the convergence rate at multiple r_i.
If the rate is power-law (O(1/M^p) for p ≤ 2), the augmented-Nyström
scheme is structurally weak — even at M = 64 the per-angle kernel
representation will not match the existing rank-1 Mark closure to the
1e-6 tolerance the L1 verification suite demands.

If this diagnostic shows fast convergence (M=16 → 1e-6), the path is
viable; if it shows slow convergence (M=64 → 1e-3), the path has a
structural flaw at the kernel-discretization level that no eigenvalue
post-processing will fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow standalone pytest discovery (no PYTHONPATH manipulation needed).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_sphere_augmented_nystrom_a_kernel_derivation import (  # noqa: E402
    K_per_angle_via_observer,
    _G_bc_uniform_observer,
    _mu_min,
)


def G_bc_via_naive_nystrom(r_i: float, R: float, sig_t: float,
                            M: int) -> tuple[float, np.ndarray]:
    """Build G_bc[i, m] = w_m · K(r_i, µ_m) on M-node GL on µ ∈ [0, 1].

    Returns (G_bc_total, G_bc_row) where G_bc_total is the M=1
    Lambertian collapse and G_bc_row is the per-mode array.
    """
    pts, wts = np.polynomial.legendre.leggauss(M)
    mu_pts = 0.5 * (pts + 1.0)
    mu_wts = 0.5 * wts
    K_vals = np.array([K_per_angle_via_observer(r_i, mu, R, sig_t)
                       for mu in mu_pts])
    G_bc_row = mu_wts * K_vals
    # Lambertian collapse: ψ^- ≡ 1/π → φ = (1/π) · Σ_m G_bc[i,m]
    G_bc_total = float(np.sum(G_bc_row) / np.pi)
    return G_bc_total, G_bc_row


@pytest.mark.parametrize("r_i,R,sig_t", [
    (0.3, 1.0, 0.5),
    (0.5, 1.0, 0.5),
    (0.7, 1.0, 1.0),
    (0.9, 1.0, 2.0),
])
def test_naive_nystrom_convergence_rate(r_i, R, sig_t):
    """Quantify how fast ‖G_bc_naive(M) − G_bc_ref‖ → 0."""
    G_ref = _G_bc_uniform_observer(r_i, R, sig_t, N=2048)
    rel_errs = {}
    for M in [4, 8, 16, 32, 64, 128, 256]:
        G_M, _ = G_bc_via_naive_nystrom(r_i, R, sig_t, M)
        rel_errs[M] = abs(G_M - G_ref) / max(abs(G_ref), 1e-30)
    # Fit log-log slope between M=32 and M=256
    Ms = np.array([32, 64, 128, 256])
    es = np.array([rel_errs[M] for M in Ms])
    slope = np.polyfit(np.log(Ms), np.log(es), 1)[0]
    print(f"\n  r_i={r_i}, R={R}, σ_t={sig_t}: convergence slope = {slope:.3f}")
    print(f"    Errors: " + ", ".join(f"M={M}:{rel_errs[M]:.2e}"
                                       for M in [4, 16, 64, 256]))
    # The hypothesis: slope worse than -1.5 confirms structural defect.
    # Spectral convergence should give exponential decay; we expect O(1/M).
    assert slope < 0.0, f"Naive Nyström not converging at all (slope {slope})"


def main():
    print("=" * 78)
    print("Naive Nyström-collocation on µ ∈ [0,1]: convergence vs M")
    print("=" * 78)
    print()
    print(f"  {'r_i':>5} {'R':>5} {'σ_t':>5} {'µ_min':>7} {'G_ref':>12}")
    for r_i, R, sig_t in [(0.3, 1.0, 0.5), (0.5, 1.0, 0.5),
                          (0.7, 1.0, 1.0), (0.9, 1.0, 2.0)]:
        mm = _mu_min(r_i, R)
        G_ref = _G_bc_uniform_observer(r_i, R, sig_t, N=2048)
        print(f"  {r_i:>5.2f} {R:>5.2f} {sig_t:>5.2f} "
              f"{mm:>7.4f} {G_ref:>12.6e}")
        print(f"    {'M':>5} {'G_naive':>14} {'rel_err':>12}")
        for M in [4, 8, 16, 32, 64, 128, 256, 512]:
            G_M, _ = G_bc_via_naive_nystrom(r_i, R, sig_t, M)
            rel_err = abs(G_M - G_ref) / max(abs(G_ref), 1e-30)
            print(f"    {M:>5d} {G_M:>14.10f} {rel_err:>12.2e}")
        Ms = np.array([32, 64, 128, 256, 512])
        es = np.array([])
        for M in Ms:
            G_M, _ = G_bc_via_naive_nystrom(r_i, R, sig_t, M)
            es = np.append(es, abs(G_M - G_ref) / max(abs(G_ref), 1e-30))
        slope = np.polyfit(np.log(Ms), np.log(es), 1)[0]
        print(f"    log-log slope (M≥32) = {slope:.3f}  "
              f"(spectral if << -3, power if ~ -1)")
        print()


if __name__ == "__main__":
    main()
