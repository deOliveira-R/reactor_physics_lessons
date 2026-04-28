"""Diagnostic 05: how fast does the actual ψ⁺(µ) at the surface decay
in shifted-Legendre modes for thin vs thick sphere?

Created by numerics-investigator on 2026-04-27.

Hypothesis 4 from the user: rank-N truncation may not capture the
sharp angular peak of ψ⁺(µ) at thin-cell surface. The diagnostic
projects ψ⁺(µ) (estimated from a uniform volume source — the
'reference' angular flux at surface for an infinite-medium-equivalent
case) onto P̃_0..P̃_{N-1} and reports the mode magnitudes.

For an infinite-medium-equivalent sphere with isotropic source Q_v
uniformly distributed in the volume, the surface angular flux ψ⁺(µ)
has the analytical form

    ψ⁺(µ) = (Q_v / (4π σ_t)) · (1 - exp(-σ_t · L(µ)))

where L(µ) = 2R · µ is the chord length from inner surface to outer
surface for a sphere ray with surface cosine µ relative to the
outward normal.

For thin sphere (σ_t · 2R = 5 MFP): the peak at µ=1 (forward) is
~Q_v/(4π σ_t) · (1 - e^-5) ≈ 0.99 · (Q_v/4πσ_t).
At µ=0 (grazing) ψ⁺ → 0.
This is a fairly steep ramp, so high N modes will have non-negligible
content.

For thick sphere (σ_t · 2R = 10 MFP): essentially saturated for all
µ > 0.1. Much closer to constant; rank-1 captures most of it.

Project ψ⁺(µ) = Σ_n a_n P̃_n(µ) by:
    a_n = (2n+1) ∫_0^1 ψ⁺(µ) P̃_n(µ) dµ.

Report a_n / a_0 (relative magnitudes).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import legendre
from numpy.polynomial.legendre import leggauss


def shifted_legendre(n, mu):
    """P̃_n(µ) = P_n(2µ - 1)."""
    return legendre(n)(2.0 * mu - 1.0)


def project_psi_plus(psi_func, n_modes, n_quad=128):
    """Project psi^+(mu) on shifted-Legendre P̃_0..P̃_{n_modes-1}."""
    nodes, wts = leggauss(n_quad)
    mu = 0.5 * (nodes + 1.0)
    w_mu = 0.5 * wts
    psi = np.array([psi_func(m) for m in mu])
    coefs = np.zeros(n_modes)
    for n in range(n_modes):
        Pn = shifted_legendre(n, mu)
        # Standard Legendre orthonormality on [0,1]: int_0^1 P̃_n P̃_m dµ
        #   = 1/(2n+1) δ_{nm}, so a_n = (2n+1) ∫ ψ P̃_n dµ.
        coefs[n] = (2 * n + 1) * np.sum(w_mu * psi * Pn)
    return coefs


@pytest.mark.parametrize(
    "tag,sigt,R",
    [
        ("thin τR=2.5", 0.5, 5.0),
        ("thick τR=5.0", 1.0, 5.0),
    ],
)
def test_psi_plus_mode_decay(tag, sigt, R, capsys):
    with capsys.disabled():
        # ψ⁺(µ) ~ (1 - exp(-σ_t · 2R · µ))  (chord cosine convention)
        # The 2R · µ chord length is for a ray exiting at outward cosine µ
        # from a surface point on a sphere — it's the ray going TO the
        # opposite surface point. (Independent of which surface point by
        # symmetry.)
        chord_thru = 2.0 * R
        def psi_plus(mu):
            if mu <= 0.0:
                return 0.0
            return 1.0 - np.exp(-sigt * chord_thru * mu)

        coefs = project_psi_plus(psi_plus, n_modes=8)
        print(f"\n=== {tag} (σ_t·2R = {sigt*chord_thru}) ===")
        print(f"  shifted-Legendre coefs of ψ⁺(µ):")
        for n, a in enumerate(coefs):
            ratio = a / coefs[0] if coefs[0] != 0 else 0.0
            print(f"    a_{n} = {a:+.6e}, |a_n/a_0| = {abs(ratio):.6e}")

        # Reconstruction error
        nodes, wts = leggauss(256)
        mu = 0.5 * (nodes + 1.0)
        w_mu = 0.5 * wts
        psi_true = np.array([psi_plus(m) for m in mu])
        l2_norm = np.sqrt(np.sum(w_mu * psi_true ** 2))
        for N_trunc in (1, 2, 3, 4, 6, 8):
            psi_trunc = sum(
                coefs[n] * shifted_legendre(n, mu)
                for n in range(N_trunc)
            )
            err = np.sqrt(np.sum(w_mu * (psi_true - psi_trunc) ** 2))
            print(f"  N={N_trunc}: L2 truncation error/L2 norm = "
                  f"{err / l2_norm:.6e}")


if __name__ == "__main__":
    import sys
    for tag, sigt, R in [
        ("thin τR=2.5 (σ_t·2R=5)", 0.5, 5.0),
        ("thick τR=5.0 (σ_t·2R=10)", 1.0, 5.0),
    ]:
        chord_thru = 2.0 * R
        def psi_plus(mu, _sigt=sigt, _ct=chord_thru):
            if mu <= 0.0:
                return 0.0
            return 1.0 - np.exp(-_sigt * _ct * mu)
        coefs = project_psi_plus(psi_plus, n_modes=12, n_quad=256)
        print(f"\n=== {tag} ===")
        for n, a in enumerate(coefs):
            ratio = a / coefs[0] if coefs[0] != 0 else 0.0
            print(f"  a_{n} = {a:+.6e}, |a_n/a_0| = {abs(ratio):.6e}")

        nodes, wts = leggauss(512)
        mu = 0.5 * (nodes + 1.0)
        w_mu = 0.5 * wts
        psi_true = np.array([psi_plus(m) for m in mu])
        l2_norm = np.sqrt(np.sum(w_mu * psi_true ** 2))
        for N_trunc in (1, 2, 3, 4, 6, 8, 10):
            psi_trunc = sum(
                coefs[n] * shifted_legendre(n, mu)
                for n in range(N_trunc)
            )
            err = np.sqrt(np.sum(w_mu * (psi_true - psi_trunc) ** 2))
            print(f"  N={N_trunc}: L2 trunc err / L2 = "
                  f"{err / l2_norm:.6e}")
    sys.exit(0)
