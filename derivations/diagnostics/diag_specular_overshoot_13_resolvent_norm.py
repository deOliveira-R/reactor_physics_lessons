"""Diagnostic: operator norm of (I-T·R)⁻¹ as N grows.

Created by numerics-investigator on 2026-04-27.

Hypothesis: ‖(I-T·R)⁻¹‖ → ∞ as N → ∞ because the continuous limit
operator 1/(1-e^(-σ·2Rµ)) is unbounded at µ = 0.

If true, the matrix (I-T·R)⁻¹ should grow without bound, confirming
that the multi-bounce closure CANNOT converge as N → ∞ in matrix form.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from orpheus.derivations.peierls_geometry import (
    _shifted_legendre_eval,
    reflection_specular,
)


def shifted_legendre(n, mu):
    return _shifted_legendre_eval(n, mu)


def build_T_spec_sphere(sigt, R, N, n_quad=128):
    nodes, wts = leggauss(n_quad)
    mu = 0.5 * (nodes + 1.0)
    w = 0.5 * wts
    chord = 2.0 * R
    decay = np.exp(-sigt * chord * mu)
    T = np.zeros((N, N))
    for m in range(N):
        Pm = shifted_legendre(m, mu)
        for n in range(N):
            Pn = shifted_legendre(n, mu)
            T[m, n] = 2.0 * np.sum(w * mu * Pm * Pn * decay)
    return T


def test_resolvent_norm_grows(capsys):
    with capsys.disabled():
        R = 5.0
        sigt = 0.5
        print(f"\n=== ‖(I-T·R)⁻¹‖ as N grows; thin τ_R=2.5 ===")
        print(f"  N | ρ(T·R) | ‖(I-TR)⁻¹‖_2 | ‖(I-TR)⁻¹‖_F | trace((I-TR)⁻¹)")

        for N in (1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25):
            T = build_T_spec_sphere(sigt, R, N)
            R_op = reflection_specular(N)
            TR = T @ R_op
            try:
                ITR_inv = np.linalg.inv(np.eye(N) - TR)
                rho = float(np.max(np.abs(np.linalg.eigvals(TR))))
                norm2 = float(np.linalg.norm(ITR_inv, 2))
                normF = float(np.linalg.norm(ITR_inv, 'fro'))
                tr = float(np.trace(ITR_inv))
                print(f"  {N:2d}| {rho:.4f} | {norm2:.4e}    | {normF:.4e}    | {tr:.4e}")
            except Exception as e:
                print(f"  {N}: failed: {e}")

        print(f"\n=== Same, very-thin τ_R=1.0 ===")
        sigt = 0.2
        for N in (1, 2, 4, 6, 8, 10, 12, 14, 16, 20):
            T = build_T_spec_sphere(sigt, R, N)
            R_op = reflection_specular(N)
            TR = T @ R_op
            try:
                ITR_inv = np.linalg.inv(np.eye(N) - TR)
                rho = float(np.max(np.abs(np.linalg.eigvals(TR))))
                norm2 = float(np.linalg.norm(ITR_inv, 2))
                normF = float(np.linalg.norm(ITR_inv, 'fro'))
                tr = float(np.trace(ITR_inv))
                print(f"  N={N:2d}: ρ={rho:.4f}, ‖.‖_2={norm2:.4e}, "
                      f"‖.‖_F={normF:.4e}, tr={tr:.4e}")
            except Exception as e:
                print(f"  N={N}: failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
