"""Diagnostic: numerical conditioning of R_spec at high N.

Created by numerics-investigator on 2026-04-27.

R_spec = (1/2) M⁻¹ where M is symmetric tridiagonal with M_nn = 1/(2(2n+1))
and M_n,n+1 = (n+1)/(2(2n+1)(2n+3)).

The matrix M has condition number that grows polynomially with N.
At high N, R_spec entries blow up, R_spec applied to *anything*
amplifies. Even rounding error in J⁺ becomes huge after R · J⁺.

Check: cond(M), max(R), and entry-wise behavior. Where does this
become catastrophic?
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    reflection_specular,
)


def build_M(N):
    M = np.zeros((N, N))
    for n in range(N):
        M[n, n] = 1.0 / (2.0 * (2 * n + 1))
        if n + 1 < N:
            off = (n + 1) / (2.0 * (2 * n + 1) * (2 * n + 3))
            M[n, n + 1] = off
            M[n + 1, n] = off
    return M


def test_R_spec_conditioning(capsys):
    """Track cond(M), max(R_spec), eigvals(R_spec) as N grows."""
    with capsys.disabled():
        print(f"\n=== R_spec = (1/2) M⁻¹ conditioning ===")
        print(f"  N | cond(M)    | max(R)   | min eig(R)  | max eig(R)  | max growth")

        max_R_prev = 1.0
        for N in (1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 30):
            M = build_M(N)
            R_op = reflection_specular(N)
            cond_M = np.linalg.cond(M)
            max_R = float(np.max(np.abs(R_op)))
            eigs = np.linalg.eigvals(R_op).real
            min_eig = float(np.min(eigs))
            max_eig = float(np.max(eigs))
            growth = max_R / max_R_prev
            print(f"  {N:2d}| {cond_M:.2e}  | {max_R:8.2f} | {min_eig:.4f}      | "
                  f"{max_eig:.4f}     | {growth:.2f}x")
            max_R_prev = max_R


def test_J_plus_amplification_by_R(capsys):
    """If we feed J⁺ = small noise, what does R·J⁺ look like?"""
    with capsys.disabled():
        print(f"\n=== R · (small perturbation) — does noise amplify? ===")
        rng = np.random.default_rng(42)
        for N in (4, 8, 12, 16, 20, 25):
            R_op = reflection_specular(N)
            # Random unit vector perturbation
            noise = rng.standard_normal(N)
            noise /= np.linalg.norm(noise)
            R_noise = R_op @ noise
            print(f"  N={N:2d}: noise norm = 1.0, |R·noise| = {np.linalg.norm(R_noise):.2f}, "
                  f"max |R·noise| component = {np.max(np.abs(R_noise)):.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
