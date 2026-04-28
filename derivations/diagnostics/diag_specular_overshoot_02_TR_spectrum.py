"""Diagnostic: examine the spectral structure of T·R vs the continuous T_op.

Created by numerics-investigator on 2026-04-27.
If this test catches a real bug, promote to ``tests/derivations/test_peierls_specular_bc.py``.

QUESTION: In the continuous (function-space) picture, T_op is a multiplication
operator on µ ∈ [0,1] with kernel f(µ) → e^(-σ·2Rµ) f(µ), so its
spectrum is the range {e^(-σ·2Rµ) : µ ∈ [0,1]} = (e^(-σ·2R), 1] ⊂ (0,1].
R_op is identity (specular preserves ψ). So T·R should have spectrum
in (0,1] in the continuous sense.

But the rank-N matrix T·R has eigenvalues that GROW with N (max
ρ(T·R) = 0.077 at N=1 → 0.82 at N=8). What's happening?

We compare:
1. Eigvals of T·R matrix at increasing N.
2. The function exp(-2σRµ) sampled and projected onto the basis —
   does the rank-N Galerkin spectrum approach the continuous spectrum?
3. Is the "extra" eigenvalue at N≥2 spurious (from basis incompleteness)?
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
    decay = np.exp(-sigt * 2.0 * R * mu)
    T = np.zeros((N, N))
    for m in range(N):
        Pm = shifted_legendre(m, mu)
        for n in range(N):
            Pn = shifted_legendre(n, mu)
            T[m, n] = 2.0 * np.sum(w * mu * Pm * Pn * decay)
    return T


def build_M(N):
    """M_nm = ∫_0^1 µ P̃_n(µ) P̃_m(µ) dµ."""
    M = np.zeros((N, N))
    for n in range(N):
        M[n, n] = 1.0 / (2.0 * (2 * n + 1))
        if n + 1 < N:
            off = (n + 1) / (2.0 * (2 * n + 1) * (2 * n + 3))
            M[n, n + 1] = off
            M[n + 1, n] = off
    return M


def build_M0(N):
    """M0_nm = ∫_0^1 P̃_n(µ) P̃_m(µ) dµ (no µ weight). Diagonal."""
    M0 = np.zeros((N, N))
    for n in range(N):
        M0[n, n] = 1.0 / (2 * n + 1)
    return M0


def test_TR_spectrum_grows_with_N(capsys):
    """ρ(T·R) grows with N — pin and analyze the growth pattern."""
    with capsys.disabled():
        sigt = 0.5
        R = 5.0
        chord_max = 2 * R
        tau_R = sigt * R
        max_attenuation = 1.0  # at µ=0, e^0 = 1
        min_attenuation = np.exp(-sigt * chord_max)  # at µ=1
        print(f"\n=== T·R spectrum scan, σ_t={sigt}, R={R}, τ_R={tau_R} ===")
        print(f"Continuous T_op spectrum: ({min_attenuation:.4f}, 1]")
        print(f"Largest physically meaningful eigenvalue should be ≤ 1.\n")

        for N in (1, 2, 3, 4, 6, 8, 10, 12):
            T = build_T_spec_sphere(sigt, R, N)
            R_op = reflection_specular(N)
            M = build_M(N)
            TR = T @ R_op
            eigs_TR = np.linalg.eigvals(TR)
            eigs_T = np.linalg.eigvals(T)
            eigs_R = np.linalg.eigvals(R_op)

            sorted_eigs_TR = sorted(np.abs(eigs_TR), reverse=True)
            print(f"  N={N:2d}: ρ(T·R) = {sorted_eigs_TR[0]:.6f}, "
                  f"all eigs(T·R): {[f'{x:.4f}' for x in sorted_eigs_TR]}")
            print(f"        eigs(T): {sorted(np.abs(eigs_T), reverse=True)}")
            print(f"        eigs(R): {sorted(np.abs(eigs_R), reverse=True)}")
            # Check identity 2·M·R = I (this is the core of R_op)
            ident_check = 2.0 * M @ R_op
            err_id = np.max(np.abs(ident_check - np.eye(N)))
            print(f"        |2·M·R - I|_inf = {err_id:.2e} (should be 0)")


def test_T_eigenvalues_vs_continuous_spectrum(capsys):
    """Are T's eigenvalues in (0, 1]? They are EXPECTED to be."""
    with capsys.disabled():
        sigt = 0.5
        R = 5.0
        print(f"\n=== T's eigenvalue range ===")
        print("Continuous T_op = mult by exp(-σ·2R·µ); spectrum = "
              f"({np.exp(-sigt*2*R):.4f}, 1].\n")

        for N in (1, 2, 4, 6, 8, 10):
            T = build_T_spec_sphere(sigt, R, N)
            # T is symmetric, eigvals real.
            eigs = sorted(np.linalg.eigvalsh(T), reverse=True)
            print(f"  N={N:2d}: T eigs = {[f'{x:.6f}' for x in eigs]}")
            # T should have all eigenvalues > 0 (positive operator)
            assert all(e > -1e-12 for e in eigs), f"T has negative eigvals at N={N}"
            # The largest eigenvalue should be small (≤ T_00 = P_ss ≈ 0.077)
            # because T is a Galerkin projection of the multiplication
            # operator, and its operator norm under the µ-weighted inner
            # product is bounded by sup |kernel| = max e^(-σ·2Rµ) = 1.

            # But wait — these are the eigenvalues of T as a matrix in the
            # standard inner product, not as an operator under µ-weighted
            # inner product. The matrix eigenvalues need NOT match the
            # operator eigenvalues!


def test_TR_under_correct_inner_product(capsys):
    """Compute T·R eigvals under the µ-weighted inner product (the
    *physical* inner product for partial currents).

    The natural inner product for partial currents J⁺ is
    <a, b> = a^T M b where M is the partial-current overlap.
    The matrix R = (1/2)M⁻¹ is NOT self-adjoint under standard product
    but it IS self-adjoint under M-weighted product.

    Under the M-inner product, the relevant "spectrum" of T·R is given
    by generalized eigenvalues T·R·v = λ·M·v (or equivalently, eigvals
    of M^(-1/2) T R M^(1/2)).

    If under the *correct* inner product, ρ(T·R) ≤ P_ss^(max) and
    BOUNDED, the matrix-form spectral radius is just an artifact of
    the wrong inner product.
    """
    with capsys.disabled():
        sigt = 0.5
        R = 5.0
        print(f"\n=== T·R spectrum under M-weighted inner product ===")

        for N in (1, 2, 4, 6, 8, 10):
            T = build_T_spec_sphere(sigt, R, N)
            R_op = reflection_specular(N)
            M = build_M(N)

            TR_mat = T @ R_op
            # Standard eigvals
            eigs_std = sorted(np.abs(np.linalg.eigvals(TR_mat)), reverse=True)

            # Generalized eigvals: T R v = λ M v
            # equivalently λ = eigvals( M⁻¹ T R )
            try:
                from scipy.linalg import eig
                eigs_gen, _ = eig(TR_mat, M)
                eigs_gen = sorted(np.abs(eigs_gen), reverse=True)
            except Exception:
                eigs_gen = ["ERR"]

            # Try a third interpretation: T should map J⁺ → J⁺_new.
            # Under the J⁺-natural M-weighted norm, the operator
            # T R: J⁺ → J⁺ has spectrum given by similarity transform
            # M^{1/2} T R M^{-1/2}.
            from scipy.linalg import sqrtm
            Msr = sqrtm(M)
            Msr_inv = np.linalg.inv(Msr)
            sym_TR = Msr @ TR_mat @ Msr_inv
            eigs_sym = sorted(np.abs(np.linalg.eigvals(sym_TR)), reverse=True)

            print(f"  N={N:2d}: std ρ={eigs_std[0]:.4f}, "
                  f"M-weighted ρ={eigs_gen[0]:.4f}, "
                  f"M-similarity ρ={eigs_sym[0]:.4f}")
            print(f"        std eigs:    {[f'{x:.4f}' for x in eigs_std]}")
            print(f"        gen eigs:    {[f'{x:.4f}' for x in eigs_gen]}")
            print(f"        sym eigs:    {[f'{x:.4f}' for x in eigs_sym]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
