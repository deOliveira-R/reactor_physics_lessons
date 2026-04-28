"""Diagnostic: spectrum of the basis-coef multi-bounce operator vs T·R matrix.

Created by numerics-investigator on 2026-04-27.

KEY DERIVATION: In partial-current-moment (J⁺) space, the per-bounce
operator T·R has matrix eigenvalues that grow as O(N²) — but this is
because R's matrix amplifies high-order basis vectors in J⁺ space.

In *basis-coefficient* (c) space, the same physics is operator
K_op = M⁻¹ T_M where T_M_nm = ∫ µ P̃_n P̃_m e^(-τ) dµ. This is the
"per-bounce attenuation in basis-coef space."

Check: in basis-coef space, ρ(K_op) should be ≤ 1 (the multiplication
operator e^(-τ) has spectrum ≤ 1). If it is, then the matrix T·R
eigenvalues are *similarity-equivalent* to K_op's (via the M factor).
If not, then the construction has a basis incompatibility.
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


def build_T_M(sigt, R, N, n_quad=128):
    """T_M_nm = ∫_0^1 µ P̃_n P̃_m e^(-σ·2R·µ) dµ."""
    nodes, wts = leggauss(n_quad)
    mu = 0.5 * (nodes + 1.0)
    w = 0.5 * wts
    decay = np.exp(-sigt * 2.0 * R * mu)
    T_M = np.zeros((N, N))
    for m in range(N):
        Pm = shifted_legendre(m, mu)
        for n in range(N):
            Pn = shifted_legendre(n, mu)
            T_M[m, n] = float(np.sum(w * mu * Pm * Pn * decay))
    return T_M


def build_M(N):
    M = np.zeros((N, N))
    for n in range(N):
        M[n, n] = 1.0 / (2.0 * (2 * n + 1))
        if n + 1 < N:
            off = (n + 1) / (2.0 * (2 * n + 1) * (2 * n + 3))
            M[n, n + 1] = off
            M[n + 1, n] = off
    return M


def build_T_spec_full(sigt, R, N, n_quad=128):
    """T = 2 T_M (the implementation form, J⁺ ↔ J⁺ map)."""
    return 2.0 * build_T_M(sigt, R, N, n_quad)


def test_basis_coef_operator_spectrum(capsys):
    """K_op = M⁻¹ T_M is the per-bounce operator in basis-coef space.

    Should have spectrum bounded by max e^(-τ) over µ ∈ [0,1] = 1.
    Specifically: K_op's eigenvalues should be NEAR the eigenvalues of
    the multiplication operator (function on [0,1]).
    """
    with capsys.disabled():
        sigt = 0.5
        R = 5.0
        print(f"\n=== Basis-coef operator K_op = M⁻¹ T_M, σ_t={sigt}, R={R} ===")
        print("Continuous spectrum: e^(-σ·2R·µ) for µ ∈ [0,1] = "
              f"({np.exp(-sigt*2*R):.4f}, 1].\n")

        for N in (1, 2, 4, 6, 8, 10, 12):
            M = build_M(N)
            T_M = build_T_M(sigt, R, N)
            T_spec = build_T_spec_full(sigt, R, N)
            R_op = reflection_specular(N)

            K_op_basis = np.linalg.solve(M, T_M)
            eigs_basis = sorted(np.abs(np.linalg.eigvals(K_op_basis)),
                                reverse=True)

            TR_mat = T_spec @ R_op
            eigs_TR = sorted(np.abs(np.linalg.eigvals(TR_mat)), reverse=True)

            print(f"  N={N:2d}: K_op (basis-coef) ρ = {eigs_basis[0]:.6f}, "
                  f"all eigs: {[f'{x:.4f}' for x in eigs_basis]}")
            print(f"        T·R (J⁺ moments) eigs:        "
                  f"{[f'{x:.4f}' for x in eigs_TR]}")

            # ASSERTION: K_op_basis should equal T·R as a matrix.
            # The two operators should be IDENTICAL because they both
            # represent the same per-bounce physics — just expressed in
            # different bases.
            #
            # K_op_basis: c_old → c_new where c are basis coefs.
            # T·R: J⁺_old → J⁺_new where J⁺ are partial-current moments.
            #
            # The basis change c → J⁺ is multiplication by 2π·M.
            # So T·R = (2π M) · K_op_basis · (2π M)⁻¹.
            similarity_check = (np.linalg.solve(M, T_spec @ R_op @ M)) / 1.0
            err_sim = np.max(np.abs(similarity_check - K_op_basis))
            print(f"        |M⁻¹·T·R·M - K_op| = {err_sim:.2e} "
                  "(SHOULD be 0 if similarity holds)")


def test_what_does_K_op_basis_eigenvalue_mean(capsys):
    """Check whether the dominant eigenvalue of K_op (basis-coef) is
    bounded by the right physical thing."""
    with capsys.disabled():
        sigt = 0.5
        R = 5.0
        print(f"\n=== What's the right bound on ρ(K_op)? ===")
        print(f"σ_t·R = {sigt*R}, max attenuation = {np.exp(-sigt*2*R):.4f}\n")

        # The "right" rank-1 multi-bounce identity:
        # P_ss = 2 ∫_0^1 µ e^(-σ·2R·µ) dµ — this is BOTH:
        #   (a) the rank-1 multi-bounce eigenvalue
        #   (b) NOT the spectrum of the continuous multiplication operator
        # P_ss is a moment, NOT an eigenvalue of multiplication.
        nodes, wts = leggauss(128)
        mu = 0.5 * (nodes + 1.0)
        w = 0.5 * wts
        P_ss = float(2.0 * np.sum(w * mu * np.exp(-sigt*2*R*mu)))
        print(f"P_ss = 2 ∫ µ e^(-2σRµ) dµ = {P_ss:.6f}")

        # The Hébert factor 1/(1-P_ss) is the rank-1 sum — eigenvalue 1
        # has multiplicity 1 in the rank-1 case.
        hebert_factor = 1.0 / (1.0 - P_ss)
        print(f"1/(1-P_ss) = {hebert_factor:.6f}\n")

        for N in (1, 2, 4, 6, 8, 10):
            M = build_M(N)
            T_M = build_T_M(sigt, R, N)
            K_op_basis = np.linalg.solve(M, T_M)
            eigs = sorted(np.abs(np.linalg.eigvals(K_op_basis)), reverse=True)
            sum_op = np.linalg.solve(np.eye(N) - K_op_basis, np.eye(N))
            sum_op_diag = np.diag(sum_op)
            print(f"  N={N:2d}: ρ(K_op)={eigs[0]:.4f}, "
                  f"diag(I-K_op)⁻¹={[f'{x:.3f}' for x in sum_op_diag]}, "
                  f"trace(I-K_op)⁻¹={np.trace(sum_op):.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
