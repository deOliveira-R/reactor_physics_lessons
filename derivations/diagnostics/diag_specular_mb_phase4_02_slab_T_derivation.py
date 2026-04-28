"""Diagnostic: Slab T matrix derivation + rank-1 identity check.

Created by numerics-investigator on 2026-04-28.

DERIVATION
----------
Slab specular has TWO faces (outer at x=L, inner at x=0), each with N
partial-current modes — total mode space is ℝ^(2N), grouped by face.
Specular reflection R is block-diagonal:

    R_slab = [[R_face, 0      ],
              [0,      R_face ]]   with R_face = (1/2) M^{-1} .

A ray leaving one face heading inward at cos angle µ ∈ [0,1] to that
face's inward normal travels chord L/µ through the slab and arrives
AT THE OTHER FACE with the same µ (this time cos angle µ to that
face's outward normal). Single-transit slab T is therefore PURELY
OFF-DIAGONAL in the face-block decomposition:

    T_slab^(1-trans) = [[0,    T_oi],
                        [T_io, 0  ]]

where the face-to-face transit kernel for homogeneous slab is

    T_oi^(mn) = T_io^(mn) = 2 ∫_0^1 µ P̃_m(µ) P̃_n(µ) e^{-Σ_t L / µ} dµ.

For multi-region the chord crosses regions piecewise: τ(µ) = (1/µ) ·
Σ_k Σ_{t,k} ℓ_k where ℓ_k is the slab segment in region k.

There is NO self-block (T_oo = T_ii = 0) because a single transit at
constant direction cannot leave outer face and return to outer face
without first reflecting at inner face. Self-coupling appears only at
ORDER ≥ 2 in the geometric series (T·R)^k, k ≥ 2.

RANK-1 CHECK
------------
At m=n=0, T_oi^(0,0) reduces to a closed-form exponential integral.
Substitute u = 1/µ:
    T_oi = 2 ∫_0^1 µ e^{-σL/µ} dµ = 2 ∫_1^∞ e^{-σL u} / u^3 du = 2 E_3(σL).

This is the slab "P_ss" analog. Check: 2 E_3(0) = 2 · (1/2) = 1 (full
free-streaming transmission) ✓.

For multi-region: T_oi^(0,0) = 2 ∫_0^1 µ e^{-(Σ τ_k)/µ} dµ = 2 E_3(τ_total)
since τ depends only on µ as a single (1/µ) factor against the total
optical thickness τ_total = Σ_k Σ_{t,k} L_k.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    _shifted_legendre_eval,
    _slab_E_n,
)


def build_T_specular_slab(
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_modes: int,
    *,
    n_quad: int = 64,
) -> np.ndarray:
    """Reference slab specular T matrix.

    Returns a (2N, 2N) block matrix. Block ordering: [outer modes,
    inner modes]. Self-blocks are exactly zero by single-transit
    construction.

    Parameters
    ----------
    radii : (R_regions,) outer x-coordinates of each slab region; the
            last entry is L = total slab thickness.
    sig_t : (R_regions,) per-region total cross section.
    n_modes : N (per-face mode count).
    n_quad : Gauss–Legendre nodes on µ ∈ [0,1].

    The off-diagonal block T_oi has

        T_oi[m, n] = 2 ∫_0^1 µ P̃_m(µ) P̃_n(µ) e^{-τ_total / µ} dµ

    with τ_total = Σ_k Σ_{t,k} (radii[k] - radii[k-1])  and radii[-1] = L.
    """
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    L = float(radii[-1])
    region_lengths = np.diff(np.concatenate([[0.0], radii]))
    tau_total = float(np.sum(sig_t * region_lengths))

    # Half-range Gauss-Legendre on µ ∈ [0,1].
    nodes, wts = np.polynomial.legendre.leggauss(n_quad)
    mu = 0.5 * (nodes + 1.0)
    w = 0.5 * wts

    # Multi-region: τ(µ) = τ_total / µ for slab (chord factor 1/µ
    # multiplies the SUM of σ·L_k since slab geometry has no shell
    # crossings).
    decay = np.exp(-tau_total / mu)
    mu_w = w * mu

    T_oi = np.zeros((n_modes, n_modes))
    for m in range(n_modes):
        Pm = _shifted_legendre_eval(m, mu)
        for n in range(n_modes):
            Pn = _shifted_legendre_eval(n, mu)
            T_oi[m, n] = 2.0 * float(np.sum(mu_w * Pm * Pn * decay))

    T = np.zeros((2 * n_modes, 2 * n_modes))
    T[:n_modes, n_modes:] = T_oi
    T[n_modes:, :n_modes] = T_oi  # T_io = T_oi for slab symmetry
    return T


def test_slab_T_rank1_equals_2E3(capsys):
    """T_oi^(0,0) at rank-1 should equal 2 E_3(τ_total) by closed-form."""
    with capsys.disabled():
        cases = [
            ("thin",      np.array([5.0]), np.array([0.5])),  # τ_L=2.5
            ("thick",     np.array([5.0]), np.array([1.0])),  # τ_L=5
            ("very-thin", np.array([5.0]), np.array([0.2])),  # τ_L=1
            ("MR",        np.array([2.0, 5.0]), np.array([0.6, 0.4])),
        ]
        print(f"\n{'case':<10} {'T_oi[0,0]':<15} {'2*E_3(τ_tot)':<15} {'rel_err':<10}")
        for name, radii, sig_t in cases:
            T = build_T_specular_slab(radii, sig_t, 1, n_quad=128)
            L_lengths = np.diff(np.concatenate([[0.0], radii]))
            tau_tot = float(np.sum(sig_t * L_lengths))
            twoE3 = 2.0 * _slab_E_n(3, tau_tot)
            T_oi_00 = T[0, 1]  # off-diagonal block, [outer mode 0, inner mode 0]
            rel = abs(T_oi_00 - twoE3) / twoE3
            print(f"{name:<10} {T_oi_00:<15.10f} {twoE3:<15.10f} {rel:.2e}")
            assert rel < 1e-10, (
                f"T_oi^(0,0) != 2 E_3(τ) for {name}: rel_err={rel:.3e}"
            )


def test_slab_T_self_blocks_zero(capsys):
    """The diagonal (self-face) blocks must be EXACTLY zero — single-
    transit slab cannot leave a face and return without an
    intermediate reflection."""
    with capsys.disabled():
        radii = np.array([5.0])
        sig_t = np.array([0.5])
        for N in (1, 2, 4, 8):
            T = build_T_specular_slab(radii, sig_t, N, n_quad=128)
            T_oo = T[:N, :N]
            T_ii = T[N:, N:]
            T_oi = T[:N, N:]
            T_io = T[N:, :N]
            err_oi_io = float(np.max(np.abs(T_oi - T_io)))
            print(f"N={N}: ‖T_oo‖={np.max(np.abs(T_oo)):.2e}, "
                  f"‖T_ii‖={np.max(np.abs(T_ii)):.2e}, "
                  f"‖T_oi-T_io‖={err_oi_io:.2e}")
            assert np.all(T_oo == 0.0)
            assert np.all(T_ii == 0.0)
            # For homogeneous slab: T_oi == T_io (face symmetry).
            assert err_oi_io < 1e-15


def test_slab_T_oi_symmetric(capsys):
    """T_oi must itself be symmetric in mode indices (m, n) by
    integral symmetry."""
    with capsys.disabled():
        radii = np.array([5.0])
        sig_t = np.array([0.5])
        for N in (2, 4, 8):
            T = build_T_specular_slab(radii, sig_t, N, n_quad=128)
            T_oi = T[:N, N:]
            sym = float(np.max(np.abs(T_oi - T_oi.T)))
            print(f"N={N}: ‖T_oi - T_oi.T‖_max={sym:.2e}")
            assert sym < 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
