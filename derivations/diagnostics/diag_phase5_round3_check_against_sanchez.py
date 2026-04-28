"""Round 3 — Check A2 chord-substitution K_bc against Sanchez Eq. (A6).

If they agree, then we have the right formula and "just" need
diagonal regularisation. If they disagree, we have the wrong formula
and the per-pair half-M1 reading isn't correct.
"""
from __future__ import annotations

import numpy as np

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    composite_gl_r,
    compute_K_bc_specular_continuous_mu_sphere,
)

from derivations.diagnostics.diag_phase5_round3_adaptive_quadrature import (
    R_THIN, SIG_T,
    compute_K_bc_chord_substitution_half_M1,
    compute_K_bc_per_pair_half_M1,
)


def test_a2_vs_sanchez(capsys):
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        # Use simple uniform grid (not GL, no panels structure)
        r_nodes = np.linspace(0.5, 4.5, 5)
        r_wts = np.full(5, 1.0)  # dummy weights (we just compare K_bc shape)

        # Sanchez ref impl
        K_sanchez = compute_K_bc_specular_continuous_mu_sphere(
            r_nodes, radii, sig_t_g, n_quad=128,
        )
        # A2 chord-subst (homogeneous half-M1)
        K_a2 = compute_K_bc_chord_substitution_half_M1(
            SPHERE_1D, r_nodes, r_wts, radii, sig_t_g, n_quad=128,
        )
        K_a1 = compute_K_bc_per_pair_half_M1(
            SPHERE_1D, r_nodes, r_wts, radii, sig_t_g, n_quad=128,
        )

        print(f"\n=== A2 vs Sanchez (Eq. A6) ===")
        print(f"K_sanchez:")
        print(K_sanchez)
        print(f"\nK_a2:")
        print(K_a2)
        print(f"\nratio K_a2/K_sanchez:")
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(np.abs(K_sanchez) > 1e-15,
                             K_a2 / K_sanchez, np.nan)
        print(ratio)

        print(f"\nratio K_a2/K_a1:")
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_a = np.where(np.abs(K_a1) > 1e-15,
                               K_a2 / K_a1, np.nan)
        print(ratio_a)
        print(f"  median = {np.nanmedian(ratio_a):.6f}")
        print(f"  std    = {np.nanstd(ratio_a):.6e}")
