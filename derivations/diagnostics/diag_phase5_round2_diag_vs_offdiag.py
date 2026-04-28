"""Round 2 — diagonal vs off-diagonal Q-convergence diagnostic.

Hypothesis: M2 K_bc has the SAME 1/µ² diagonal singularity as the
Sanchez form (Front A finding). At every K_max, the diagonal entries
`K_bc[i, i]` diverge linearly with Q while off-diagonal entries
converge spectrally. The structure is independent of M2 vs Sanchez:
both inherit the surface-diagonal 1/µ² when `r_i = r_j`.

This is the ROOT CAUSE of all 4 round-1 fronts failing AND M2
failing identically.

Test: for every (i,j) on a small 4-node grid, sweep Q and tabulate
‖K[i,j]‖ vs Q. Diagonal vs off-diagonal.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    composite_gl_r,
)

from derivations.diagnostics.diag_phase5_round2_per_pair_quadrature import (
    compute_K_bc_M2_per_pair,
)


def test_diag_vs_offdiag(capsys):
    with capsys.disabled():
        radii = np.array([5.0])
        sig_t_g = np.array([0.5])
        # Tiny 3-panel grid so we can read the matrix
        r_nodes, r_wts, _ = composite_gl_r(
            radii, 1, 3, dps=15, inner_radius=0.0,
        )
        N = len(r_nodes)
        print(f"\n=== Diagonal vs off-diagonal entries — Q-convergence ===")
        print(f"r_nodes = {r_nodes.tolist()} (N={N})")
        # Reference at Q=512
        K_ref = compute_K_bc_M2_per_pair(
            SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
            n_quad=512, K_max=10,
        )
        print(f"Reference K[i,j] at Q=512:")
        for i in range(N):
            row = ""
            for j in range(N):
                row += f"{K_ref[i,j]:11.4e} "
            print(f"  i={i}: {row}")

        # Now sweep Q and compute relative diff to Q=512 PER CELL
        print(f"\n  i,j   | Q=16     Q=32     Q=64    Q=128    Q=256    Q=512   | rel_diff")
        for i in range(N):
            for j in range(N):
                vals = []
                for Q in (16, 32, 64, 128, 256, 512):
                    K_q = compute_K_bc_M2_per_pair(
                        SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                        n_quad=Q, K_max=10,
                    )
                    vals.append(K_q[i, j])
                rel = abs(vals[3] - vals[5]) / max(abs(vals[5]), 1e-30)
                tag = "DIAG" if i == j else "OFF"
                print(f"  ({i},{j}) {tag} | "
                      + "  ".join(f"{v:+.3e}" for v in vals)
                      + f" | rel(128 vs 512) = {rel:.3e}")
