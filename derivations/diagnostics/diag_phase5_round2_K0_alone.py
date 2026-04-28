"""Round 2 — Is K^(0) alone Q-divergent?

K^(0) (k=0 in M2) has integrand `G F · 1` with NO multi-bounce
factor. If this alone Q-diverges on the diagonal, the singularity is
in the F·G product (cos(ω) factors), NOT in T(µ).

If K^(0) Q-converges on the diagonal but the SUM Σ_k K^(k) doesn't,
the singularity is summed in from k≥1 terms via the e^{-k·τ}·1/cos²
overlapping issue.
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


def test_K0_alone_q_conv(capsys):
    with capsys.disabled():
        radii = np.array([5.0])
        sig_t_g = np.array([0.5])
        r_nodes, r_wts, _ = composite_gl_r(
            radii, 1, 3, dps=15, inner_radius=0.0,
        )
        N = len(r_nodes)
        print(f"\n=== K^(0) (k=0 only, NO multi-bounce) — Q-conv ===")
        print(f"r_nodes = {r_nodes.tolist()}")
        for K_max in (0, 1, 5, 50):
            print(f"\n  K_max = {K_max}")
            print(f"  i,j   | Q=16     Q=64     Q=128    Q=512   | rel(128 vs 512)")
            for i in range(N):
                for j in range(N):
                    vals = []
                    for Q in (16, 64, 128, 512):
                        K = compute_K_bc_M2_per_pair(
                            SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                            n_quad=Q, K_max=K_max,
                        )
                        vals.append(K[i, j])
                    rel = abs(vals[2] - vals[3]) / max(abs(vals[3]), 1e-30)
                    tag = "DIAG" if i == j else "OFF"
                    print(f"  ({i},{j}) {tag} | "
                          + "  ".join(f"{v:+.3e}" for v in vals)
                          + f" | rel = {rel:.3e}")
