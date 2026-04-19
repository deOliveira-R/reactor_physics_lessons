"""Step-wise verification of the unified Peierls solver against the
mesh-independent reference in :mod:`orpheus.derivations.peierls_reference`.

Layer-by-layer:

1. **Slab K matrix row-sum identity** (this file, first class).
   ``K @ [1,1,...]`` of the ORPHEUS slab kernel matrix equals the
   closed-form uniform-source-pure-absorber flux, analytically:

   .. math::

      \\varphi(x) = \\frac{1}{2\\Sigma_t}[2 - E_2(\\Sigma_t x)
                                          - E_2(\\Sigma_t(L-x))]

   Tests the slab kernel prefactors, the E₁ evaluation, the
   singularity-subtracted diagonal panels, and the Gauss-Legendre
   quadrature weights — all at once. No BC closure enters.

2. **Cylinder K matrix row-sum identity** (next). Built on the
   ``Ki_1`` kernel; analytical row-sum via
   :func:`~orpheus.derivations.peierls_reference.curvilinear_K_vol_element`
   adaptive mpmath.quad reference.

3. **Sphere K matrix row-sum identity** (after cylinder). Same
   methodology with ``exp`` kernel.

4. **BC closure tests** (after Layer 1-3 pass). Test K_bc
   element-by-element vs the ``R = (1/2) B⁻¹`` canonical form.
"""
from __future__ import annotations

import mpmath
import numpy as np
import pytest

from orpheus.derivations.peierls_reference import (
    slab_uniform_source_analytical,
    slab_K_vol_element,
)
from orpheus.derivations import peierls_slab


# ═══════════════════════════════════════════════════════════════════════
# Layer 1 — Slab solver infrastructure verification
# ═══════════════════════════════════════════════════════════════════════

def _build_slab_K_and_nodes(L, sig_t, n_panels=8, p_order=6, dps=30):
    """Helper: use ORPHEUS slab internals to build K matrix + nodes."""
    # Build composite Gauss-Legendre quadrature exactly as the solver does.
    with mpmath.workdps(dps):
        boundaries = [mpmath.mpf(0), mpmath.mpf(L)]
        gl_ref, gl_wt = peierls_slab._gl_nodes_weights(p_order, dps)

        x_all, w_all, panel_bounds = [], [], []
        pw = mpmath.mpf(L) / n_panels
        for pidx in range(n_panels):
            pa = mpmath.mpf(pidx) * pw
            pb = pa + pw
            i_start = len(x_all)
            xs, ws = peierls_slab._map_to_interval(gl_ref, gl_wt, pa, pb)
            x_all.extend(xs)
            w_all.extend(ws)
            i_end = len(x_all)
            panel_bounds.append((pa, pb, i_start, i_end))

        node_panel = []
        for pidx, (pa, pb, i_start, i_end) in enumerate(panel_bounds):
            node_panel.extend([pidx] * (i_end - i_start))
        sig_t_at_node = [[mpmath.mpf(sig_t)] for _ in x_all]

        K_per_group = peierls_slab._build_kernel_matrix(
            x_nodes=x_all, w_nodes=w_all,
            panel_bounds=panel_bounds, node_panel=node_panel,
            sig_t_at_node=sig_t_at_node, boundaries=boundaries,
            sig_t_per_region=[[mpmath.mpf(sig_t)]],
            n_regions=1, ng=1, dps=dps,
        )
    return K_per_group[0], x_all, w_all, panel_bounds


@pytest.mark.l1
@pytest.mark.verifies("peierls-unified")
class TestSlabKernelRowSum:
    """ORPHEUS slab K-matrix row-sum equals closed-form flux from
    uniform source on pure absorber with vacuum BC.

    Identity: :math:`\\sum_j K_{ij} \\cdot 1 \\;=\\;
    \\int_0^L \\tfrac{1}{2}E_1(\\Sigma_t|x_i - x'|)\\,\\mathrm d x'
    \\;=\\; \\tfrac{1}{2\\Sigma_t}[2 - E_2(\\Sigma_t x_i)
                                   - E_2(\\Sigma_t(L-x_i))]`

    Any deviation exposes a bug in: kernel formula, quadrature weights,
    diagonal-panel singularity subtraction, or the E₁ evaluation.
    """

    @pytest.mark.parametrize("L, sig_t", [
        (1.0, 1.0),   # 1 MFP slab
        (5.0, 1.0),   # 5 MFP slab
        (10.0, 0.5), # 5 MFP optical, thicker in real units
    ])
    def test_row_sum_matches_analytical_uniform_source(self, L, sig_t):
        dps = 30
        n_panels = 8
        p_order = 6
        K, x_nodes, w_nodes, panel_bounds = _build_slab_K_and_nodes(
            L, sig_t, n_panels=n_panels, p_order=p_order, dps=dps,
        )

        # Compute K @ [1, 1, ...]: mpmath matrix × column vector
        N = len(x_nodes)
        row_sum_K = [sum(K[i, j] for j in range(N)) for i in range(N)]

        # Analytical reference at each node (at full mpmath dps)
        row_sum_analytical = [
            slab_uniform_source_analytical(float(x_nodes[i]), L, sig_t, dps=dps)
            for i in range(N)
        ]

        # Relative error at each node
        max_rel_err = mpmath.mpf(0)
        for i in range(N):
            ref = row_sum_analytical[i]
            got = row_sum_K[i]
            if abs(ref) > 1e-30:
                rel = abs(got - ref) / abs(ref)
                if rel > max_rel_err:
                    max_rel_err = rel

        # With panels=8, p_order=6, GL quadrature should achieve ~1e-8 or better
        # (p_order=6 gives 2×6-1 = 11th-order polynomial exact → spectral
        # convergence for smooth integrand; the log singularity is handled
        # by product-integration in the diagonal panel).
        assert max_rel_err < 1e-8, (
            f"Slab K-matrix row-sum disagrees with analytical "
            f"uniform-source flux by {float(max_rel_err):.3e} "
            f"(L={L}, Σ_t={sig_t}, panels={n_panels}, p={p_order}, dps={dps}). "
            f"Expected < 1e-8. Localises bug to kernel formula, "
            f"quadrature, or singularity subtraction."
        )

    def test_row_sum_convergence_with_panels(self):
        """Refining n_panels drives the row-sum error to zero.

        With the unified basis-aware adaptive-quadrature assembly (issue
        #113), even at n_panels=2, p=4 the row-sum hits the dps=30 floor
        (~1e-16). Monotonicity is only required above the noise floor.
        """
        L, sig_t = 2.0, 1.0
        errors = []
        for n_panels in (2, 4, 8, 16):
            K, x_nodes, _, _ = _build_slab_K_and_nodes(
                L, sig_t, n_panels=n_panels, p_order=4, dps=30,
            )
            N = len(x_nodes)
            max_err = 0.0
            for i in range(N):
                ref = slab_uniform_source_analytical(float(x_nodes[i]), L, sig_t, dps=30)
                got = sum(K[i, j] for j in range(N))
                if abs(ref) > 1e-30:
                    err = float(abs(got - ref) / abs(ref))
                    max_err = max(max_err, err)
            errors.append(max_err)

        # Monotonicity: enforce only above the mpmath dps=30 noise floor.
        noise_floor = 1e-13
        for k in range(len(errors) - 1):
            if errors[k] < noise_floor:
                continue  # already at floor — permutations of O(1e-16) are noise
            assert errors[k + 1] <= errors[k] * 1.1, (
                f"Row-sum error did not decrease under panel refinement: "
                f"errors = {errors}"
            )
        # Finest mesh reaches < 1e-10 (with the fix, actually near 1e-16)
        assert errors[-1] < 1e-10, (
            f"At n_panels=16, row-sum error = {errors[-1]:.3e}, expected < 1e-10"
        )


@pytest.mark.l1
@pytest.mark.verifies("peierls-unified")
class TestSlabKMatrixElementwiseVsReference:
    """Element-by-element comparison of ORPHEUS slab K[i,j] to the
    adaptive-mpmath reference :func:`slab_K_vol_element` at dps=40.

    Both compute the same integral; agreement to ~1e-10 confirms the
    production code's kernel assembly is free of implementation bugs.
    """

    @pytest.mark.catches("ERR-028")
    def test_small_case_elementwise_agreement(self):
        L, sig_t = 1.0, 1.0
        dps = 40
        n_panels, p_order = 4, 4
        K, x_nodes, _, panel_bounds = _build_slab_K_and_nodes(
            L, sig_t, n_panels=n_panels, p_order=p_order, dps=dps,
        )
        N = len(x_nodes)

        # Element-by-element: test 4 strategically-chosen entries.
        # Off-diagonal first (smooth integrand, easy);
        # diagonal-panel entries test the singularity subtraction.
        test_indices = [(0, 0), (0, N-1), (N//2, N//2), (N-1, 0)]
        for i, j in test_indices:
            ref = slab_K_vol_element(
                i, j, x_nodes, panel_bounds, L, sig_t, dps=dps,
            )
            got = K[i, j]
            abs_diff = abs(got - ref)
            rel_diff = abs_diff / abs(ref) if abs(ref) > 1e-30 else abs_diff
            assert rel_diff < 1e-10, (
                f"K[{i},{j}] disagreement: production = {got}, "
                f"reference (adaptive) = {ref}, rel diff = {float(rel_diff):.3e}"
            )

    @pytest.mark.catches("ERR-027")
    def test_cross_panel_boundary_neighbour_elementwise(self):
        """Regression gate for :issue:`113` — cross-panel K[i,j] entries
        at 1e-10 vs the adaptive reference, specifically around the
        panel-boundary neighbour that used to fail at ~1.4e-2.

        Before the fix: production used ``(1/2) E_1(τ_ij)·w_j``
        (one-point GL collocation) for cross-panel entries; for
        2 panels × p=4 with L=Σ_t=1 the panel-boundary neighbour
        K[4, 3] disagreed with the adaptive reference at 1.4e-2
        because the :math:`E_1` near-log structure at :math:`x'=x_i`
        sits just 0.035 outside the source panel.
        """
        L, sig_t = 1.0, 1.0
        n_panels, p_order, dps = 2, 4, 40
        K, x_nodes, _, panel_bounds = _build_slab_K_and_nodes(
            L, sig_t, n_panels=n_panels, p_order=p_order, dps=dps,
        )

        # Node layout for n_panels=2, p=4: nodes 0..3 in panel 0, 4..7 in panel 1.
        # Node 3 (panel 0) ↔ Node 4 (panel 1) straddle the boundary at x=0.5.
        for i, j in [(4, 3), (3, 4), (3, 3), (4, 4)]:
            ref = slab_K_vol_element(
                i, j, x_nodes, panel_bounds, L, sig_t, dps=dps,
            )
            got = K[i, j]
            rel = float(abs(got - ref) / abs(ref)) if abs(ref) > 1e-30 else 0.0
            assert rel < 1e-10, (
                f"K[{i},{j}] production vs adaptive reference disagree: "
                f"rel_diff = {rel:.3e} (panels=2, p=4). "
                f"Regression of issue #113 cross-panel near-singular bug."
            )

    @pytest.mark.catches("ERR-027")
    @pytest.mark.parametrize("n_panels", [2, 4, 8, 16])
    def test_cross_panel_scaling_gates_at_1e10(self, n_panels):
        """Regression gate for :issue:`113` — the worst adjacent-panel
        K[i,j] must match the adaptive reference at 1e-10 under any
        panel refinement. Before the fix, this error scaled only as
        O(h) / was even non-monotonic.
        """
        L, sig_t = 1.0, 1.0
        p_order, dps = 4, 30
        K, x_nodes, _, panel_bounds = _build_slab_K_and_nodes(
            L, sig_t, n_panels=n_panels, p_order=p_order, dps=dps,
        )
        N = len(x_nodes)
        # node_panel from panel_bounds
        node_panel = []
        for pidx, (_, _, i0, i1) in enumerate(panel_bounds):
            node_panel.extend([pidx] * (i1 - i0))

        max_rel = 0.0
        worst = None
        for i in range(N):
            for j in range(N):
                if node_panel[i] == node_panel[j]:
                    continue
                if abs(node_panel[i] - node_panel[j]) > 1:
                    continue  # only adjacent panels
                ref = slab_K_vol_element(
                    i, j, x_nodes, panel_bounds, L, sig_t, dps=dps,
                )
                got = K[i, j]
                if abs(ref) > 1e-30:
                    rel = float(abs(got - ref) / abs(ref))
                    if rel > max_rel:
                        max_rel, worst = rel, (i, j)
        assert max_rel < 1e-10, (
            f"At n_panels={n_panels}, worst adjacent-panel K entry "
            f"K[{worst}] differs from adaptive ref by {max_rel:.3e}. "
            f"Regression of issue #113."
        )
