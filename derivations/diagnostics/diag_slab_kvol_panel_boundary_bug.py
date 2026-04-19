"""Diagnostic: slab Peierls K-matrix cross-panel near-singular integrand bug.

Created by numerics-investigator on 2026-04-18.

Hypothesis under test
---------------------
``peierls_slab._build_kernel_matrix`` applies its log-singularity
subtraction (via ``_product_log_weights``) ONLY when the observer node
``i`` and the source node ``j`` lie in the same panel. For cross-panel
entries it uses the naive fixed-order Gauss-Legendre formula

    K[i, j]  =  (1/2) * E_1(Sigma_t * |x_i - x_j|) * w_j

This is fine when the observer's optical distance to the source panel
is large (so the integrand is smooth), but FAILS when the observer is
within a small optical distance of the source panel boundary: the
integrand's logarithmic near-singularity at ``x' = x_i`` is then just
outside the source panel, and fixed GL cannot resolve the resulting
near-log behaviour. Error is ~1% at panel-boundary neighbours, shrinks
as O(h) under panel refinement.

Target entry: 2 panels, p_order=4, L=1, Sigma_t=1. Observer node 4 at
x=0.5347 (panel 1), source basis L_3 supported on panel 0. Log
singularity at x'=0.5347 is just 0.035 beyond panel 0's boundary
x=0.5. Production vs reference disagree at ~1.45e-2 relative.

If this diagnostic catches a real bug, promote to
``tests/derivations/test_peierls_slab.py`` (or equivalent).
"""
from __future__ import annotations

import mpmath
import numpy as np
import pytest

from orpheus.derivations import peierls_slab
from orpheus.derivations.peierls_geometry import lagrange_basis_on_panels
from orpheus.derivations.peierls_reference import (
    slab_K_vol_element,
    slab_uniform_source_analytical,
)


def _build_slab_K(L, sig_t, n_panels, p_order, dps):
    """Mirror the production kernel build, returning K plus geometry.

    Duplicates the _build_slab_K_and_nodes helper in
    tests/derivations/test_peierls_reference.py but kept here to stay
    self-contained.
    """
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
    return K_per_group[0], x_all, w_all, panel_bounds, node_panel


@pytest.mark.l1
def test_slab_row_sum_should_converge_spectrally_under_p_refinement():
    """Row-sum error should converge SPECTRALLY, not O(h).

    For a Fredholm equation with a log-type singularity, Nystrom
    with piecewise polynomial basis and proper singularity subtraction
    achieves spectral convergence (super-algebraic in p). Observing
    only O(h) convergence is the fingerprint of a missing subtraction
    step — here the cross-panel near-singular contributions.

    Panels x2, err/2  -> O(h). Panels x2, err/4 -> O(h^2). Panels x2,
    err/10 or better -> spectral (what a correct Nystrom should give).
    """
    L, sig_t = 1.0, 1.0
    p_order = 4
    dps = 30

    errors = []
    for n_panels in (2, 4, 8, 16):
        K, x_nodes, _, _, _ = _build_slab_K(L, sig_t, n_panels, p_order, dps)
        N = len(x_nodes)
        max_err = 0.0
        for i in range(N):
            ref = slab_uniform_source_analytical(
                float(x_nodes[i]), L, sig_t, dps=dps,
            )
            got = sum(K[i, k] for k in range(N))
            if abs(ref) > 1e-30:
                rel = float(abs(got - ref) / abs(ref))
                max_err = max(max_err, rel)
        errors.append(max_err)

    ratios = [errors[k] / errors[k + 1] for k in range(len(errors) - 1)]
    # Spectral would be ratios >= 10 (or better); O(h^2) gives ~4;
    # O(h) gives ~2. At p=4 and smooth integrand the correct method
    # is spectral. We gate at ratio >= 4 (i.e. at least quadratic).
    assert min(ratios) >= 4.0 - 0.2, (
        f"Slab row-sum convergence is sub-quadratic (O(h) bug signature).\n"
        f"  panels = {[2, 4, 8, 16]}\n"
        f"  errors = {[f'{e:.3e}' for e in errors]}\n"
        f"  ratios = {[f'{r:.2f}' for r in ratios]}\n"
        f"Root cause: _build_kernel_matrix uses naive GL for cross-panel\n"
        f"entries (see peierls_slab.py lines ~156-164); when the observer\n"
        f"is within a small optical distance of the source panel the\n"
        f"near-log integrand is not resolved. Fix by extending\n"
        f"_product_log_weights to cross-panel sweeps."
    )


@pytest.mark.l1
def test_slab_K_boundary_neighbour_matches_reference():
    """K[i, j] for adjacent-panel neighbours straddling a panel boundary
    must match the adaptive mpmath reference to 1e-10.

    With 2 panels, p=4, L=1, Sigma_t=1:
      node 3: x=0.4653 (panel 0, last node)
      node 4: x=0.5347 (panel 1, first node)
    Panel boundary at x=0.5. Observer 4 looks at basis L_3 in panel 0;
    log singularity at x'=0.5347 is only 0.035 outside panel 0.

    The production fixed-GL integrand evaluation gives ~1.4% error
    here. The fix (singularity subtraction extended across panels, OR
    adaptive quadrature for near-singular integrand) should drop this
    to below 1e-10.
    """
    L, sig_t = 1.0, 1.0
    n_panels, p_order, dps = 2, 4, 40
    K, x_nodes, _, panel_bounds, _ = _build_slab_K(
        L, sig_t, n_panels, p_order, dps,
    )

    for i, j in [(4, 3), (3, 4), (3, 3), (4, 4)]:
        ref = slab_K_vol_element(
            i, j, x_nodes, panel_bounds, L, sig_t, dps=dps,
        )
        got = K[i, j]
        rel = float(abs(got - ref) / abs(ref)) if abs(ref) > 1e-30 else 0.0
        assert rel < 1e-10, (
            f"K[{i},{j}] production vs adaptive reference disagree: "
            f"rel_diff = {rel:.3e} (panels=2 p=4). "
            f"Smoking gun for the cross-panel near-singular bug "
            f"(see diag_slab_kvol_panel_boundary_bug.py docstring)."
        )


@pytest.mark.l1
def test_slab_cross_panel_bug_isolation_via_direct_integral():
    """Isolate the exact integral that fails.

    Direct test of the cross-panel contribution to K[4, 3]:
      I = int_0^0.5  (1/2) E_1(sig_t * |x_4 - x'|) * L_3(x') dx'

    Three methods:
      A. Production: (1/2) E_1(sig_t |x_4 - x_3|) * w_3  (collocation)
      B. mpmath.quad over [0, 0.5]                       (adaptive)
      C. mpmath.quad over [0, 0.25, 0.4, 0.5]            (subdivided)

    Methods B and C must agree to machine precision (integrand is
    smooth inside [0, 0.5]; near-log behaviour only as x' -> 0.5).
    Method A must disagree with B by ~1.4e-3 -- this is the bug.
    """
    L, sig_t = 1.0, 1.0
    n_panels, p_order, dps = 2, 4, 40
    _K, x_nodes, w_nodes, panel_bounds, _ = _build_slab_K(
        L, sig_t, n_panels, p_order, dps,
    )

    x_i = mpmath.mpf(x_nodes[4])       # observer at 0.535, panel 1
    j = 3                              # source basis node 3 in panel 0
    x_j = mpmath.mpf(x_nodes[j])
    w_j = mpmath.mpf(w_nodes[j])
    x_nodes_f = np.array([float(x) for x in x_nodes])

    def integrand(x_prime):
        x_p = mpmath.mpf(x_prime)
        if x_p == x_i:
            return mpmath.mpf(0)
        tau = sig_t * abs(x_i - x_p)
        kernel = mpmath.expint(1, tau) / 2
        L_vals = lagrange_basis_on_panels(x_nodes_f, panel_bounds, float(x_p))
        return kernel * mpmath.mpf(float(L_vals[j]))

    with mpmath.workdps(dps):
        # A: production collocation value
        prod = mpmath.expint(1, sig_t * abs(x_i - x_j)) * w_j / 2
        # B: adaptive
        adapt = mpmath.quad(integrand, [mpmath.mpf(0), mpmath.mpf(0.5)])
        # C: adaptive with manual subdivision near the near-singularity
        subdiv = mpmath.quad(integrand, [
            mpmath.mpf(0), mpmath.mpf(0.25),
            mpmath.mpf(0.4), mpmath.mpf(0.5),
        ])

    adapt_vs_subdiv = float(abs(adapt - subdiv))
    prod_vs_adapt = float(abs(prod - adapt))

    # B and C must agree
    assert adapt_vs_subdiv < 1e-15, (
        f"Reference integrals disagree: adapt={float(adapt)}, "
        f"subdiv={float(subdiv)}, diff={adapt_vs_subdiv}"
    )
    # A must disagree with B by ~1.4e-3. If this assertion ever holds,
    # the production code has been FIXED and this test should be moved
    # to a regression test that gates correctness (reverse the assert).
    assert prod_vs_adapt > 1e-6, (
        "Production cross-panel collocation is now within 1e-6 of the "
        "adaptive reference, meaning the bug has been fixed. Flip this "
        "assertion and move to tests/derivations/."
    )


@pytest.mark.l1
@pytest.mark.parametrize("n_panels", [2, 4, 8, 16])
def test_slab_cross_panel_error_scales_with_panel_thickness(n_panels):
    """As panels -> infinity the cross-panel near-singular error shrinks
    only at O(h) rate (since one GL point effectively resolves the
    integrand over the whole neighbouring panel). Quantify this scaling.
    """
    L, sig_t = 1.0, 1.0
    p_order, dps = 4, 30
    K, x_nodes, _, panel_bounds, node_panel = _build_slab_K(
        L, sig_t, n_panels, p_order, dps,
    )
    N = len(x_nodes)

    # Find the entry with max error straddling a panel boundary
    max_rel = 0.0
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
                    max_rel = rel

    # Record for printing (pytest -v)
    print(
        f"  panels={n_panels:3d}  max_adjacent_cross_panel_rel_err = "
        f"{max_rel:.3e}"
    )
    # If the fix is in place this should be tiny (<1e-10). For now,
    # we gate that the error exists and is in the known O(h)-ish range.
    # The error scales only weakly with refinement (~1/h^0.5 or so at
    # thin panels), not O(h) as a pure log-singular Nystrom bug would,
    # because the nearest-neighbour geometry is scale-invariant in h.
    # When the bug is fixed, this assertion flips to `< 1e-10`.
    assert max_rel > 1e-4, (
        f"Cross-panel error at n_panels={n_panels} is only {max_rel:.3e} "
        f"— smaller than the bug signature. The fix may be in place; "
        f"flip this to `assert max_rel < 1e-10` and promote to the "
        f"permanent suite."
    )


if __name__ == "__main__":
    # Quick CLI run for iterative investigation
    L, sig_t = 1.0, 1.0
    print("Slab Peierls K-matrix cross-panel bug isolation")
    print("=" * 70)
    for n_panels in (2, 4, 8, 16):
        K, x_nodes, _, panel_bounds, node_panel = _build_slab_K(
            L, sig_t, n_panels, 4, 30,
        )
        N = len(x_nodes)
        # Row-sum
        max_rs = 0.0
        for i in range(N):
            ref = slab_uniform_source_analytical(
                float(x_nodes[i]), L, sig_t, dps=30,
            )
            got = sum(K[i, k] for k in range(N))
            if abs(ref) > 1e-30:
                max_rs = max(max_rs, float(abs(got - ref) / abs(ref)))
        # Worst adjacent-panel K entry
        max_cp = 0.0
        worst = None
        for i in range(N):
            for j in range(N):
                if node_panel[i] == node_panel[j]:
                    continue
                if abs(node_panel[i] - node_panel[j]) > 1:
                    continue
                ref = slab_K_vol_element(
                    i, j, x_nodes, panel_bounds, L, sig_t, dps=30,
                )
                rel = float(abs(K[i, j] - ref) / abs(ref)) if abs(ref) > 1e-30 else 0.0
                if rel > max_cp:
                    max_cp, worst = rel, (i, j)
        print(
            f"  panels={n_panels:3d}  row_sum_max_err={max_rs:.3e}   "
            f"worst cross-panel K[{worst[0]},{worst[1]}] rel_err={max_cp:.3e}"
        )
