r"""L1 equivalence: slab moment-form K matches the legacy E_1 Nyström
and the adaptive polar reference at machine precision.

Three independent paths must agree element-by-element on the slab K
matrix:

1. **Moment-form** (this branch under test) — closed-form
   :math:`E_n` polynomial moments.
2. **Legacy E_1 Nyström** —
   :func:`orpheus.derivations.peierls_slab._basis_kernel_weights`
   using adaptive ``mpmath.quad`` on the
   :math:`\int L_j(x') E_1(\tau)\,dx'` integrand.
3. **Adaptive polar reference** —
   :func:`orpheus.derivations.peierls_reference.slab_polar_K_vol_element`
   using nested ``mpmath.quad`` over (μ, ρ).

All three produce the SAME mathematical quantity (modulo a
:math:`\Sigma_t(r_i)` factor between unified and legacy forms).
Equivalence to ~1e-12 is the gate for retiring the legacy and
polar paths in favour of the moment form.
"""
from __future__ import annotations

import mpmath
import numpy as np
import pytest

from orpheus.derivations import peierls_slab
from orpheus.derivations.peierls_reference import slab_polar_K_vol_element
from orpheus.derivations.peierls_geometry import (
    SLAB_POLAR_1D,
    build_volume_kernel,
    composite_gl_r,
)


# ═══════════════════════════════════════════════════════════════════════
# Helper: reference K (legacy E_1 Nyström, σ_t prefactor folded in for
# unified-form comparison)
# ═══════════════════════════════════════════════════════════════════════

def _legacy_unified_K(
    L: float, sig_t: float, n_panels: int, p_order: int, dps: int = 30,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Return the legacy :math:`E_1` Nyström K matrix scaled by
    :math:`\\Sigma_t(r_i)` (i.e. the unified-form K) for a homogeneous
    slab of length ``L``."""
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
            x_all, w_all, panel_bounds, node_panel,
            sig_t_at_node, boundaries, [[mpmath.mpf(sig_t)]],
            n_regions=1, ng=1, dps=dps,
        )
        K_legacy = np.array(
            [[float(K_per_group[0][i, j]) for j in range(len(x_all))]
             for i in range(len(x_all))]
        )

    # Unified form folds in σ_t(r_i) on the LHS:
    #   σ_t · φ_i = Σ_j K^{unified}_{ij} q_j  vs  φ_i = Σ_j K^{legacy}_{ij} q_j
    K_unified = sig_t * K_legacy

    x_arr = np.array([float(xi) for xi in x_all])
    return K_unified, x_arr, panel_bounds


# ═══════════════════════════════════════════════════════════════════════
# Test cases — single homogeneous slab
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
@pytest.mark.verifies("peierls-equation")
@pytest.mark.parametrize(
    "L, sig_t, n_panels, p_order",
    [
        (1.0, 1.0, 2, 4),   # baseline
        (1.0, 1.0, 4, 6),   # finer
        (2.0, 0.5, 3, 4),   # different dimensionalities
        (1.0, 5.0, 2, 4),   # optically thick
        (0.5, 2.0, 4, 4),   # short slab
    ],
)
def test_slab_moments_match_legacy_E1(
    L: float, sig_t: float, n_panels: int, p_order: int,
) -> None:
    """Moment-form K equals the legacy E_1 Nyström K to 1e-12."""
    # Reference: legacy E_1 Nyström (with σ_t prefactor → unified form).
    K_ref, x_ref, _ = _legacy_unified_K(L, sig_t, n_panels, p_order, dps=30)

    # Moment-form K via build_volume_kernel.
    radii = np.array([L])
    sig_t_arr = np.array([sig_t])
    r_nodes, _, panel_bounds = composite_gl_r(
        radii, n_panels_per_region=n_panels, p_order=p_order, dps=30,
    )

    # Sanity: same node distribution as legacy (composite GL uses same
    # underlying mpmath gauss_quadrature → modulo dps round-tripping).
    np.testing.assert_allclose(r_nodes, x_ref, atol=1e-12)

    K_moments = build_volume_kernel(
        SLAB_POLAR_1D, r_nodes, panel_bounds, radii, sig_t_arr,
        n_angular=0, n_rho=0, dps=30,
    )

    # Element-wise relative tolerance, with absolute floor for tiny entries.
    rel = np.abs(K_moments - K_ref) / np.maximum(np.abs(K_ref), 1e-30)
    assert np.max(rel) < 1e-12, (
        f"max rel err = {np.max(rel):.3e} at index {np.unravel_index(np.argmax(rel), rel.shape)}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Test cases — heterogeneous (two-region) slab
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
@pytest.mark.verifies("peierls-equation")
@pytest.mark.parametrize(
    "thicknesses, sig_t_regions, n_panels_per_region, p_order",
    [
        ([1.0, 1.0], [1.0, 0.5], 2, 4),
        ([0.5, 1.5], [2.0, 0.3], 3, 4),
        ([1.0, 0.5], [0.8, 4.0], 4, 4),
    ],
)
def test_slab_moments_heterogeneous_match_legacy(
    thicknesses: list[float],
    sig_t_regions: list[float],
    n_panels_per_region: int,
    p_order: int,
) -> None:
    """Moment-form K matches legacy E_1 Nyström for heterogeneous slabs."""
    n_regions = len(thicknesses)
    L_total = sum(thicknesses)
    boundaries_cum = np.cumsum([0.0] + thicknesses)

    with mpmath.workdps(30):
        boundaries_mp = [mpmath.mpf(b) for b in boundaries_cum]
        gl_ref, gl_wt = peierls_slab._gl_nodes_weights(p_order, 30)

        x_all, w_all, panel_bounds = [], [], []
        node_panel = []
        sig_t_at_node = []
        for r in range(n_regions):
            pw = (boundaries_mp[r + 1] - boundaries_mp[r]) / n_panels_per_region
            for pidx in range(n_panels_per_region):
                pa = boundaries_mp[r] + pidx * pw
                pb = pa + pw
                i_start = len(x_all)
                xs, ws = peierls_slab._map_to_interval(gl_ref, gl_wt, pa, pb)
                x_all.extend(xs)
                w_all.extend(ws)
                i_end = len(x_all)
                panel_bounds.append((pa, pb, i_start, i_end))
                node_panel.extend([len(panel_bounds) - 1] * (i_end - i_start))
                for _ in range(i_end - i_start):
                    sig_t_at_node.append([mpmath.mpf(sig_t_regions[r])])

        sig_t_per_region = [[mpmath.mpf(sig_t_regions[r])] for r in range(n_regions)]
        K_per_group = peierls_slab._build_kernel_matrix(
            x_all, w_all, panel_bounds, node_panel,
            sig_t_at_node, boundaries_mp, sig_t_per_region,
            n_regions=n_regions, ng=1, dps=30,
        )
        K_legacy = np.array(
            [[float(K_per_group[0][i, j]) for j in range(len(x_all))]
             for i in range(len(x_all))]
        )

    x_ref = np.array([float(xi) for xi in x_all])
    sig_t_at_node_f = np.array([float(s[0]) for s in sig_t_at_node])
    K_legacy_unified = (sig_t_at_node_f[:, None] * K_legacy)  # σ_t,i prefactor

    # Build via moment form
    radii = boundaries_cum[1:]
    sig_t_arr = np.array(sig_t_regions, dtype=float)
    r_nodes, _, panel_bounds_geom = composite_gl_r(
        radii, n_panels_per_region=n_panels_per_region, p_order=p_order, dps=30,
    )

    np.testing.assert_allclose(r_nodes, x_ref, atol=1e-12)

    K_moments = build_volume_kernel(
        SLAB_POLAR_1D, r_nodes, panel_bounds_geom, radii, sig_t_arr,
        n_angular=0, n_rho=0, dps=30,
    )

    rel = np.abs(K_moments - K_legacy_unified) / np.maximum(np.abs(K_legacy_unified), 1e-30)
    assert np.max(rel) < 1e-12, (
        f"max rel err = {np.max(rel):.3e} at index {np.unravel_index(np.argmax(rel), rel.shape)}; "
        f"thicknesses={thicknesses}, sig_t={sig_t_regions}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Element-level: moment-form K[i,j] matches the adaptive polar reference
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
@pytest.mark.verifies("peierls-equation")
def test_slab_moments_element_matches_polar_reference() -> None:
    """Element-wise: moment-form K[i,j] matches adaptive polar
    reference :func:`slab_polar_K_vol_element` at 1e-10."""
    L = 1.0
    sig_t = 1.0
    n_panels = 2
    p_order = 4

    radii = np.array([L])
    sig_t_arr = np.array([sig_t])
    r_nodes, _, panel_bounds = composite_gl_r(
        radii, n_panels_per_region=n_panels, p_order=p_order, dps=30,
    )

    K_moments = build_volume_kernel(
        SLAB_POLAR_1D, r_nodes, panel_bounds, radii, sig_t_arr,
        n_angular=0, n_rho=0, dps=30,
    )

    N = len(r_nodes)
    # Spot-check a handful of (i, j) entries against the adaptive reference.
    for i, j in [(0, 0), (0, N - 1), (N // 2, N // 2), (N - 1, 0), (1, 3)]:
        K_ref = float(slab_polar_K_vol_element(
            i, j, r_nodes, panel_bounds, L, sig_t, dps=30,
        ))
        rel = abs(K_moments[i, j] - K_ref) / max(abs(K_ref), 1e-30)
        assert rel < 1e-10, (
            f"[{i},{j}]: K_mom={K_moments[i,j]:.10e}, K_ref={K_ref:.10e}, rel={rel:.3e}"
        )
