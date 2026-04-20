r"""Archived: slab K matrix assembly via the **moment-form Nyström**
architecture (closed-form polynomial moments).

Extracted verbatim from `orpheus/derivations/peierls_geometry.py`
(commit before 2026-04-19 archival). Originally implemented during
the Peierls unification close-out (branch
``investigate/peierls-solver-bugs``); archived because the
verification-side machinery uses adaptive ``mpmath.quad`` per element
(``K_vol_element_adaptive``) and does not need a fast slab K assembly.
The moment form is the *production* path for a future higher-order
discrete CP solver.

See `derivations/archive/README.md` and **GitHub Issue #117** for the
full architecture, performance characteristics, conditioning caveats,
and trigger conditions for re-introducing this code into production.

This module is **not** part of the active import path. To use it:

.. code-block:: python

   import sys
   sys.path.insert(0, "derivations/archive")
   from peierls_slab_moments_assembly import _build_volume_kernel_slab_moments

   from peierls_moments import slab_segment_moments_mp  # also archived

Dependencies (all archived alongside in ``derivations/archive/``):

- ``peierls_moments`` — closed-form moment recursions for E_n, Ki_n, exp.

Live dependencies (still in ``orpheus.derivations``):

- ``CurvilinearGeometry`` — geometry abstraction with
  ``which_annulus`` method.
"""

from __future__ import annotations

import mpmath
import numpy as np

# Archived peer module
from peierls_moments import slab_segment_moments_mp as _slab_segment_moments_mp


def _build_volume_kernel_slab_moments(
    geometry,
    r_nodes: np.ndarray,
    panel_bounds: list[tuple[float, float, int, int]],
    radii: np.ndarray,
    sig_t: np.ndarray,
    dps: int = 30,
) -> np.ndarray:
    r"""Slab K matrix via the **moment-form Nyström** architecture —
    zero inner quadrature.

    For each (observer ``i``, source panel) pair, the contribution is
    obtained in closed form:

    .. math::

       K[i, j_a] \;\mathrel{+}=\;
         \Sigma_t(r_i)\,C_d \cdot \frac{1}{\Sigma_{t,p}}
         \sum_{m=0}^{p-1} c_{a,m}^{(i,p)} \big[J_m^{E_1}(u_b) - J_m^{E_1}(u_a)\big]

    where :math:`C_d = 1/2` is the geometry prefactor,
    :math:`\Sigma_{t,p}` is the source-panel total XS,
    :math:`[u_a, u_b]` is the optical-depth range across the source-panel
    integration sub-interval, :math:`c_{a,m}^{(i,p)}` are the monomial
    coefficients of the cardinal Lagrange basis :math:`L_a` re-expressed
    in the local optical-depth coordinate.

    Self-panel (observer inside source panel) splits the integration
    at :math:`x' = x_i` so the integrand stays C^∞ on each half — no
    log-singularity to subtract.

    Heterogeneous slabs are handled by walking material regions to
    compute the offset :math:`u_a` for cross-panel observers.

    See GitHub Issue #117 for the full mathematical context, the
    Vandermonde conditioning analysis, and the trigger conditions
    for re-introducing this code into a higher-order production CP.
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    panel_bounds = [
        (float(pa), float(pb), int(i_start), int(i_end))
        for (pa, pb, i_start, i_end) in panel_bounds
    ]
    N = len(r_nodes)
    L = float(radii[-1])
    pref = geometry.prefactor

    region_edges = np.concatenate(([0.0], radii.astype(float)))
    cum_tau_at_edges = np.zeros(len(region_edges))
    for k in range(len(radii)):
        cum_tau_at_edges[k + 1] = (
            cum_tau_at_edges[k] + float(sig_t[k]) * (region_edges[k + 1] - region_edges[k])
        )

    def cumulative_tau_at(x: float) -> float:
        x_clamped = min(max(x, 0.0), L)
        k = geometry.which_annulus(x_clamped, radii)
        return cum_tau_at_edges[k] + float(sig_t[k]) * (x_clamped - region_edges[k])

    K = np.zeros((N, N))

    for s_pidx, (pa_s, pb_s, j_start, j_end) in enumerate(panel_bounds):
        x_l = float(pa_s)
        x_r = float(pb_s)
        x_mid = 0.5 * (x_l + x_r)
        k_panel = geometry.which_annulus(x_mid, radii)
        sig_t_panel = float(sig_t[k_panel])
        if sig_t_panel <= 0.0:
            continue

        panel_nodes = r_nodes[j_start:j_end]
        p = j_end - j_start

        for i in range(N):
            x_i = float(r_nodes[i])
            sig_t_i = float(sig_t[geometry.which_annulus(x_i, radii)])

            pieces: list[tuple[float, float, float, float]] = []
            if x_i < x_l:
                tau_xi = cumulative_tau_at(x_i)
                tau_xl = cumulative_tau_at(x_l)
                pieces.append((x_l, x_r, +1.0, tau_xl - tau_xi))
            elif x_i > x_r:
                tau_xi = cumulative_tau_at(x_i)
                tau_xr = cumulative_tau_at(x_r)
                pieces.append((x_l, x_r, -1.0, tau_xi - tau_xr))
            else:
                if x_r - x_i > 1e-15:
                    pieces.append((x_i, x_r, +1.0, 0.0))
                if x_i - x_l > 1e-15:
                    pieces.append((x_l, x_i, -1.0, 0.0))

            for (x_a_seg, x_b_seg, sign, delta) in pieces:
                seg_width = x_b_seg - x_a_seg
                if seg_width <= 1e-15:
                    continue

                u_lo = delta
                u_hi = delta + sig_t_panel * seg_width

                if sign > 0:
                    u_at_nodes = delta + sig_t_panel * (panel_nodes - x_a_seg)
                else:
                    u_at_nodes = delta + sig_t_panel * (x_b_seg - panel_nodes)

                M_mp = _slab_segment_moments_mp(u_lo, u_hi, p - 1, dps)

                with mpmath.workdps(dps):
                    V_mp = mpmath.matrix(p, p)
                    for k in range(p):
                        u_k_mp = mpmath.mpf(float(u_at_nodes[k]))
                        u_k_pow = mpmath.mpf(1)
                        for m in range(p):
                            V_mp[k, m] = u_k_pow
                            u_k_pow *= u_k_mp
                    M_col = mpmath.matrix(p, 1)
                    for m in range(p):
                        M_col[m, 0] = M_mp[m]
                    w_mp = mpmath.lu_solve(V_mp.T, M_col)
                    w = np.array([float(w_mp[a, 0]) for a in range(p)])

                K[i, j_start:j_end] += (pref * sig_t_i / sig_t_panel) * w

    return K
