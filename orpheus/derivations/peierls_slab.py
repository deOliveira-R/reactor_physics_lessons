r"""Peierls integral equation reference for slab CP verification.

Solves the 1-D slab Peierls (integral transport) equation via Nyström
quadrature at mpmath precision, producing
:class:`ContinuousReferenceSolution` objects whose operator form is
``"integral-peierls"``.

The Peierls equation for a homogeneous or piecewise-constant slab
:math:`[0, L]` reads:

.. math::

   \Sigma_t(x)\,\varphi(x)
   = \tfrac12 \int_0^L E_1\!\bigl(\tau(x,x')\bigr)\,q(x')\,\mathrm{d}x'
     + S_{\rm bc}(x)

where :math:`q = \Sigma_s\varphi + \frac{\chi\,\nu\Sigma_f}{k}\varphi`
is the total isotropic source and :math:`S_{\rm bc}` accounts for
boundary re-entry (white BC).

The :math:`E_1` kernel has a logarithmic singularity at :math:`x=x'`:

.. math::

   E_1(z) = \bigl[-\ln z - \gamma\bigr] + R(z),
   \qquad R(z)\equiv E_1(z)+\ln z+\gamma,\quad R(0)=0.

The Nyström method uses **singularity subtraction**: standard GL weights
for the smooth remainder *R*, and **product-integration weights**
(computed via mpmath.quad) for the :math:`-\ln|x-x'|` part.

This module is the Phase-4.1 deliverable of the verification campaign.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath
import numpy as np

from ._kernels import e_n_mp
from ._quadrature import adaptive_mpmath
from ._reference import (
    ContinuousReferenceSolution,
    ProblemSpec,
    Provenance,
)
from ._xs_library import LAYOUTS, get_mixture, get_xs
from .peierls_geometry import (
    gl_nodes_weights,
    lagrange_basis_on_panels,
    map_gl_to,
)

# Topological class — slab has two parallel boundary faces at x=0 and
# x=L, so it belongs to the same F.4-applicable topology class as
# hollow cylinder and hollow sphere. See
# :ref:`theory-peierls-capabilities` (Class A) and
# :file:`.claude/plans/topology-based-consolidation.md`. The native
# E_1 Nyström machinery in this module is distinct from the curvilinear
# CurvilinearGeometry machinery (slab's log singularity does not factor
# through the polar reformulation); a future session will unify via
# arbitrary-precision quadrature on the polar form. The TOPOLOGY label
# below is what the case-builder uses to register slab alongside hollow
# cyl / hollow sph, even while the implementation stays separate.
TOPOLOGY: str = "two_surface"


# ═══════════════════════════════════════════════════════════════════════
# Product-integration for the log singularity
# ═══════════════════════════════════════════════════════════════════════

def _product_log_weights(panel_a, panel_b, x_eval, nodes, dps):
    r"""Modified weights for :math:`\int_a^b f(x')(-\ln|x_i-x'|)\,dx'`.

    Computes one weight per node by exactly integrating each Lagrange
    basis polynomial against the :math:`-\ln|\cdot|` weight via
    mpmath.quad.
    """
    p = len(nodes)
    weights = []
    with mpmath.workdps(dps + 10):
        for j in range(p):
            # Build Lagrange basis L_j as a closure
            def _basis(x, _j=j):
                val = mpmath.mpf(1)
                for m in range(p):
                    if m == _j:
                        continue
                    val *= (x - nodes[m]) / (nodes[_j] - nodes[m])
                return val

            def integrand(x):
                d = abs(x_eval - x)
                if d == 0:
                    return mpmath.mpf(0)
                return _basis(x) * (-mpmath.log(d))

            # Q6.L1: route through the unified adaptive contract while
            # preserving the original behavior of letting mpmath.quad
            # auto-detect the log singularity (no breakpoint hint).
            q = adaptive_mpmath(
                float(panel_a), float(panel_b), dps=dps + 10,
            )
            weights.append(mpmath.mpf(q.integrate(integrand)))
    return weights


def _basis_kernel_weights(
    panel_a, panel_b, x_eval, nodes, optical_path_fn, dps,
):
    r"""Basis-aware weights for the half-:math:`E_1` Peierls integral.

    Returns

    .. math::

       w_j = \int_{p_a}^{p_b} E_1\!\bigl(\tau(x_i,x')\bigr)\,L_j(x')\,
              \mathrm d x'

    for each Lagrange basis node ``j`` supported on the source panel
    :math:`[p_a, p_b]`. The observer :math:`x_{\rm eval}` may lie
    inside or outside the source panel:

    * **Outside** (cross-panel): integrand is smooth but non-polynomial;
      naive fixed-order GL collocation (``E_1(τ_ij)·w_j``) gives ~1%
      error at p=4. Adaptive :func:`mpmath.quad` on :math:`[p_a, p_b]`
      is accurate to machine :math:`\mathrm{dps}`.

    * **Inside** (same-panel): :math:`E_1` has an integrable log
      singularity at :math:`x'=x_{\rm eval}` AND the smooth remainder
      :math:`R(\tau) = E_1(\tau)+\ln\tau+\gamma` has a derivative
      discontinuity there. Providing :math:`[p_a, x_{\rm eval}, p_b]`
      as subdivision hint to :func:`mpmath.quad` lets the adaptive
      rule resolve both the kink and the log in one shot — replaces
      the older singularity-subtraction scheme with a unified
      implementation that exactly mirrors the adaptive reference in
      :func:`peierls_reference.slab_K_vol_element`.
    """
    p = len(nodes)
    weights = []
    inside = panel_a <= x_eval <= panel_b
    with mpmath.workdps(dps + 10):
        for j in range(p):
            def _basis(x, _j=j):
                val = mpmath.mpf(1)
                for m in range(p):
                    if m == _j:
                        continue
                    val *= (x - nodes[m]) / (nodes[_j] - nodes[m])
                return val

            def integrand(x):
                tau = optical_path_fn(x)
                if tau <= 0:
                    # Integrable log singularity at x = x_eval; mpmath.quad
                    # with the subdivision hint below avoids endpoint evals.
                    return mpmath.mpf(0)
                return _basis(x) * e_n_mp(1, tau, dps)

            # Q6.L1: route through the unified adaptive contract.  The
            # x_eval subdivision hint (when *strictly* inside the
            # panel) lets the adaptive engine resolve both the E_1 log
            # singularity at the observer point AND the C¹ kink in the
            # smooth remainder R(τ) = E_1(τ)+ln τ+γ.  When x_eval
            # coincides with a panel endpoint there is no interior
            # singularity to hint at — adaptive integration on
            # [panel_a, panel_b] is already correct.
            interior_kink = panel_a < x_eval < panel_b
            q = adaptive_mpmath(
                float(panel_a), float(panel_b),
                breakpoints=(float(x_eval),) if interior_kink else (),
                dps=dps + 10,
            )
            weights.append(mpmath.mpf(q.integrate(integrand)))
    return weights


# ═══════════════════════════════════════════════════════════════════════
# Nyström kernel builder
# ═══════════════════════════════════════════════════════════════════════

def _build_kernel_matrix(
    x_nodes: list,
    w_nodes: list,
    panel_bounds: list[tuple],
    node_panel: list[int],
    sig_t_at_node: list,
    boundaries: list,
    sig_t_per_region: list[list],
    n_regions: int,
    ng: int,
    dps: int,
) -> object:
    r"""Build the half-E₁ kernel matrix K[i,j,g] for cell-interior transport.

    Returns an mpmath matrix of shape ``(N, N)`` for **each group** g,
    stored as a list of mpmath matrices.  The caller multiplies by XS
    to assemble scatter/fission kernels.

    ``K_g[i,j] = (1/2) E_1(τ(x_i, x_j, g)) * w_j`` with
    product-integration on the diagonal panel.
    """
    N = len(x_nodes)

    def optical_path(xi, xj, g):
        if xi == xj:
            return mpmath.mpf(0)
        a, b = (min(xi, xj), max(xi, xj))
        tau = mpmath.mpf(0)
        for rr in range(n_regions):
            ra, rb = boundaries[rr], boundaries[rr + 1]
            oa, ob = max(a, ra), min(b, rb)
            if oa < ob:
                tau += sig_t_per_region[rr][g] * (ob - oa)
        return tau

    K_per_group: list[object] = []

    # Unified basis-aware Nyström assembly. Every K[i, j] is
    #     (1/2) ∫_{source panel} E_1(τ(x_i, x')) L_j(x') dx'
    # evaluated with adaptive :func:`mpmath.quad`. This mirrors the
    # adaptive reference in :func:`peierls_reference.slab_K_vol_element`
    # and replaces two previously-buggy fixed-GL code paths:
    #
    # * **Cross-panel** used ``E_1(τ_ij)·w_j`` (one-point collocation)
    #   which fails at ~1% for p=4 on any pair with modest optical
    #   distance, worst at panel-boundary neighbours.
    # * **Same-panel** used singularity subtraction with GL collocation
    #   of the smooth remainder R(τ) = E_1(τ)+ln τ+γ — but R has a
    #   derivative kink at x'=x_i that GL cannot resolve, producing
    #   ~1% error on the diagonal panel too.
    #
    # See issues #113 and diag_slab_kvol_panel_boundary_bug.py.
    with mpmath.workdps(dps):
        for g in range(ng):
            K = mpmath.matrix(N, N)

            for s_pidx, (pa_s, pb_s, j_start, j_end) in enumerate(panel_bounds):
                source_nodes = x_nodes[j_start:j_end]
                for i in range(N):
                    x_i = x_nodes[i]

                    def op_path(x_prime, _xi=x_i, _g=g):
                        return optical_path(_xi, x_prime, _g)

                    ws = _basis_kernel_weights(
                        pa_s, pb_s, x_i, source_nodes, op_path, dps,
                    )
                    for j_loc, w in enumerate(ws):
                        K[i, j_start + j_loc] = w / 2

            K_per_group.append(K)

    return K_per_group


def _build_system_matrices(
    K_per_group: list,
    x_nodes: list,
    w_nodes: list,
    sig_t_at_node: list,
    sig_s_at_node: list,
    nusigf_at_node: list,
    chi_at_node: list,
    boundaries: list,
    sig_t_per_region: list[list],
    n_regions: int,
    ng: int,
    dps: int,
    boundary: str,
) -> tuple[object, object, object]:
    r"""Assemble scatter and fission operator matrices (dim × dim).

    The Peierls equation for scalar flux is:

    .. math::

       \varphi(x) = \tfrac12\!\int E_1(\tau)\,q(x')\,dx' + \varphi_{\rm bc}(x)

    Returns ``(A, B)`` where ``A = I − K_scatter``, ``B = K_fission``,
    and the eigenvalue problem is ``A\varphi = (1/k)\,B\varphi``.
    """
    N = len(x_nodes)
    dim = N * ng

    with mpmath.workdps(dps):
        K_scatter = mpmath.matrix(dim, dim)
        K_fission = mpmath.matrix(dim, dim)

        # Cell-interior kernel → scatter and fission operators
        for ge in range(ng):
            Kg = K_per_group[ge]
            for i in range(N):
                for j in range(N):
                    kij = Kg[i, j]
                    if kij == 0:
                        continue
                    for gs in range(ng):
                        row = i * ng + ge
                        col = j * ng + gs
                        K_scatter[row, col] += kij * sig_s_at_node[j][gs][ge]
                        K_fission[row, col] += kij * chi_at_node[i][ge] * nusigf_at_node[j][gs]

        # White-BC: add separable re-entry kernel
        if boundary == "white":
            L = boundaries[-1]

            def optical_from_face(x, face, g):
                a, b = (mpmath.mpf(0), x) if face == "left" else (x, L)
                if a >= b:
                    return mpmath.mpf(0)
                tau = mpmath.mpf(0)
                for rr in range(n_regions):
                    ra, rb = boundaries[rr], boundaries[rr + 1]
                    oa, ob = max(a, ra), min(b, rb)
                    if oa < ob:
                        tau += sig_t_per_region[rr][g] * (ob - oa)
                return tau

            # E₂ vectors from each face
            e2L: list[list] = []  # e2L[g][i]
            e2R: list[list] = []
            for g in range(ng):
                e2l, e2r = [], []
                for i in range(N):
                    e2l.append(e_n_mp(2, optical_from_face(x_nodes[i], "left", g), dps))
                    e2r.append(e_n_mp(2, optical_from_face(x_nodes[i], "right", g), dps))
                e2L.append(e2l)
                e2R.append(e2r)

            # Slab transmission T_g = 2 E₃(τ_total(g))
            for g in range(ng):
                tau_tot = sum(
                    sig_t_per_region[r][g] * (boundaries[r + 1] - boundaries[r])
                    for r in range(n_regions)
                )
                T_g = 2 * e_n_mp(3, tau_tot, dps)
                denom = 1 - T_g * T_g
                if abs(denom) < mpmath.power(10, -dps + 2):
                    continue  # optically thin — skip

                # White-BC kernel for the φ equation (no Σ_t factor):
                #   φ_bc(x_i) = 1/(1-T²) × Σ_j w_j q_j ×
                #     [e2L_i (e2L_j + T·e2R_j) + e2R_i (e2R_j + T·e2L_j)]
                for i in range(N):
                    for j in range(N):
                        bc = (
                            e2L[g][i] * (e2L[g][j] + T_g * e2R[g][j])
                            + e2R[g][i] * (e2R[g][j] + T_g * e2L[g][j])
                        ) * w_nodes[j] / denom

                        for gs in range(ng):
                            row = i * ng + g
                            col = j * ng + gs
                            K_scatter[row, col] += bc * sig_s_at_node[j][gs][g]
                            K_fission[row, col] += bc * chi_at_node[i][g] * nusigf_at_node[j][gs]

        # A = I - K_scatter (identity, not Σ_t diagonal)
        A = mpmath.matrix(dim, dim)
        for idx in range(dim):
            A[idx, idx] = mpmath.mpf(1)
        A -= K_scatter
        B = K_fission

    return A, B


# ═══════════════════════════════════════════════════════════════════════
# Solution container
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PeierlsSlabSolution:
    """Result of a Peierls Nyström solve on a 1-D slab."""

    x_nodes: np.ndarray
    """Quadrature node positions, shape ``(N,)``."""

    phi_values: np.ndarray
    """Flux at each node and group, shape ``(N, ng)``."""

    k_eff: float | None
    """Eigenvalue (None for fixed-source problems)."""

    slab_length: float
    n_groups: int
    n_quad: int
    precision_digits: int

    _panel_bounds: list | None = None

    def phi(self, x: np.ndarray, g: int = 0) -> np.ndarray:
        """Evaluate flux at arbitrary points via the piecewise Lagrange
        basis on composite-panel nodes.

        Falls through to ``np.interp`` when no panel structure is
        attached (legacy paths that predate the panel Nyström); the
        normal production path goes through
        :func:`~.peierls_geometry.lagrange_basis_on_panels`.
        """
        x = np.asarray(x, dtype=float).ravel()

        if self._panel_bounds is None:
            return np.interp(x, self.x_nodes, self.phi_values[:, g])

        values_g = self.phi_values[:, g]
        result = np.empty_like(x)
        for idx, xv in enumerate(x):
            basis = lagrange_basis_on_panels(
                self.x_nodes, self._panel_bounds, float(xv),
            )
            result[idx] = float(np.dot(basis, values_g))
        return result


# ═══════════════════════════════════════════════════════════════════════
# Main solver interface
# ═══════════════════════════════════════════════════════════════════════

def solve_peierls_eigenvalue(
    sig_t_regions: list[np.ndarray],
    sig_s_matrices: list[np.ndarray],
    nu_sig_f_all: list[np.ndarray],
    chi_all: list[np.ndarray],
    thicknesses: list[float],
    *,
    n_panels_per_region: int = 16,
    p_order: int = 6,
    precision_digits: int = 30,
    boundary: str = "white",
) -> PeierlsSlabSolution:
    r"""Solve the multi-group Peierls k-eigenvalue equation on a 1-D slab.

    Parameters
    ----------
    sig_t_regions : list of (ng,) arrays
        Total XS per region.
    sig_s_matrices : list of (ng, ng) arrays
        P0 scattering matrices per region. Convention:
        ``sig_s[g_src, g_dst]`` = rate from ``g_src`` into ``g_dst``
        (first index = source group, second = destination). Matches
        the project-wide ``sig_s`` convention documented in
        Sphinx §theory-peierls-multigroup and exposed by
        :func:`orpheus.derivations._xs_library.get_xs`.
    nu_sig_f_all : list of (ng,) arrays
        :math:`\nu\Sigma_f` per region.
    chi_all : list of (ng,) arrays
        Fission spectrum per region.
    thicknesses : list of float
        Region thicknesses.
    n_panels_per_region : int
        GL panels per material region.
    p_order : int
        GL order per panel.
    precision_digits : int
        mpmath working precision.
    boundary : {"white", "vacuum"}
        Boundary condition type.

    Returns
    -------
    PeierlsSlabSolution
    """
    dps = precision_digits
    n_regions = len(thicknesses)
    ng = len(sig_t_regions[0])

    with mpmath.workdps(dps):
        # Region boundaries
        boundaries = [mpmath.mpf(0)]
        for t in thicknesses:
            boundaries.append(boundaries[-1] + mpmath.mpf(t))

        # Sigma_t per region at mpmath precision
        sig_t_per_region = [
            [mpmath.mpf(float(sig_t_regions[r][g])) for g in range(ng)]
            for r in range(n_regions)
        ]

        def region_of(x):
            for r in range(n_regions):
                if x <= boundaries[r + 1] or r == n_regions - 1:
                    return r
            return n_regions - 1

        # Build composite GL quadrature
        gl_ref, gl_wt = gl_nodes_weights(p_order, dps)
        x_all: list = []
        w_all: list = []
        panel_bounds: list[tuple] = []

        for r in range(n_regions):
            pw = (boundaries[r + 1] - boundaries[r]) / n_panels_per_region
            for pidx in range(n_panels_per_region):
                pa = boundaries[r] + pidx * pw
                pb = pa + pw
                xp, wp = map_gl_to(gl_ref, gl_wt, pa, pb)
                panel_bounds.append((pa, pb, len(x_all), len(x_all) + len(xp)))
                x_all.extend(xp)
                w_all.extend(wp)

        N = len(x_all)

        # Node-to-panel mapping
        node_panel = [0] * N
        for pidx, (_, _, i0, i1) in enumerate(panel_bounds):
            for i in range(i0, i1):
                node_panel[i] = pidx

        # Per-node XS at mpmath precision
        sig_t_at_node = [sig_t_per_region[region_of(x_all[i])] for i in range(N)]
        sig_s_at_node = [
            [[mpmath.mpf(float(sig_s_matrices[region_of(x_all[i])][g1, g2]))
              for g2 in range(ng)] for g1 in range(ng)]
            for i in range(N)
        ]
        nusigf_at_node = [
            [mpmath.mpf(float(nu_sig_f_all[region_of(x_all[i])][g])) for g in range(ng)]
            for i in range(N)
        ]
        chi_at_node = [
            [mpmath.mpf(float(chi_all[region_of(x_all[i])][g])) for g in range(ng)]
            for i in range(N)
        ]

    # Build E₁ kernel matrices (one per group)
    K_per_group = _build_kernel_matrix(
        x_all, w_all, panel_bounds, node_panel,
        sig_t_at_node, boundaries, sig_t_per_region,
        n_regions, ng, dps,
    )

    # Assemble A and B matrices
    A, B = _build_system_matrices(
        K_per_group, x_all, w_all,
        sig_t_at_node, sig_s_at_node, nusigf_at_node, chi_at_node,
        boundaries, sig_t_per_region, n_regions, ng, dps, boundary,
    )

    dim = N * ng

    # Power iteration for the dominant eigenvalue.
    # Eigenproblem: A·φ = (1/k)·B·φ  where A = Σ_t − K_scatter,
    #                                        B = K_fission.
    # Standard fission source iteration:
    #   1) q^(n) = (1/k^(n)) B·φ^(n)
    #   2) A·φ^(n+1) = q^(n)
    #   3) k^(n+1) = k^(n) · ‖B·φ^(n+1)‖ / ‖B·φ^(n)‖
    with mpmath.workdps(dps):
        phi = mpmath.matrix(dim, 1)
        for i in range(dim):
            phi[i] = mpmath.mpf(1)

        tol = mpmath.power(10, -(dps - 5))
        k_val = mpmath.mpf(1)

        # Initial fission source norm
        fiss_old = B * phi
        prod_old = sum(abs(fiss_old[i]) for i in range(dim))

        for iteration in range(500):
            # Step 1: fission source divided by k
            q = mpmath.matrix(dim, 1)
            for i in range(dim):
                q[i] = fiss_old[i] / k_val

            # Step 2: transport solve
            phi_new = mpmath.lu_solve(A, q)

            # Step 3: update k
            fiss_new = B * phi_new
            prod_new = sum(abs(fiss_new[i]) for i in range(dim))
            k_new = k_val * prod_new / prod_old if prod_old > 0 else k_val

            # Normalise φ
            n_new = sum(abs(phi_new[i]) for i in range(dim))
            for i in range(dim):
                phi_new[i] /= n_new

            # Re-compute fiss for normalised phi (for next iteration)
            fiss_norm = B * phi_new
            prod_norm = sum(abs(fiss_norm[i]) for i in range(dim))

            converged = abs(k_new - k_val) < tol and iteration > 5
            phi, k_val = phi_new, k_new
            fiss_old, prod_old = fiss_norm, prod_norm

            if converged:
                break

    # Extract results
    k_eff = float(k_val)
    phi_arr = np.zeros((N, ng))
    for i in range(N):
        for g in range(ng):
            phi_arr[i, g] = float(phi[i * ng + g])

    # Normalise: unit integral per group
    x_f = np.array([float(xi) for xi in x_all])
    w_f = np.array([float(wi) for wi in w_all])
    for g in range(ng):
        integral = np.dot(w_f, phi_arr[:, g])
        if abs(integral) > 1e-30:
            phi_arr[:, g] /= integral

    # Panel bounds for interpolation (float)
    pb_float = [(float(a), float(b), i0, i1) for a, b, i0, i1 in panel_bounds]

    return PeierlsSlabSolution(
        x_nodes=x_f,
        phi_values=phi_arr,
        k_eff=k_eff,
        slab_length=sum(thicknesses),
        n_groups=ng,
        n_quad=N,
        precision_digits=precision_digits,
        _panel_bounds=pb_float,
    )


# ═══════════════════════════════════════════════════════════════════════
# ContinuousReferenceSolution builders
# ═══════════════════════════════════════════════════════════════════════

_MAT_IDS = {1: [2], 2: [2, 0], 4: [2, 3, 1, 0]}


def _build_peierls_slab_case(
    ng_key: str,
    n_regions: int,
    n_panels_per_region: int = 16,
    p_order: int = 6,
    precision_digits: int = 30,
) -> ContinuousReferenceSolution:
    """Build a Peierls reference matching the corresponding cp_slab case."""
    from .cp_slab import _THICKNESSES

    layout = LAYOUTS[n_regions]
    ng = int(ng_key[0])
    thicknesses = _THICKNESSES[n_regions]

    xs_list = [get_xs(region, ng_key) for region in layout]
    sig_t_regions = [xs["sig_t"] for xs in xs_list]
    sig_s_matrices = [xs["sig_s"] for xs in xs_list]
    nu_sig_f_all = [xs["nu"] * xs["sig_f"] for xs in xs_list]
    chi_all = [xs["chi"] for xs in xs_list]

    sol = solve_peierls_eigenvalue(
        sig_t_regions, sig_s_matrices, nu_sig_f_all, chi_all,
        thicknesses,
        n_panels_per_region=n_panels_per_region,
        p_order=p_order,
        precision_digits=precision_digits,
        boundary="white",
    )

    def phi_fn(x: np.ndarray, g: int = 0) -> np.ndarray:
        return sol.phi(x, g)

    mat_ids = _MAT_IDS[n_regions]
    materials = {
        mat_ids[i]: get_mixture(region, ng_key)
        for i, region in enumerate(layout)
    }

    return ContinuousReferenceSolution(
        name=f"peierls_slab_{ng}eg_{n_regions}rg",
        problem=ProblemSpec(
            materials=materials,
            geometry_type="slab",
            geometry_params={
                "length": sum(thicknesses),
                "thicknesses": thicknesses,
                "mat_ids": mat_ids,
            },
            boundary_conditions={"left": "white", "right": "white"},
            is_eigenvalue=True,
            n_groups=ng,
        ),
        operator_form="integral-peierls",
        phi=phi_fn,
        k_eff=sol.k_eff,
        provenance=Provenance(
            citation=(
                "Case & Zweifel 1967, Ch. 4; "
                "Kress 2014 (Nyström for Fredholm equations)"
            ),
            derivation_notes=(
                f"Nyström discretisation with {n_panels_per_region} panels × "
                f"{p_order} GL points per region, E₁ kernel with singularity "
                f"subtraction (product-integration for log part), "
                f"white BC via E₂ re-entry closure."
            ),
            sympy_expression=None,
            precision_digits=precision_digits,
        ),
        equation_labels=("peierls-equation",),
        vv_level="L1",
        description=f"{ng}G {n_regions}-region slab Peierls (E₁ Nyström, white BC)",
        tolerance="O(h^2)",
    )


