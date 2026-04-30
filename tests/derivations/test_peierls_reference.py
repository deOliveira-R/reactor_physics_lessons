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
   :func:`~orpheus.derivations.peierls_geometry.K_vol_element_adaptive`
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
    slab_uniform_source_white_bc_analytical,
    slab_K_vol_element,
    cylinder_uniform_source_analytical,
    sphere_uniform_source_analytical,
)
from orpheus.derivations import peierls_slab
from orpheus.derivations.peierls_geometry import (
    SLAB_POLAR_1D,
    SPHERE_1D,
    CYLINDER_1D,
    K_vol_element_adaptive,
    build_closure_operator,
    build_volume_kernel,
    build_volume_kernel_adaptive,
    build_white_bc_correction_rank_n,
    composite_gl_r,
    compute_G_bc,
    compute_G_bc_inner,
    compute_G_bc_outer,
    compute_P_esc,
    compute_P_esc_inner,
    compute_P_esc_outer,
    CurvilinearGeometry,
    gl_nodes_weights,
    lagrange_basis_on_panels,
    map_gl_to,
    reflection_mark,
)


# ═══════════════════════════════════════════════════════════════════════
# Layer 1 — Slab solver infrastructure verification
# ═══════════════════════════════════════════════════════════════════════

def _build_slab_K_and_nodes(L, sig_t, n_panels=8, p_order=6, dps=30):
    """Helper: use ORPHEUS slab internals to build K matrix + nodes."""
    # Build composite Gauss-Legendre quadrature exactly as the solver does.
    with mpmath.workdps(dps):
        boundaries = [mpmath.mpf(0), mpmath.mpf(L)]
        gl_ref, gl_wt = gl_nodes_weights(p_order, dps)

        x_all, w_all, panel_bounds = [], [], []
        pw = mpmath.mpf(L) / n_panels
        for pidx in range(n_panels):
            pa = mpmath.mpf(pidx) * pw
            pb = pa + pw
            i_start = len(x_all)
            xs, ws = map_gl_to(gl_ref, gl_wt, pa, pb)
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
@pytest.mark.catches("ERR-032")
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


# ═══════════════════════════════════════════════════════════════════════
# Layer 3 — Curvilinear K volume-kernel verification (issue #114)
# ═══════════════════════════════════════════════════════════════════════

def _build_sphere_K(R, sig_t, n_panels, p_order,
                    *, n_angular=32, n_rho=32, dps=25):
    """Helper: build a single-region sphere K via production assembly."""
    with mpmath.workdps(dps):
        gl_ref, gl_wt = gl_nodes_weights(p_order, dps)
        x_all, w_all, pbs = [], [], []
        pw = mpmath.mpf(R) / n_panels
        for pidx in range(n_panels):
            pa = mpmath.mpf(pidx) * pw
            pb = pa + pw
            i_start = len(x_all)
            xs, ws = map_gl_to(gl_ref, gl_wt, pa, pb)
            x_all.extend(xs); w_all.extend(ws)
            pbs.append((pa, pb, i_start, len(x_all)))
    x_nodes = np.array([float(x) for x in x_all])
    pbs_f = [(float(pa), float(pb), i0, i1) for pa, pb, i0, i1 in pbs]
    radii = np.array([R])
    sig_t_arr = np.array([sig_t])
    K = build_volume_kernel(
        SPHERE_1D, x_nodes, pbs_f, radii, sig_t_arr,
        n_angular=n_angular, n_rho=n_rho, dps=dps,
    )
    return K, x_nodes, pbs_f, radii, sig_t_arr


def _shell_avg_sphere_K(i, j, x_nodes, pbs, R, sig_t, *, dps=40):
    """Independent reference via shell-average of the 3-D isotropic point
    kernel (used as the arbiter against the polar-adaptive reference).

        <exp(-Σ_t d)/(4π d²)>_shell =
            (1/(8π r_i r')) * [E_1(Σ_t|r_i-r'|) - E_1(Σ_t(r_i+r'))]

    Volume-integrated against L_j(r') gives
        K[i, j] = Σ_t(r_i) * (1/(2 r_i)) * ∫_0^R r' * [E_1 - E_1] * L_j dr'
    with subdivision at 0, r_i, R, and each panel boundary (handles both
    the log singularity at r'=r_i and the Lagrange-basis kinks).
    """
    r_i = mpmath.mpf(x_nodes[i])
    s = mpmath.mpf(sig_t)

    def integrand(rp):
        rp_mp = mpmath.mpf(rp)
        if rp_mp == 0 or rp_mp == r_i:
            return mpmath.mpf(0)
        tau1 = s * abs(r_i - rp_mp)
        tau2 = s * (r_i + rp_mp)
        term = mpmath.expint(1, tau1) - mpmath.expint(1, tau2)
        L_vals = lagrange_basis_on_panels(x_nodes, pbs, float(rp_mp))
        return rp_mp * term * mpmath.mpf(float(L_vals[j]))

    breaks_set = {mpmath.mpf(0), mpmath.mpf(R), r_i}
    for pa, pb, _, _ in pbs:
        breaks_set.add(mpmath.mpf(pa))
        breaks_set.add(mpmath.mpf(pb))
    breaks = sorted(breaks_set)
    with mpmath.workdps(dps):
        val = mpmath.quad(integrand, breaks)
    return s / (2 * r_i) * val


@pytest.mark.l1
@pytest.mark.verifies("peierls-unified")
class TestSphereKMatrixElementwise:
    """Element-wise comparison of the sphere production K (via
    :func:`~peierls_geometry.build_volume_kernel`) to the
    mesh-independent shell-average reference (:func:`_shell_avg_sphere_K`).

    Before the fix (:issue:`114`), cross-panel-crossing rays produced
    0.5–5% per-entry error, plateauing at ~1e-3 even with n_rho=128
    (see retired ``diag_sphere_kvol_ray_crossing.py``). After the fix
    (ρ + ω subdivision at panel crossings and tangent angles), each
    entry converges algebraically in (n_angular, n_rho).
    """

    @pytest.mark.catches("ERR-029")
    @pytest.mark.parametrize("i,j", [(0, 0), (2, 2), (5, 5), (3, 3), (1, 1)])
    def test_sphere_K_entry_vs_shell_avg_reference(self, i, j):
        R, sig_t = 1.0, 1.0
        n_panels, p_order = 2, 3
        K, x_nodes, pbs, _, _ = _build_sphere_K(
            R, sig_t, n_panels, p_order,
            n_angular=64, n_rho=32, dps=25,
        )
        ref = float(_shell_avg_sphere_K(i, j, x_nodes, pbs, R, sig_t, dps=40))
        got = float(K[i, j])
        assert abs(ref) > 1e-10, f"Shell-avg ref vanishes at K[{i},{j}]"
        rel = abs(got - ref) / abs(ref)
        # Algebraic (not spectral) convergence due to the sqrt cusp at
        # tangent critical angles; gate at 1e-5 which n=64, n_rho=32 meets.
        assert rel < 1e-5, (
            f"Sphere K[{i},{j}]: production={got:.6e}, shell-avg ref={ref:.6e}, "
            f"rel_diff={rel:.3e}. Regression of issue #114 (ρ / ω subdivision "
            f"at ray/panel-boundary crossings and tangent angles)."
        )

    @pytest.mark.catches("ERR-029")
    def test_sphere_K_converges_under_refinement(self):
        """At fixed panels (p=3, 2 panels), refining n_angular=n_rho
        monotonically reduces the K[3,3] error vs the shell-avg reference.
        Pre-fix this was non-monotonic (oscillating 4e-2 → 7e-2 → 1e-2).
        """
        R, sig_t = 1.0, 1.0
        K, x_nodes, pbs, _, _ = _build_sphere_K(
            R, sig_t, 2, 3, n_angular=8, n_rho=8, dps=20,
        )
        ref = float(_shell_avg_sphere_K(3, 3, x_nodes, pbs, R, sig_t, dps=40))
        errors = []
        for n_q in (8, 16, 32, 64):
            K, *_ = _build_sphere_K(
                R, sig_t, 2, 3,
                n_angular=n_q, n_rho=n_q, dps=20,
            )
            errors.append(abs(float(K[3, 3]) - ref) / abs(ref))

        # Allow small noise floor tolerance but require non-oscillating
        # monotonic decrease.
        for k in range(len(errors) - 1):
            assert errors[k + 1] <= errors[k] * 1.5, (
                f"Sphere K[3,3] error oscillates under refinement: "
                f"errors = {errors}. Regression of issue #114."
            )
        assert errors[-1] < 1e-4, (
            f"At n_angular=n_rho=64, K[3,3] error = {errors[-1]:.3e}, "
            f"expected < 1e-4."
        )


# ═══════════════════════════════════════════════════════════════════════
# Layer 4 — Unified slab polar-form (same adaptive-mpmath methodology
# as curvilinear; proves slab is a first-class CurvilinearGeometry kind)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
@pytest.mark.verifies("peierls-unified")
class TestSlabPolarReferenceEquivalence:
    """Two independent constructions of the slab K matrix must agree
    at machine precision:

    * :func:`slab_K_vol_element` — adaptive ``mpmath.quad`` in real-space
      against the classical :math:`E_1` Nyström integrand.
    * :func:`K_vol_element_adaptive(SLAB_POLAR_1D, ...)` — adaptive
      ``mpmath.quad`` in observer-centred polar :math:`(\\mu, \\rho)`
      coordinates, the unified verification primitive.

    The polar form has a :math:`\\Sigma_t` prefactor from the unified
    operator :math:`\\Sigma_t \\varphi = K q`; the :math:`E_1` form
    does not. So the equivalence statement is

    .. math:: K_{\\rm polar} = \\Sigma_t \\cdot K_{E_1}.

    This is the "two independent formulations agree" verification —
    any bug in either construction breaks agreement.
    """

    def test_adaptive_polar_matches_E1_reference(self):
        """Unified polar primitive equals :math:`\\Sigma_t` times the
        classical :math:`E_1` real-space reference to machine precision."""
        L, sig_t = 1.0, 1.0
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])
        dps = 25
        _, x_nodes, _, panel_bounds = _build_slab_K_and_nodes(
            L, sig_t, n_panels=2, p_order=4, dps=dps,
        )

        for i, j in [(0, 0), (3, 3), (4, 3), (0, 5)]:
            ref_e1 = float(slab_K_vol_element(
                i, j, x_nodes, panel_bounds, L, sig_t, dps=dps,
            ))
            ref_polar = float(K_vol_element_adaptive(
                SLAB_POLAR_1D, i, j, x_nodes, panel_bounds, radii,
                sig_t_arr, dps=dps,
            ))
            expected_polar = sig_t * ref_e1
            if abs(expected_polar) > 1e-30:
                rel = abs(ref_polar - expected_polar) / abs(expected_polar)
            else:
                rel = abs(ref_polar - expected_polar)
            assert rel < 1e-10, (
                f"Polar form and E_1 form disagree at K[{i},{j}]: "
                f"polar={ref_polar:.12e}, σ·E_1={expected_polar:.12e}, "
                f"rel_diff={rel:.3e}. Mathematical equivalence broken."
            )


@pytest.mark.l1
@pytest.mark.verifies("peierls-unified")
class TestSlabPolarBuildVolumeKernel:
    """``build_volume_kernel(SLAB_POLAR_1D, ...)`` routes through the
    unified adaptive primitive (one ``mpmath.quad`` per K element,
    machine precision by construction). The earlier
    moment-form / τ-Laguerre fast paths have been archived (Issue
    #117 captures the moment form for future production CP).

    These two tests are the production-tier verification: K matrix
    elements match the legacy :math:`E_1` form at machine precision,
    and row-sum identity matches the analytical vacuum-BC flux.
    Small N (one panel × p=2 = 2 nodes) keeps the adaptive cost
    bounded.
    """

    def test_matches_E1_reference_at_machine_precision(self):
        """Unified slab K via ``build_volume_kernel(SLAB_POLAR_1D, ...)``
        matches the legacy :math:`E_1` reference to machine precision
        (adaptive `mpmath.quad` is exact)."""
        L, sig_t = 1.0, 1.0
        dps = 25
        K_legacy, x_nodes, _, panel_bounds = _build_slab_K_and_nodes(
            L, sig_t, n_panels=1, p_order=2, dps=dps,
        )
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])

        K_unified = build_volume_kernel(
            SLAB_POLAR_1D, x_nodes, panel_bounds, radii, sig_t_arr,
            n_angular=0, n_rho=0, dps=dps,
        )
        N = len(x_nodes)
        for i in range(N):
            for j in range(N):
                # Unified operator: K_unified ≡ Σ_t · K_legacy
                ref = sig_t * float(K_legacy[i, j])
                if abs(ref) > 1e-30:
                    rel = abs(K_unified[i, j] - ref) / abs(ref)
                    assert rel < 1e-10, (
                        f"K[{i},{j}] disagreement: unified={K_unified[i,j]:.6e}, "
                        f"σ·E_1={ref:.6e}, rel={rel:.3e}"
                    )

    def test_row_sum_identity(self):
        """``K · [1,1,...]`` equals :math:`\\Sigma_t \\varphi(x)` for
        uniform unit source on a pure-absorber vacuum-BC slab. Tests
        the full assembly in one shot, independent of the legacy E_1
        reference."""
        L, sig_t = 1.0, 1.0
        dps = 25
        _, x_nodes, _, panel_bounds = _build_slab_K_and_nodes(
            L, sig_t, n_panels=1, p_order=2, dps=dps,
        )
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])
        K = build_volume_kernel(
            SLAB_POLAR_1D, x_nodes, panel_bounds, radii, sig_t_arr,
            n_angular=0, n_rho=0, dps=dps,
        )
        N = len(x_nodes)
        for i in range(N):
            ref = sig_t * float(slab_uniform_source_analytical(
                x_nodes[i], L, sig_t, dps=dps,
            ))
            got = float(K[i].sum())
            if abs(ref) > 1e-30:
                rel = abs(got - ref) / abs(ref)
                assert rel < 1e-10, (
                    f"Row-sum K[{i}, :].sum() = {got:.10e} vs "
                    f"σ·φ_analytical = {ref:.10e}, rel={rel:.3e}"
                )


# (Cylinder-polar equivalence tests were retired 2026-04-19: cylinder-polar
# is mathematically equivalent to cylinder-1d; the assembly was archived to
# ``derivations/archive/peierls_cylinder_polar_assembly.py``.)


# ═══════════════════════════════════════════════════════════════════════
# Layer 5 — Cylinder / sphere K-matrix row-sum identity at machine
# precision against the new analytical vacuum-BC references (the
# 2026-04-20 strategic milestone).
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
@pytest.mark.verifies("peierls-unified")
class TestCylinderKernelRowSum:
    r"""Cylinder K-matrix row-sum equals the semi-analytical vacuum-BC
    uniform-source flux at machine precision.

    Identity: :math:`\sum_j K_{ij}\cdot 1 \;=\; \Sigma_t\,\varphi_{\rm cyl}(r_i)`
    where :math:`\varphi_{\rm cyl}` is the reference from
    :func:`cylinder_uniform_source_analytical` (single ``mpmath.quad``
    over in-plane azimuth; Bickley :math:`\mathrm{Ki}_2` absorbing the
    out-of-plane polar integral).

    The LHS K is built via
    :func:`build_volume_kernel_adaptive(CYLINDER_1D, ...)` — the unified
    verification primitive at machine precision. Any deviation isolates
    the bug to one of:

    * the analytical reference (the new closed-form integral),
    * the unified primitive's geometry primitives (:meth:`CurvilinearGeometry.rho_max`,
      :meth:`~.source_position`, :meth:`~.optical_depth_along_ray`,
      :meth:`~.volume_kernel_mp`).

    The parametrisation sweeps one-MFP, thin-larger-radius, and
    thick-smaller-radius regimes. Small N (one panel × ``p=2`` = 2
    nodes) keeps the N² adaptive K cost bounded.
    """

    @pytest.mark.parametrize("R, sig_t", [
        (1.0, 1.0),   # 1 MFP cylinder
        (2.0, 0.5),   # same optical thickness, larger R
        (0.5, 2.0),   # same optical thickness, smaller R
    ])
    def test_row_sum_matches_analytical_uniform_source(self, R, sig_t):
        dps = 25
        radii = np.array([R])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, panel_bounds = composite_gl_r(
            radii, n_panels_per_region=1, p_order=2, dps=dps,
        )
        K = build_volume_kernel_adaptive(
            CYLINDER_1D, r_nodes, panel_bounds, radii, sig_t_arr,
            dps=dps,
        )
        N = len(r_nodes)
        for i in range(N):
            ref = sig_t * float(cylinder_uniform_source_analytical(
                float(r_nodes[i]), R, sig_t, dps=dps,
            ))
            got = float(K[i].sum())
            if abs(ref) > 1e-30:
                rel = abs(got - ref) / abs(ref)
                assert rel < 1e-10, (
                    f"Cylinder K row-sum at r={r_nodes[i]:.4f} "
                    f"(R={R}, Σ_t={sig_t}): K-sum={got:.10e}, "
                    f"σ·φ_analytical={ref:.10e}, rel={rel:.3e}. "
                    f"Localises bug to the analytical reference or the "
                    f"unified primitive's geometry primitives."
                )


@pytest.mark.l1
@pytest.mark.verifies("peierls-unified")
class TestSphereKernelRowSum:
    r"""Sphere K-matrix row-sum equals the semi-analytical vacuum-BC
    uniform-source flux at machine precision.

    Identity: :math:`\sum_j K_{ij}\cdot 1 \;=\; \Sigma_t\,\varphi_{\rm sph}(r_i)`
    where :math:`\varphi_{\rm sph}` is the reference from
    :func:`sphere_uniform_source_analytical` (single ``mpmath.quad``
    over :math:`\mu = \cos\Theta`; spherical symmetry collapses the
    azimuthal integral analytically).

    The LHS K is built via
    :func:`build_volume_kernel_adaptive(SPHERE_1D, ...)`. Same pattern
    and rationale as :class:`TestCylinderKernelRowSum`.
    """

    @pytest.mark.parametrize("R, sig_t", [
        (1.0, 1.0),   # 1 MFP sphere
        (2.0, 0.5),   # same optical thickness, larger R
        (0.5, 2.0),   # same optical thickness, smaller R
    ])
    def test_row_sum_matches_analytical_uniform_source(self, R, sig_t):
        dps = 25
        radii = np.array([R])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, panel_bounds = composite_gl_r(
            radii, n_panels_per_region=1, p_order=2, dps=dps,
        )
        K = build_volume_kernel_adaptive(
            SPHERE_1D, r_nodes, panel_bounds, radii, sig_t_arr,
            dps=dps,
        )
        N = len(r_nodes)
        for i in range(N):
            ref = sig_t * float(sphere_uniform_source_analytical(
                float(r_nodes[i]), R, sig_t, dps=dps,
            ))
            got = float(K[i].sum())
            if abs(ref) > 1e-30:
                rel = abs(got - ref) / abs(ref)
                assert rel < 1e-10, (
                    f"Sphere K row-sum at r={r_nodes[i]:.4f} "
                    f"(R={R}, Σ_t={sig_t}): K-sum={got:.10e}, "
                    f"σ·φ_analytical={ref:.10e}, rel={rel:.3e}. "
                    f"Localises bug to the analytical reference or the "
                    f"unified primitive's geometry primitives."
                )


# ═══════════════════════════════════════════════════════════════════════
# Layer 6 — Slab BC tensor-network (rank-1 Mark / white) verification
# Closes Issue #118: compute_P_esc / compute_G_bc slab-polar gap. The
# curvilinear BC machinery now supports slab-polar via:
#   1. polymorphic `ray_direction_cosine` replacing hard-coded
#      `np.cos(omega_pts)` in compute_P_esc / compute_P_esc_mode
#   2. explicit slab-polar branch in compute_G_bc using the closed-form
#      `2·[E_2(Σ_t x) + E_2(Σ_t(L − x))]` sum-of-faces expression
#   3. slab cases for `radial_volume_weight` (→ 1) and
#      `rank1_surface_divisor` (→ 2, two unit-area faces)
#
# Rank-1 Mark closure is INTENTIONALLY approximate — it omits the
# `T = 2·E_3(τ_L)` self-feedback transmission in the partial-current
# balance. The same approximation is present for cylinder and sphere
# (see `TestWhiteBCThickLimit` classes). The slab fixes bring the
# machinery to parity, not exactness.
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSlabPescClosedForm:
    r"""Factor-level verification — slab :math:`P_{\rm esc}`.

    For a homogeneous slab :math:`[0, L]` the uncollided escape
    probability from interior observer :math:`x_i` admits the closed
    form

    .. math::

       P_{\rm esc}^{\rm slab}(x_i) \;=\;
         \tfrac{1}{2}\bigl[E_2(\Sigma_t\,x_i)
                         + E_2(\Sigma_t\,(L - x_i))\bigr],

    deriving from the :math:`\mu`-integration of the isotropic point
    kernel over the two half-spaces. The
    :func:`~orpheus.derivations.peierls_geometry.compute_P_esc` slab
    branch evaluates this via :func:`mpmath.expint` directly
    (machine-precision; bypasses the grazing-ray GL stiffness of the
    unified quadrature form).
    """

    @pytest.mark.parametrize("L, sig_t", [
        (1.0, 1.0),
        (2.0, 0.5),
        (0.5, 2.0),
    ])
    def test_matches_E2_sum_at_machine_precision(self, L, sig_t):
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
        P_esc = compute_P_esc(
            SLAB_POLAR_1D, r_nodes, radii, sig_t_arr,
            n_angular=32, dps=25,
        )
        for i, x_i in enumerate(r_nodes):
            tau = sig_t * float(x_i)
            tau_prime = sig_t * (L - float(x_i))
            expected = 0.5 * (float(mpmath.expint(2, tau))
                              + float(mpmath.expint(2, tau_prime)))
            rel = abs(P_esc[i] - expected) / abs(expected)
            assert rel < 1e-14, (
                f"Slab P_esc(x={x_i:.4f}) deviates from closed-form "
                f"(L={L}, Σ_t={sig_t}): got={P_esc[i]:.16e}, "
                f"expected={expected:.16e}, rel={rel:.3e}"
            )


@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSlabGbcClosedForm:
    r"""Factor-level verification — slab :math:`G_{\rm bc}`.

    For a homogeneous slab with unit uniform isotropic inward partial
    currents at *both* faces, the scalar flux at interior
    :math:`x_i` is

    .. math::

       G_{\rm bc}^{\rm slab}(x_i) \;=\;
           2\,\bigl[E_2(\Sigma_t\,x_i) + E_2(\Sigma_t\,(L - x_i))\bigr],

    deriving from the :math:`\psi_{\rm in} = J^-/\pi` convention (2π
    azimuthal factor absorbed) multiplied by two faces. The
    :func:`~orpheus.derivations.peierls_geometry.compute_G_bc` slab
    branch evaluates this closed form directly.
    """

    @pytest.mark.parametrize("L, sig_t", [
        (1.0, 1.0),
        (2.0, 0.5),
        (0.5, 2.0),
    ])
    def test_matches_E2_sum_at_machine_precision(self, L, sig_t):
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
        G_bc = compute_G_bc(
            SLAB_POLAR_1D, r_nodes, radii, sig_t_arr,
            n_surf_quad=32, dps=25,
        )
        for i, x_i in enumerate(r_nodes):
            tau = sig_t * float(x_i)
            tau_prime = sig_t * (L - float(x_i))
            expected = 2.0 * (float(mpmath.expint(2, tau))
                              + float(mpmath.expint(2, tau_prime)))
            rel = abs(G_bc[i] - expected) / abs(expected)
            assert rel < 1e-14, (
                f"Slab G_bc(x={x_i:.4f}) deviates from closed-form "
                f"(L={L}, Σ_t={sig_t}): got={G_bc[i]:.16e}, "
                f"expected={expected:.16e}, rel={rel:.3e}"
            )


@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSlabKbcStructure:
    r"""Structural checks on the rank-1 Mark K_bc for slab-polar.

    1. Symmetry under :math:`x \leftrightarrow L - x` for symmetric
       configurations: the homogeneous slab's K_bc row-sum must be
       invariant under the mirror swap (diagnostic that caught the
       pre-fix bug, where the cylinder fall-through produced
       :math:`0.17` vs :math:`0.35`).

    2. Row-sum closed form (first-order Mark, single region, uniform
       Σ_t):

       .. math::

          \sum_j K_{\rm bc}[i, j]\cdot 1 \;=\;
              \bigl(\tfrac{1}{2} - E_3(\Sigma_t L)\bigr)\,
              \bigl[E_2(\Sigma_t\,x_i) + E_2(\Sigma_t\,(L - x_i))\bigr].

       The factor :math:`1/2 - E_3(\tau_L)` comes from the
       :math:`P_{\rm esc}` volume integral and the Peierls antiderivative
       identity :math:`\int_0^{\tau_L} E_2\,\mathrm du = 1/2 - E_3(\tau_L)`.
    """

    def test_K_bc_row_sum_symmetric_under_mirror(self):
        L, sig_t = 1.0, 1.0
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])
        r_nodes, r_wts, _ = composite_gl_r(radii, 2, 4, dps=25)
        K_bc = build_white_bc_correction_rank_n(
            SLAB_POLAR_1D, r_nodes, r_wts, radii, sig_t_arr,
            n_angular=32, n_surf_quad=32, dps=25, n_bc_modes=1,
        )
        row_sums = K_bc.sum(axis=1)
        N = len(r_nodes)
        for i in range(N // 2):
            rel = abs(row_sums[i] - row_sums[N - 1 - i]) / abs(row_sums[i])
            assert rel < 1e-12, (
                f"Slab K_bc row-sum not symmetric under x ↔ L−x at "
                f"i={i}: got {row_sums[i]:.10e} vs {row_sums[N-1-i]:.10e}"
            )

    @pytest.mark.parametrize("L, sig_t", [
        (1.0, 1.0),
        (2.0, 0.5),
        (0.5, 2.0),
    ])
    def test_K_bc_row_sum_matches_first_order_closed_form(self, L, sig_t):
        # Tolerance is set by GL quadrature error on ∫ P_esc(x) dx,
        # which has mild log singularities at x ∈ {0, L} from the E_2
        # boundary behaviour. At n_panels=8, p=8 this converges to
        # ~1e-6 relative error; the formula itself is exact.
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])
        r_nodes, r_wts, _ = composite_gl_r(radii, 8, 8, dps=25)
        K_bc = build_white_bc_correction_rank_n(
            SLAB_POLAR_1D, r_nodes, r_wts, radii, sig_t_arr,
            n_angular=32, n_surf_quad=32, dps=25, n_bc_modes=1,
        )
        row_sums = K_bc.sum(axis=1)
        tau_L = sig_t * L
        E3_tau_L = float(mpmath.expint(3, tau_L))
        prefactor = 0.5 - E3_tau_L
        for i, x_i in enumerate(r_nodes):
            tau = sig_t * float(x_i)
            tau_prime = sig_t * (L - float(x_i))
            E2_sum = (float(mpmath.expint(2, tau))
                      + float(mpmath.expint(2, tau_prime)))
            expected = prefactor * E2_sum
            rel = abs(row_sums[i] - expected) / abs(expected)
            assert rel < 1e-5, (
                f"Slab K_bc row-sum(x={x_i:.4f}) disagrees with "
                f"(1/2 − E_3(τ_L))·(E_2 + E_2) at rel={rel:.3e} "
                f"(L={L}, Σ_t={sig_t}): got={row_sums[i]:.10e}, "
                f"expected={expected:.10e}"
            )

    def test_K_bc_row_sum_converges_under_refinement(self):
        """GL quadrature error on the P_esc integral decreases
        algebraically with panel count (the E_2 log behaviour at the
        faces keeps convergence sub-spectral)."""
        L, sig_t = 1.0, 1.0
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])
        tau_L = sig_t * L
        E3_tau_L = float(mpmath.expint(3, tau_L))
        prefactor = 0.5 - E3_tau_L

        prev_err = float("inf")
        for n_panels, p_order in [(2, 4), (4, 6), (8, 8)]:
            r_nodes, r_wts, _ = composite_gl_r(radii, n_panels, p_order, dps=25)
            K_bc = build_white_bc_correction_rank_n(
                SLAB_POLAR_1D, r_nodes, r_wts, radii, sig_t_arr,
                n_angular=32, n_surf_quad=32, dps=25, n_bc_modes=1,
            )
            row_sums = K_bc.sum(axis=1)
            max_rel = 0.0
            for i, x_i in enumerate(r_nodes):
                tau = sig_t * float(x_i)
                tau_prime = sig_t * (L - float(x_i))
                E2_sum = (float(mpmath.expint(2, tau))
                          + float(mpmath.expint(2, tau_prime)))
                expected = prefactor * E2_sum
                max_rel = max(max_rel, abs(row_sums[i] - expected) / abs(expected))
            assert max_rel < prev_err, (
                f"K_bc row-sum error did not decrease: "
                f"(n_panels={n_panels}, p={p_order}) → {max_rel:.3e}, "
                f"previous {prev_err:.3e}"
            )
            prev_err = max_rel
        assert prev_err < 1e-5, (
            f"Finest refinement (n_panels=8, p=8) did not reach 1e-5: "
            f"got {prev_err:.3e}"
        )


@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSlabWhiteBCInfiniteMediumIdentity:
    r"""The Wigner-Seitz identity for slab white BC with uniform
    source on a pure absorber:
    :math:`\varphi_{\rm white}(x) \equiv 1/\Sigma_t` for all :math:`x`,
    any :math:`L`. This is the
    :func:`~orpheus.derivations.peierls_reference.slab_uniform_source_white_bc_analytical`
    closed form (as of commit correcting the earlier algebra bug in
    commit 2538cfe).

    The identity holds because white BC conserves neutrons; for a
    uniform material the cell is indistinguishable from an infinite
    medium at equilibrium, where the pure-absorber balance
    :math:`\Sigma_t\,\varphi = S` gives :math:`\varphi = 1/\Sigma_t`
    pointwise.
    """

    @pytest.mark.parametrize("L, sig_t", [
        (0.1, 1.0),   # thin cell
        (1.0, 1.0),   # 1 MFP
        (5.0, 2.0),   # thick, different Σ_t
        (100.0, 0.5),
    ])
    def test_flux_equals_1_over_sigma_t(self, L, sig_t):
        expected = 1.0 / sig_t
        for x_frac in [0.0, 0.2, 0.5, 0.8, 1.0]:
            x = x_frac * L
            phi = float(slab_uniform_source_white_bc_analytical(
                x, L, sig_t, dps=40,
            ))
            rel = abs(phi - expected) / abs(expected)
            assert rel < 1e-30, (
                f"Slab φ_white(x={x:.3f}, L={L}, Σ_t={sig_t}) = "
                f"{phi:.20e}, expected {expected:.20e} (1/Σ_t), "
                f"rel={rel:.3e}"
            )


@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSlabRankNNotImplementedGuard:
    r"""Rank-N :math:`(n > 0)` BC for slab-polar requires per-face mode
    decomposition (two faces → :math:`A = \mathbb{R}^{2N}` rather than
    :math:`\mathbb{R}^{N}`). The unified
    :func:`~orpheus.derivations.peierls_geometry.compute_P_esc_mode`
    and :func:`~orpheus.derivations.peierls_geometry.compute_G_bc_mode`
    explicitly raise :class:`NotImplementedError` for slab at
    :math:`n_{\rm mode} > 0`; the rank-1 (:math:`n_{\rm bc\_modes} = 1`,
    mode-0-only) path is the supported configuration.
    """

    def test_P_esc_mode_raises_for_slab_polar_at_mode_n(self):
        from orpheus.derivations.peierls_geometry import compute_P_esc_mode
        radii = np.array([1.0])
        sig_t = np.array([1.0])
        r_nodes, _, _ = composite_gl_r(radii, 1, 2, dps=20)
        with pytest.raises(NotImplementedError, match="slab-polar"):
            compute_P_esc_mode(
                SLAB_POLAR_1D, r_nodes, radii, sig_t, n_mode=1,
                n_angular=16, dps=20,
            )

    def test_G_bc_mode_raises_for_slab_polar_at_mode_n(self):
        from orpheus.derivations.peierls_geometry import compute_G_bc_mode
        radii = np.array([1.0])
        sig_t = np.array([1.0])
        r_nodes, _, _ = composite_gl_r(radii, 1, 2, dps=20)
        with pytest.raises(NotImplementedError, match="slab-polar"):
            compute_G_bc_mode(
                SLAB_POLAR_1D, r_nodes, radii, sig_t, n_mode=1,
                n_surf_quad=16, dps=20,
            )


# ═══════════════════════════════════════════════════════════════════════
# Layer 6b — Per-surface P_esc / G_bc primitives (Phase F.2)
#
# The Phase F per-face :class:`BoundaryClosureOperator` requires P_esc
# and G_bc to be resolved *per boundary* — not merely summed over all
# boundaries of the cell. These tests verify the additive split:
#
# - Slab: face 0 and face L each carry half of the rank-1 flux. Per-face
#   closed forms match :math:`(1/2)\,E_2` and :math:`2\,E_2` at machine
#   precision; the sum over the two faces reproduces the legacy
#   :func:`compute_P_esc` / :func:`compute_G_bc` (backwards-compat pin).
# - Solid cyl/sph (regime A, ``inner_radius == 0``): the "inner"
#   primitive returns an all-zero array (sentinel); the "outer"
#   primitive matches the legacy single-surface function bit-exactly.
# - Hollow cyl/sph (``inner_radius > 0``): the inner primitive must be
#   strictly positive in the annulus; regime B (``r_0 → 0⁺``) sends the
#   inner contribution and ``|outer − solid|`` to zero monotonically.
# ═══════════════════════════════════════════════════════════════════════


def _E2(tau):
    if tau > 0.0:
        return float(mpmath.expint(2, tau))
    return 1.0


@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSlabPescPerFace:
    r"""Per-face slab :math:`P_{\rm esc}` closed-form verification."""

    @pytest.mark.parametrize("L, sig_t", [
        (1.0, 1.0),
        (2.0, 0.5),
        (0.5, 2.0),
    ])
    def test_outer_face_matches_E2_at_machine_precision(self, L, sig_t):
        r"""Face :math:`x = L`:
        :math:`P_{\rm esc,L}(x_i) = \tfrac{1}{2} E_2(\Sigma_t(L - x_i))`."""
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
        P_out = compute_P_esc_outer(
            SLAB_POLAR_1D, r_nodes, radii, sig_t_arr,
            n_angular=32, dps=25,
        )
        for i, x_i in enumerate(r_nodes):
            expected = 0.5 * _E2(sig_t * (L - float(x_i)))
            rel = abs(P_out[i] - expected) / abs(expected)
            assert rel < 1e-14, (
                f"P_esc_outer(x={x_i:.4f}) rel={rel:.3e}"
            )

    @pytest.mark.parametrize("L, sig_t", [
        (1.0, 1.0),
        (2.0, 0.5),
        (0.5, 2.0),
    ])
    def test_inner_face_matches_E2_at_machine_precision(self, L, sig_t):
        r"""Face :math:`x = 0`:
        :math:`P_{\rm esc,0}(x_i) = \tfrac{1}{2} E_2(\Sigma_t x_i)`."""
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
        P_in = compute_P_esc_inner(
            SLAB_POLAR_1D, r_nodes, radii, sig_t_arr,
            n_angular=32, dps=25,
        )
        for i, x_i in enumerate(r_nodes):
            expected = 0.5 * _E2(sig_t * float(x_i))
            rel = abs(P_in[i] - expected) / abs(expected)
            assert rel < 1e-14, (
                f"P_esc_inner(x={x_i:.4f}) rel={rel:.3e}"
            )

    @pytest.mark.parametrize("L, sig_t", [
        (1.0, 1.0),
        (2.0, 0.5),
        (0.5, 2.0),
    ])
    def test_sum_equals_legacy_compute_P_esc(self, L, sig_t):
        r"""Backwards-compat pin: per-face sum reproduces
        :func:`compute_P_esc` exactly (the legacy single-surface function)."""
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
        P_tot = compute_P_esc(SLAB_POLAR_1D, r_nodes, radii, sig_t_arr)
        P_out = compute_P_esc_outer(SLAB_POLAR_1D, r_nodes, radii, sig_t_arr)
        P_in = compute_P_esc_inner(SLAB_POLAR_1D, r_nodes, radii, sig_t_arr)
        err = np.max(np.abs(P_tot - P_out - P_in))
        assert err < 1e-14, f"per-face sum vs legacy err={err:.3e}"


@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSlabGbcPerFace:
    r"""Per-face slab :math:`G_{\rm bc}` closed-form verification."""

    @pytest.mark.parametrize("L, sig_t", [
        (1.0, 1.0),
        (2.0, 0.5),
        (0.5, 2.0),
    ])
    def test_outer_face_matches_2E2_at_machine_precision(self, L, sig_t):
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
        G_out = compute_G_bc_outer(SLAB_POLAR_1D, r_nodes, radii, sig_t_arr)
        for i, x_i in enumerate(r_nodes):
            expected = 2.0 * _E2(sig_t * (L - float(x_i)))
            rel = abs(G_out[i] - expected) / abs(expected)
            assert rel < 1e-14

    @pytest.mark.parametrize("L, sig_t", [
        (1.0, 1.0),
        (2.0, 0.5),
        (0.5, 2.0),
    ])
    def test_inner_face_matches_2E2_at_machine_precision(self, L, sig_t):
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
        G_in = compute_G_bc_inner(SLAB_POLAR_1D, r_nodes, radii, sig_t_arr)
        for i, x_i in enumerate(r_nodes):
            expected = 2.0 * _E2(sig_t * float(x_i))
            rel = abs(G_in[i] - expected) / abs(expected)
            assert rel < 1e-14

    @pytest.mark.parametrize("L, sig_t", [
        (1.0, 1.0),
        (2.0, 0.5),
    ])
    def test_sum_equals_legacy_compute_G_bc(self, L, sig_t):
        radii = np.array([L])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
        G_tot = compute_G_bc(SLAB_POLAR_1D, r_nodes, radii, sig_t_arr)
        G_out = compute_G_bc_outer(SLAB_POLAR_1D, r_nodes, radii, sig_t_arr)
        G_in = compute_G_bc_inner(SLAB_POLAR_1D, r_nodes, radii, sig_t_arr)
        err = np.max(np.abs(G_tot - G_out - G_in))
        assert err < 1e-14, f"per-face G_bc sum vs legacy err={err:.3e}"


@pytest.mark.foundation
class TestSolidCylSphInnerZeroSentinel:
    r"""Regime-A sentinel: for solid cyl/sph the ``_inner`` primitives
    return all-zero arrays AND the ``_outer`` primitives match the
    legacy single-surface :func:`compute_P_esc` / :func:`compute_G_bc`
    bit-exactly. This is the contract that makes the rank-2
    :class:`BoundaryClosureOperator` reduce bit-exactly to rank-1 when
    ``inner_radius == 0``.
    """

    @pytest.mark.parametrize("geometry", [
        pytest.param(CYLINDER_1D, id="cylinder-1d"),
        pytest.param(SPHERE_1D, id="sphere-1d"),
    ])
    @pytest.mark.parametrize("R, sig_t", [
        (1.0, 1.5),
        (2.0, 0.5),
    ])
    def test_P_esc_inner_all_zero(self, geometry, R, sig_t):
        radii = np.array([R])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
        P_in = compute_P_esc_inner(geometry, r_nodes, radii, sig_t_arr)
        assert np.all(P_in == 0.0)

    @pytest.mark.parametrize("geometry", [
        pytest.param(CYLINDER_1D, id="cylinder-1d"),
        pytest.param(SPHERE_1D, id="sphere-1d"),
    ])
    @pytest.mark.parametrize("R, sig_t", [
        (1.0, 1.5),
        (2.0, 0.5),
    ])
    def test_G_bc_inner_all_zero(self, geometry, R, sig_t):
        radii = np.array([R])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
        G_in = compute_G_bc_inner(geometry, r_nodes, radii, sig_t_arr)
        assert np.all(G_in == 0.0)

    @pytest.mark.parametrize("geometry", [
        pytest.param(CYLINDER_1D, id="cylinder-1d"),
        pytest.param(SPHERE_1D, id="sphere-1d"),
    ])
    @pytest.mark.parametrize("R, sig_t", [
        (1.0, 1.5),
        (2.0, 0.5),
    ])
    def test_P_esc_outer_matches_legacy_bit_exact(self, geometry, R, sig_t):
        radii = np.array([R])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
        P_tot = compute_P_esc(geometry, r_nodes, radii, sig_t_arr)
        P_out = compute_P_esc_outer(geometry, r_nodes, radii, sig_t_arr)
        # Bit-exact (same code path, same GL nodes).
        assert np.array_equal(P_out, P_tot)

    @pytest.mark.parametrize("geometry", [
        pytest.param(CYLINDER_1D, id="cylinder-1d"),
        pytest.param(SPHERE_1D, id="sphere-1d"),
    ])
    @pytest.mark.parametrize("R, sig_t", [
        (1.0, 1.5),
        (2.0, 0.5),
    ])
    def test_G_bc_outer_matches_legacy_bit_exact(self, geometry, R, sig_t):
        radii = np.array([R])
        sig_t_arr = np.array([sig_t])
        r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
        G_tot = compute_G_bc(geometry, r_nodes, radii, sig_t_arr)
        G_out = compute_G_bc_outer(geometry, r_nodes, radii, sig_t_arr)
        assert np.array_equal(G_out, G_tot)


@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestHollowCylSphPerFacePositivityAndRegimeB:
    r"""Hollow cyl/sph per-face sanity + regime-B convergence.

    Regime B: as :math:`r_0 \to 0^+`, the hollow-cell per-face
    primitives must converge to the solid-geometry result. The inner
    contribution vanishes (geometric support in angle collapses), and
    the outer contribution approaches the solid value at an algebraic
    rate. This exposes numerical-stability bugs in the inner-intersection
    quadratic that the regime-A (``inner_radius == 0``) sentinel would
    hide.
    """

    @pytest.mark.parametrize("kind", ["cylinder-1d", "sphere-1d"])
    @pytest.mark.parametrize("r0", [0.3, 0.5])
    def test_hollow_per_face_primitives_strictly_positive(self, kind, r0):
        R = 1.0
        sig_t_val = 1.5
        geom = CurvilinearGeometry(kind=kind, inner_radius=r0)
        # Observer nodes strictly inside the annulus (r0 < r < R).
        r_nodes = np.array([r0 + 0.05, 0.5 * (r0 + R), R - 0.05])
        radii = np.array([R])
        sig_t = np.array([sig_t_val])
        P_out = compute_P_esc_outer(geom, r_nodes, radii, sig_t)
        P_in = compute_P_esc_inner(geom, r_nodes, radii, sig_t)
        G_out = compute_G_bc_outer(geom, r_nodes, radii, sig_t)
        G_in = compute_G_bc_inner(geom, r_nodes, radii, sig_t)
        assert np.all(P_out > 0.0)
        assert np.all(P_in > 0.0)
        assert np.all(G_out > 0.0)
        assert np.all(G_in > 0.0)

    @pytest.mark.parametrize("kind", ["cylinder-1d", "sphere-1d"])
    def test_regime_B_hollow_to_solid_monotone_convergence(self, kind):
        r"""For a sequence :math:`r_0 \in \{0.1, 0.01\} \cdot R`, the
        deviation from the solid result decreases strictly. At
        :math:`r_0 \le 10^{-3}` the 32-point GL rule can no longer
        resolve the tangent cone so the inner contribution rounds to
        exactly zero (numerical floor) — the convergence check is
        restricted to the regime where the primitive has visible
        support."""
        R = 1.0
        sig_t_val = 1.5
        # Observer nodes outside all tested r_0 values.
        r_nodes = np.array([0.4, 0.6, 0.8, 0.95])
        radii = np.array([R])
        sig_t = np.array([sig_t_val])
        solid = CurvilinearGeometry(kind=kind)
        P_solid = compute_P_esc_outer(solid, r_nodes, radii, sig_t)

        errs = []
        max_inners = []
        for r0 in (0.1, 0.01):
            g = CurvilinearGeometry(kind=kind, inner_radius=r0)
            P_out = compute_P_esc_outer(g, r_nodes, radii, sig_t)
            P_in = compute_P_esc_inner(g, r_nodes, radii, sig_t)
            errs.append(float(np.max(np.abs(P_out - P_solid))))
            max_inners.append(float(np.max(P_in)))

        # Monotone decrease: at least a factor-of-5 improvement per
        # decade for both outer-deviation and inner-magnitude.
        assert errs[1] < errs[0] / 5.0, f"outer err not decreasing: {errs}"
        assert max_inners[1] < max_inners[0] / 5.0, (
            f"inner magnitude not decreasing: {max_inners}"
        )
