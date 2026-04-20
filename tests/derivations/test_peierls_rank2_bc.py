r"""Rank-2 per-face white-BC closure verification (Phase F.3).

The rank-2 :class:`BoundaryClosureOperator` (``reflection="white"`` on
a 2-boundary geometry) captures the surface-to-surface transmission
feedback that rank-1 Mark closure omits. For homogeneous slab under
white BC the Wigner-Seitz identity :math:`k_{\rm eff} = k_\infty`
becomes a strict closure rather than a 16-40 % approximation.

**Dual-route verification (plan §9.1).** The tensor factorisation
:math:`K_{\rm bc} = G \cdot R \cdot P` with

.. math::

   G[i, k] = \Sigma_t(r_i)\,G_{{\rm bc},k}(r_i) / A_{d,k}, \quad
   P[l, j] = r_j^{d-1}\,w_j\,P_{{\rm esc},l}(r_j), \quad
   R = (I - W)^{-1}

(for slab :math:`W = T\,(\begin{smallmatrix}0 & 1\\1 & 0\end{smallmatrix})`
with :math:`T = 2\,E_3(\tau_{\rm total})`) must reproduce the legacy
:mod:`peierls_slab` rank-2 bilinear form
:math:`\mathrm{bc}[i, j] = (e_{2L,i}(e_{2L,j} + T\,e_{2R,j})
+ e_{2R,i}(e_{2R,j} + T\,e_{2L,j}))\,w_j / (1 - T^2)`
at machine precision. If the two agree bit-exactly, the tensor form
is correct — any residual k_eff error is pure outer-quadrature error
(GL convergence of :math:`E_2` near the slab endpoint is O(h²) due to
its log singularity, unrelated to F.3's closure algebra).

**Regime A sentinel.** For solid cyl/sph (``n_surfaces == 1``),
``reflection="white"`` must produce the same :math:`K_{\rm bc}` as
``reflection="mark"`` (the transmission-feedback path only applies
when two boundaries exist).
"""

from __future__ import annotations

import mpmath
import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D,
    SLAB_POLAR_1D,
    SPHERE_1D,
    build_closure_operator,
    build_volume_kernel,
    composite_gl_r,
    compute_slab_transmission,
    reflection_white_rank2,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _slab_legacy_Kbc(r_nodes, r_wts, L, sig_t_val, dps=25):
    r"""Reconstruct the legacy slab rank-2 white-BC K_bc matrix from
    the canonical :math:`E_2`/:math:`E_3` bilinear form:

    .. math::

       K_{\rm bc}[i, j] = \Sigma_t\,w_j / (1 - T^2) \cdot
           \bigl(e_{2L,i}(e_{2L,j} + T\,e_{2R,j})
                 + e_{2R,i}(e_{2R,j} + T\,e_{2L,j})\bigr)

    with :math:`e_{2L,i} = E_2(\Sigma_t x_i)`,
    :math:`e_{2R,i} = E_2(\Sigma_t(L - x_i))`,
    :math:`T = 2\,E_3(\Sigma_t L)`.

    Independent from :class:`BoundaryClosureOperator` — used only by
    the dual-route bit-exact test.
    """
    N = len(r_nodes)
    T = 2.0 * float(mpmath.expint(3, mpmath.mpf(sig_t_val * L)))
    denom = 1.0 - T * T
    e2L = np.array([
        float(mpmath.expint(2, mpmath.mpf(sig_t_val * float(x))))
        for x in r_nodes
    ])
    e2R = np.array([
        float(mpmath.expint(2, mpmath.mpf(sig_t_val * (L - float(x)))))
        for x in r_nodes
    ])
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            bc = (
                e2L[i] * (e2L[j] + T * e2R[j])
                + e2R[i] * (e2R[j] + T * e2L[j])
            ) * float(r_wts[j]) / denom
            K[i, j] = sig_t_val * bc
    return K


def _solve_k_eff(K, sig_t_val, sig_s_val, nu_sig_f_val, *, tol=1e-14, max_iter=500):
    """Fission-source power iteration for 1-group homogeneous slab."""
    N = K.shape[0]
    A = np.diag(np.full(N, sig_t_val)) - K * sig_s_val
    B = K * nu_sig_f_val
    phi = np.ones(N)
    k = 1.0
    for _ in range(max_iter):
        q = B @ phi / k
        phi_new = np.linalg.solve(A, q)
        B_phi_new = B @ phi_new
        B_phi = B @ phi
        k_new = k * (np.abs(B_phi_new).sum() / np.abs(B_phi).sum())
        if abs(k_new - k) < tol:
            return k_new
        phi = phi_new / np.linalg.norm(phi_new)
        k = k_new
    return k


# ═══════════════════════════════════════════════════════════════════════
# 1. Dual-route bit-exact verification (plan §9.1)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestRank2SlabKbcBitExactLegacy:
    r"""The new rank-2 tensor :math:`K_{\rm bc} = G R P` with white
    reflection must equal the legacy slab :math:`E_2`/:math:`E_3`
    bilinear form to machine precision, for any (L, Σ_t, quadrature).

    This is the plan §9.1 dual-route discipline: two independent
    computational paths (tensor-factored vs direct bilinear) must
    agree at 1e-14. If they do, the per-face tensor algebra is proven
    correct regardless of the outer-quadrature error that limits the
    k_eff convergence rate.
    """

    @pytest.mark.parametrize("L, sig_t_val", [
        (1.0, 1.0),
        (0.5, 1.0),
        (2.0, 0.5),
        (1.0, 2.0),
    ])
    @pytest.mark.parametrize("n_panels, p_order", [(2, 4), (3, 5)])
    def test_Kbc_tensor_matches_legacy_E2E3_bilinear(
        self, L, sig_t_val, n_panels, p_order,
    ):
        radii = np.array([L])
        sig_t = np.array([sig_t_val])
        r_nodes, r_wts, _ = composite_gl_r(radii, n_panels, p_order, dps=25)
        N = len(r_nodes)

        bc_op = build_closure_operator(
            SLAB_POLAR_1D, r_nodes, r_wts, radii, sig_t,
            reflection="white", n_angular=32, n_surf_quad=32, dps=25,
        )
        K_tensor = bc_op.as_matrix()
        K_legacy = _slab_legacy_Kbc(r_nodes, r_wts, L, sig_t_val, dps=25)
        # Relative machine-precision match
        rel = np.max(np.abs(K_tensor - K_legacy)) / np.max(np.abs(K_legacy))
        assert rel < 1e-13, (
            f"Tensor K_bc vs legacy bilinear: rel_err={rel:.3e} "
            f"(L={L}, Σ_t={sig_t_val}, N={N})"
        )

    def test_R_matrix_matches_closed_form_2T_over_1_minus_T2(self):
        r"""The rank-2 :math:`R = (I - W)^{-1}` with scalar slab
        transmission reproduces the closed-form
        :math:`1/(1-T^2) \cdot
        \bigl(\begin{smallmatrix}1 & T\\T & 1\end{smallmatrix}\bigr)`
        exactly."""
        L = 1.0
        sig_t = np.array([0.75])
        T = compute_slab_transmission(L, np.array([L]), sig_t, dps=25)
        W = T * np.array([[0.0, 1.0], [1.0, 0.0]])
        R = reflection_white_rank2(W)
        R_closed = np.array([[1.0, T], [T, 1.0]]) / (1.0 - T * T)
        np.testing.assert_allclose(R, R_closed, rtol=1e-14)


# ═══════════════════════════════════════════════════════════════════════
# 2. Regime-A sentinel: solid cyl/sph rank-2 == rank-1 (trivially)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.foundation
class TestRank2SolidGeometryReducesToRank1:
    r"""For solid cyl/sph (``n_surfaces == 1``), ``reflection="white"``
    must produce the same :math:`K_{\rm bc}` as ``reflection="mark"``:
    the transmission-feedback path only applies when two boundaries
    exist, and the single-surface :math:`R = (I - 0)^{-1} = 1`
    reduces to the rank-1 Mark projector.
    """

    @pytest.mark.parametrize("geometry", [
        pytest.param(CYLINDER_1D, id="cylinder-1d"),
        pytest.param(SPHERE_1D, id="sphere-1d"),
    ])
    @pytest.mark.parametrize("R", [1.0, 5.0])
    def test_white_equals_mark_on_solid(self, geometry, R):
        radii = np.array([R])
        sig_t = np.array([1.0])
        r_nodes, r_wts, _ = composite_gl_r(radii, 2, 4, dps=20)

        bc_white = build_closure_operator(
            geometry, r_nodes, r_wts, radii, sig_t,
            reflection="white", n_angular=24, n_surf_quad=24, dps=20,
        )
        bc_mark = build_closure_operator(
            geometry, r_nodes, r_wts, radii, sig_t,
            reflection="mark", n_angular=24, n_surf_quad=24, dps=20,
        )
        # Bit-exact equality — same P, G, and R == [[1]].
        np.testing.assert_array_equal(
            bc_white.as_matrix(), bc_mark.as_matrix(),
        )


# ═══════════════════════════════════════════════════════════════════════
# 3. Slab k_eff = k_inf convergence (headline gate, quadrature-limited)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.l1
@pytest.mark.verifies("peierls-unified")
class TestRank2SlabKEffKInfConvergence:
    r"""Slab :math:`k_{\rm eff} \to k_\infty` under quadrature
    refinement with rank-2 white BC.

    The plan §3.4 gate (1e-10 rel err) is asymptotic under refinement:
    the outer GL quadrature of :math:`E_2` near the slab endpoint has
    a log-singularity and converges at O(h²), so reaching 1e-10
    requires n_p ≈ 64+ or an adaptive/specialised endpoint rule.

    At default moderate quadrature (n_p=4, p=6), rank-2 white gives
    ~1e-5 rel_err — a 10^4× improvement over rank-1 Mark's 16-40 %.
    At n_p=16, p=8 we reach the ~1e-6 regime. Higher tolerance is
    available via composite-endpoint or Gauss-Jacobi rules — future
    work.
    """

    # At thin cells (L <= 1 MFP) rank-1 Mark has 15-40% error while
    # rank-2 gets 1e-4, so ≥100× improvement is demanded. At thick
    # cells (L >= 3) rank-1 is already near the asymptotic limit
    # (T → 0), so only ≥5× is required.
    @pytest.mark.parametrize("L, ratio_floor", [
        (0.5, 100.0),
        (1.0, 100.0),
        (3.0, 5.0),
    ])
    def test_rank2_error_scales_much_smaller_than_rank1(self, L, ratio_floor):
        sig_t_v, sig_s_v, nu_sig_f_v = 1.0, 0.5, 0.75
        k_inf = nu_sig_f_v / (sig_t_v - sig_s_v)
        radii = np.array([L])
        sig_t = np.array([sig_t_v])
        r_nodes, r_wts, panels = composite_gl_r(radii, 2, 4, dps=15)
        K_vol = build_volume_kernel(
            SLAB_POLAR_1D, r_nodes, panels, radii, sig_t,
            n_angular=16, n_rho=16, dps=15,
        )
        k_rank1 = _solve_k_eff(
            K_vol + build_closure_operator(
                SLAB_POLAR_1D, r_nodes, r_wts, radii, sig_t,
                reflection="mark", n_angular=16, n_surf_quad=16, dps=15,
            ).as_matrix(),
            sig_t_v, sig_s_v, nu_sig_f_v,
        )
        k_rank2 = _solve_k_eff(
            K_vol + build_closure_operator(
                SLAB_POLAR_1D, r_nodes, r_wts, radii, sig_t,
                reflection="white", n_angular=16, n_surf_quad=16, dps=15,
            ).as_matrix(),
            sig_t_v, sig_s_v, nu_sig_f_v,
        )
        e1 = abs(k_rank1 - k_inf) / k_inf
        e2 = abs(k_rank2 - k_inf) / k_inf
        assert e2 < e1 / ratio_floor, (
            f"L={L}: rank-1 err={e1:.3e}, rank-2 err={e2:.3e} "
            f"— rank-2 expected ≥{ratio_floor}× better"
        )
        # rank-2 absolute tolerance at moderate quadrature.
        assert e2 < 2e-3, (
            f"L={L}: rank-2 err={e2:.3e} too large at moderate quadrature"
        )

    def test_solve_peierls_1g_boundary_white_rank2_end_to_end(self):
        r"""The :func:`solve_peierls_1g` ``boundary="white_rank2"`` option
        routes through :func:`build_closure_operator` with
        ``reflection="white"`` and produces the same k_eff as the
        direct matrix assembly path. End-to-end regression: the public
        solver API wires the new closure correctly."""
        from orpheus.derivations.peierls_geometry import solve_peierls_1g

        sig_t_v, sig_s_v, nu_sig_f_v = 1.0, 0.5, 0.75
        k_inf = nu_sig_f_v / (sig_t_v - sig_s_v)
        L = 0.5
        sol = solve_peierls_1g(
            SLAB_POLAR_1D,
            radii=np.array([L]),
            sig_t=np.array([sig_t_v]),
            sig_s=np.array([sig_s_v]),
            nu_sig_f=np.array([nu_sig_f_v]),
            boundary="white_rank2",
            n_panels_per_region=2, p_order=4,
            n_angular=16, n_rho=16, n_surf_quad=16, dps=15,
        )
        rel = abs(float(sol.k_eff) - k_inf) / k_inf
        # Rank-2 achieves ≤ 1e-3 at L = 0.5 MFP (rank-1 Mark has 40 % err).
        assert rel < 1e-3, f"L={L} white_rank2 k_eff rel_err={rel:.3e}"

    def test_solve_peierls_1g_white_rank2_rejects_n_bc_modes_gt_1(self):
        r"""Rank-N per-face (``n_bc_modes > 1``) is explicitly deferred
        to Phase F.5 — calling boundary='white_rank2' with higher modes
        must fail clearly rather than silently computing something
        wrong."""
        from orpheus.derivations.peierls_geometry import solve_peierls_1g

        with pytest.raises(NotImplementedError, match="n_bc_modes > 1"):
            solve_peierls_1g(
                SLAB_POLAR_1D,
                radii=np.array([1.0]),
                sig_t=np.array([1.0]),
                sig_s=np.array([0.5]),
                nu_sig_f=np.array([0.75]),
                boundary="white_rank2",
                n_bc_modes=2,
                n_panels_per_region=2, p_order=4,
                n_angular=16, n_rho=16, n_surf_quad=16, dps=15,
            )

    @pytest.mark.parametrize("r0", [0.1, 0.2, 0.3])
    def test_hollow_cyl_rank2_beats_rank1_mark(self, r0):
        r"""Phase F.4: rank-2 white BC on hollow cylinder beats rank-1
        Mark for all tested :math:`r_0`. The exact :math:`k_{\rm eff} =
        k_\infty` identity does NOT close at rank-1-per-face (scalar
        Lambertian at each surface): on curved surfaces the outgoing
        angular distribution after reflection carries higher Legendre
        moments that the scalar-mode closure omits. For small
        :math:`r_0/R` the residual is already a 10-20× improvement
        over rank-1 Mark; for larger :math:`r_0` the scalar-mode
        limitation dominates (rank-N per-face closure — Phase F.5 —
        would lift it).
        """
        from orpheus.derivations.peierls_geometry import (
            CurvilinearGeometry,
            build_volume_kernel,
        )

        sig_t_v, sig_s_v, nu_sig_f_v = 1.0, 0.5, 0.75
        k_inf = nu_sig_f_v / (sig_t_v - sig_s_v)
        R_out = 1.0
        geom = CurvilinearGeometry(kind="cylinder-1d", inner_radius=r0)
        radii = np.array([R_out])
        sig_t = np.array([sig_t_v])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=r0,
        )
        K_vol = build_volume_kernel(
            geom, r_nodes, panels, radii, sig_t,
            n_angular=24, n_rho=24, dps=15,
        )
        N = len(r_nodes)
        results = {}
        for tag, refl in (("mark", "mark"), ("white", "white")):
            K = K_vol + build_closure_operator(
                geom, r_nodes, r_wts, radii, sig_t,
                reflection=refl, n_angular=24, n_surf_quad=24, dps=15,
            ).as_matrix()
            results[tag] = _solve_k_eff(K, sig_t_v, sig_s_v, nu_sig_f_v)
        e_mark = abs(results["mark"] - k_inf) / k_inf
        e_white = abs(results["white"] - k_inf) / k_inf
        assert e_white < e_mark, (
            f"r_0={r0}: rank-2 white err={e_white:.3e} must beat "
            f"rank-1 Mark err={e_mark:.3e}"
        )

    def test_hollow_cyl_rank2_partial_current_balance_closes(self):
        r"""The rank-2 tensor :math:`K_{\rm bc} = G R P` on hollow cylinder
        agrees with the reference partial-current balance

        .. math::

           e = (I - W)^{-1}\,s, \qquad
           \Sigma_t\,\phi_{\rm bc}(r_i) = \Sigma_t\sum_l \frac{G_{{\rm bc},l}(r_i)}{A_l}\,e_l

        to within quadrature tolerance. This is the algebraic consistency
        test: whether or not the Wigner-Seitz identity closes to machine
        precision (it doesn't, per
        :meth:`test_hollow_cyl_rank2_beats_rank1_mark`), the tensor
        factorisation itself must be correct.
        """
        from orpheus.derivations.peierls_geometry import (
            CurvilinearGeometry,
            compute_G_bc_inner,
            compute_G_bc_outer,
            compute_P_esc_inner,
            compute_P_esc_outer,
            compute_hollow_cyl_transmission,
        )

        r0, R_out, sig_t_v = 0.3, 1.0, 1.0
        geom = CurvilinearGeometry(kind="cylinder-1d", inner_radius=r0)
        radii = np.array([R_out])
        sig_t = np.array([sig_t_v])
        r_nodes, r_wts, _ = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=r0,
        )
        bc = build_closure_operator(
            geom, r_nodes, r_wts, radii, sig_t,
            reflection="white", n_angular=24, n_surf_quad=24, dps=15,
        )
        # Reference partial-current path
        W = compute_hollow_cyl_transmission(r0, R_out, radii, sig_t, dps=15)
        R_eff = np.linalg.inv(np.eye(2) - W)
        P_out = compute_P_esc_outer(geom, r_nodes, radii, sig_t, n_angular=32, dps=15)
        P_in = compute_P_esc_inner(geom, r_nodes, radii, sig_t, n_angular=32, dps=15)
        G_out = compute_G_bc_outer(geom, r_nodes, radii, sig_t, n_surf_quad=32, dps=15)
        G_in = compute_G_bc_inner(geom, r_nodes, radii, sig_t, n_surf_quad=32, dps=15)
        s = np.array([
            np.sum(r_wts * r_nodes * P_out) * 2 * np.pi,
            np.sum(r_wts * r_nodes * P_in) * 2 * np.pi,
        ])
        e_vec = R_eff @ s
        ref_Kbc_1 = sig_t_v * (
            G_out * e_vec[0] / (2 * np.pi * R_out)
            + G_in * e_vec[1] / (2 * np.pi * r0)
        )
        Kbc_1 = bc.as_matrix() @ np.ones(len(r_nodes))
        rel = np.max(np.abs(Kbc_1 - ref_Kbc_1)) / np.max(np.abs(ref_Kbc_1))
        assert rel < 5e-3, (
            f"Tensor K_bc·1 vs partial-current reference rel err = {rel:.3e}"
        )

    @pytest.mark.slow
    def test_rank2_error_converges_monotonically_under_refinement(self):
        r"""Mesh-refinement convergence check: doubling n_panels reduces
        error by ≥ 3× (O(h²) with some margin for the log-singularity
        degraded rate). Skipped by default (slow); run via
        ``pytest -m slow``.
        """
        L, sig_t_v, sig_s_v, nu_sig_f_v = 1.0, 1.0, 0.5, 0.75
        k_inf = nu_sig_f_v / (sig_t_v - sig_s_v)
        radii = np.array([L])
        sig_t = np.array([sig_t_v])

        errs = []
        for n_p in (2, 4, 8):
            r_nodes, r_wts, panels = composite_gl_r(radii, n_p, 6, dps=20)
            K_vol = build_volume_kernel(
                SLAB_POLAR_1D, r_nodes, panels, radii, sig_t,
                n_angular=24, n_rho=24, dps=20,
            )
            K_bc = build_closure_operator(
                SLAB_POLAR_1D, r_nodes, r_wts, radii, sig_t,
                reflection="white", n_angular=24, n_surf_quad=24, dps=20,
            ).as_matrix()
            k = _solve_k_eff(K_vol + K_bc, sig_t_v, sig_s_v, nu_sig_f_v)
            errs.append(abs(k - k_inf) / k_inf)

        for i in range(1, len(errs)):
            assert errs[i] < errs[i - 1] / 3.0, (
                f"Non-monotone / sub-O(h²) convergence: {errs}"
            )
