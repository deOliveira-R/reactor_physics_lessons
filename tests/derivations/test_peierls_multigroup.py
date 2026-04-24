r"""Multi-group Peierls eigenvalue driver (Issue #104).

:func:`~orpheus.derivations.peierls_geometry.solve_peierls_mg`
generalises :func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g`
to ``ng ≥ 1`` groups with downscatter / upscatter coupling and
χ-weighted fission.

The verification strategy has two tiers:

1. **ng = 1 bit-match regression** (this module). The MG path with
   ``ng = 1`` and synthesised ``χ = 1`` must reproduce the legacy 1G
   k_eff and every flux value *bit-exactly* (0.0 diff, not just 1e-12).
   The :func:`solve_peierls_1g` wrapper enforces this equality by
   construction — if it ever breaks, every downstream 1G caller is
   affected, so these tests are the gate that keeps the wrapper safe.

2. **2G parity** against reference solvers (``cp_cylinder`` /
   ``cp_sphere`` native 2G path; future ``peierls_slab`` 2G via
   Issue #130 Phase G.5). Lives in separate test files because the 2G
   references require XS-library data and the curvilinear references
   are still being registered in
   :func:`~orpheus.derivations.peierls_cases._class_a_cases`.

This file is the tier-1 regression gate. It must run cheaply (tens of
seconds) and on every push.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D,
    CurvilinearGeometry,
    PeierlsSolution,
    SLAB_POLAR_1D,
    SPHERE_1D,
    solve_peierls_1g,
    solve_peierls_mg,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _call_mg_ng1(
    geometry: CurvilinearGeometry,
    radii: np.ndarray,
    sig_t: np.ndarray,
    sig_s: np.ndarray,
    nu_sig_f: np.ndarray,
    **kwargs,
) -> PeierlsSolution:
    """Direct call into :func:`solve_peierls_mg` with ng=1 arrays.

    Mirrors what :func:`solve_peierls_1g` does internally but without
    the wrapper, so the test compares apples to apples (both paths
    enter ``solve_peierls_mg`` with the same shapes).
    """
    n_regions = len(sig_t)
    return solve_peierls_mg(
        geometry,
        radii,
        sig_t[:, np.newaxis],
        sig_s.reshape(n_regions, 1, 1),
        nu_sig_f[:, np.newaxis],
        np.ones((n_regions, 1)),
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════
# Tier 1 — ng=1 bit-match regression
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestMGNg1BitMatch1G:
    r"""``solve_peierls_mg(ng=1)`` must bit-exactly reproduce
    ``solve_peierls_1g`` on k_eff AND on every flux value.

    "Bit-exact" here means numerical zero difference (``diff == 0.0``),
    not a tolerance. The ng=1 MG path with ``chi = 1`` is algebraically
    identical to the original 1G assembly:

    - Single K per group ⇒ ``K_per_group[0] = K_1G``.
    - Diagonal Σ_t and scatter blocks reduce to the 1G ``diag(Σ_t) − K·Σ_s``.
    - Fission block reduces to ``K · 1 · νΣ_f`` (χ absorbed into unit).
    - Power iteration uses the same init (``phi = 1``, ``k = 1``) and
      the same convergence predicate.

    Any non-zero diff indicates a bug in either the MG path or the
    :func:`solve_peierls_1g` wrapper. The 1G wrapper is the only entry
    point for ~30 downstream callers (case builders, rank-N diagnostics,
    every existing ``solve_peierls_*_1g`` wrapper in the shape-specific
    modules), so this test gates the safety of a multi-module refactor.
    """

    @pytest.mark.parametrize("geometry_name, radii, inner_radius", [
        # (name, radii, inner_radius) — exercised on three geometries.
        pytest.param("slab-polar", np.array([1.0]), 0.0, id="slab-polar"),
        pytest.param("cylinder-1d", np.array([1.0]), 0.0, id="solid-cyl"),
        pytest.param("sphere-1d", np.array([1.0]), 0.0, id="solid-sph"),
    ])
    @pytest.mark.parametrize("boundary", ["vacuum", "white_rank1_mark"])
    def test_bitmatch_k_eff_and_flux(
        self, geometry_name, radii, inner_radius, boundary,
    ):
        if geometry_name == "slab-polar":
            geometry = SLAB_POLAR_1D
        elif geometry_name == "cylinder-1d":
            geometry = (
                CYLINDER_1D
                if inner_radius == 0.0
                else CurvilinearGeometry(
                    kind="cylinder-1d", inner_radius=inner_radius,
                )
            )
        else:  # sphere-1d
            geometry = (
                SPHERE_1D
                if inner_radius == 0.0
                else CurvilinearGeometry(
                    kind="sphere-1d", inner_radius=inner_radius,
                )
            )

        sig_t = np.array([1.0])
        sig_s = np.array([0.4])
        nu_sig_f = np.array([0.6])

        kwargs = dict(
            boundary=boundary,
            n_panels_per_region=1,
            p_order=3,
            n_angular=16,
            n_rho=16,
            n_surf_quad=16,
            dps=20,
            max_iter=200,
            tol=1e-10,
        )

        sol_1g = solve_peierls_1g(geometry, radii, sig_t, sig_s, nu_sig_f, **kwargs)
        sol_mg = _call_mg_ng1(geometry, radii, sig_t, sig_s, nu_sig_f, **kwargs)

        # Bit-exact k_eff (diff == 0, not tolerance). Both paths feed
        # the same A, B, same power iteration, same floating-point ops.
        assert sol_1g.k_eff == sol_mg.k_eff, (
            f"ng=1 MG path diverged from 1G on k_eff: "
            f"1G={sol_1g.k_eff!r}, MG={sol_mg.k_eff!r} "
            f"(geometry={geometry.kind}, boundary={boundary})"
        )

        # Bit-exact every phi value.
        assert sol_1g.phi_values.shape == sol_mg.phi_values.shape
        np.testing.assert_array_equal(
            sol_1g.phi_values, sol_mg.phi_values,
            err_msg=(
                f"ng=1 MG path diverged from 1G on flux "
                f"(geometry={geometry.kind}, boundary={boundary})"
            ),
        )

        # Sanity — the result dataclass records ng=1.
        assert sol_mg.n_groups == 1
        assert sol_1g.n_groups == 1

    def test_bitmatch_hollow_cyl_f4_closure(self):
        r"""Exercise the ``white_f4`` (rank-2 per-face) closure on a
        hollow cylinder — the most complex closure path. Bit-match the
        wrapper.
        """
        geometry = CurvilinearGeometry(kind="cylinder-1d", inner_radius=0.1)
        radii = np.array([1.0])
        sig_t = np.array([1.0])
        sig_s = np.array([0.4])
        nu_sig_f = np.array([0.6])

        kwargs = dict(
            boundary="white_f4",
            n_panels_per_region=1,
            p_order=3,
            n_angular=16,
            n_rho=16,
            n_surf_quad=16,
            dps=20,
            max_iter=200,
            tol=1e-10,
        )

        sol_1g = solve_peierls_1g(geometry, radii, sig_t, sig_s, nu_sig_f, **kwargs)
        sol_mg = _call_mg_ng1(geometry, radii, sig_t, sig_s, nu_sig_f, **kwargs)

        assert sol_1g.k_eff == sol_mg.k_eff, (
            f"hollow-cyl F.4 closure diverged: 1G={sol_1g.k_eff!r}, "
            f"MG={sol_mg.k_eff!r}"
        )
        np.testing.assert_array_equal(sol_1g.phi_values, sol_mg.phi_values)


# ═══════════════════════════════════════════════════════════════════════
# Tier 1b — MG API validation
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.foundation
class TestMGInputValidation:
    """``solve_peierls_mg`` rejects mis-shaped XS arrays with a clear
    error message. Protects against silent bugs if a caller forgets
    to reshape from 1-D (1G) to 2-D (MG) arrays.
    """

    def test_sig_t_1d_rejected(self):
        with pytest.raises(ValueError, match=r"sig_t must be shape \(n_regions, ng\)"):
            solve_peierls_mg(
                SLAB_POLAR_1D,
                np.array([1.0]),
                np.array([1.0]),                   # 1-D — wrong
                np.array([[[0.4]]]),
                np.array([[0.6]]),
                np.array([[1.0]]),
            )

    def test_sig_s_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="sig_s must be shape"):
            solve_peierls_mg(
                SLAB_POLAR_1D,
                np.array([1.0]),
                np.array([[1.0, 1.0]]),             # ng=2
                np.array([[0.4]]),                  # (1,1) — wrong, should be (1,2,2)
                np.array([[0.6, 0.6]]),
                np.array([[0.5, 0.5]]),
            )

    def test_nu_sig_f_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="nu_sig_f must be shape"):
            solve_peierls_mg(
                SLAB_POLAR_1D,
                np.array([1.0]),
                np.array([[1.0]]),
                np.array([[[0.4]]]),
                np.array([0.6]),                    # 1-D — wrong
                np.array([[1.0]]),
            )

    def test_chi_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="chi must be shape"):
            solve_peierls_mg(
                SLAB_POLAR_1D,
                np.array([1.0]),
                np.array([[1.0]]),
                np.array([[[0.4]]]),
                np.array([[0.6]]),
                np.array([1.0]),                    # 1-D — wrong
            )


# ═══════════════════════════════════════════════════════════════════════
# Tier 1c — ng=2 sanity
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestMGNg2Sanity:
    r"""Smoke test: ng=2 runs to completion and produces a plausible
    k_eff / flux. No cross-reference here — that's tier 2 (parity
    against ``cp_cylinder`` / ``cp_sphere``).

    Uses a hand-crafted 2G XS set where the two groups are physically
    identical (same Σ_t, Σ_s_self, νΣ_f, no downscatter, χ split
    50/50). This configuration decouples the two groups — the total
    MG k_eff must equal the ng=1 k_eff with the same per-group XS,
    because each group is an independent copy of the 1G problem.
    """

    def test_decoupled_2g_k_eff_equals_1g(self):
        r"""Two independent identical groups ⇒ same k_eff as one group.

        With no scatter between groups and χ split 50/50, the coupled
        ng=2 eigenvalue problem block-diagonalises into two
        independent 1G problems. Each has the same k_eff = k_1G, so
        the MG k_eff matches k_1G.
        """
        radii = np.array([1.0])

        sig_t_1g = np.array([1.0])
        sig_s_1g = np.array([0.4])
        nu_f_1g = np.array([0.6])

        kwargs = dict(
            boundary="vacuum",
            n_panels_per_region=1,
            p_order=3,
            n_angular=16,
            n_rho=16,
            n_surf_quad=16,
            dps=20,
            max_iter=200,
            tol=1e-10,
        )

        sol_1g = solve_peierls_1g(
            SLAB_POLAR_1D, radii, sig_t_1g, sig_s_1g, nu_f_1g, **kwargs,
        )

        # 2G decoupled: Σ_s is diagonal, no group-to-group scatter; χ
        # split 50/50 across two identical groups.
        sig_t_mg = np.tile(sig_t_1g[:, None], (1, 2))           # (1, 2)
        sig_s_mg = np.zeros((1, 2, 2))
        sig_s_mg[0, 0, 0] = sig_s_1g[0]
        sig_s_mg[0, 1, 1] = sig_s_1g[0]
        nu_f_mg = np.tile(nu_f_1g[:, None], (1, 2))             # (1, 2)
        chi_mg = np.full((1, 2), 0.5)                           # equal χ per group

        sol_mg = solve_peierls_mg(
            SLAB_POLAR_1D, radii, sig_t_mg, sig_s_mg, nu_f_mg, chi_mg,
            **kwargs,
        )

        assert sol_mg.n_groups == 2
        assert sol_mg.phi_values.shape == (sol_mg.n_quad_r, 2)
        # Decoupled problem — spectral radius of the 2G operator equals
        # the spectral radius of the 1G operator.
        rel = abs(sol_mg.k_eff - sol_1g.k_eff) / abs(sol_1g.k_eff)
        assert rel < 1e-10, (
            f"Decoupled 2G k_eff should match 1G k_eff: "
            f"1G={sol_1g.k_eff:.12f}, 2G={sol_mg.k_eff:.12f}, "
            f"rel_diff={rel:.3e}"
        )

    def test_downscatter_2g_runs_and_k_eff_sensible(self):
        r"""Sanity: 2G with physical downscatter produces k_eff > 0
        and flux > 0 everywhere. Not a parity test — just that the
        driver does not blow up.

        Convention (matches the XS library and ``solve_peierls_mg``
        docstring): ``sig_s[r, g_src, g_dst]``. Under this convention
        downscatter (fast → thermal, group 0 → group 1) sits in the
        upper-triangular entry ``sig_s[r, 0, 1]``.
        """
        radii = np.array([1.0])
        sig_t_mg = np.array([[1.2, 1.0]])                       # fast, thermal
        sig_s_mg = np.array([[[0.3, 0.4],                       # from fast: self 0.3, down 0.4
                              [0.0, 0.5]]])                     # from thermal: no up, self 0.5
        nu_f_mg = np.array([[0.1, 0.5]])                        # mostly thermal fission
        chi_mg = np.array([[0.95, 0.05]])                       # fission births mostly fast

        sol = solve_peierls_mg(
            SLAB_POLAR_1D, radii, sig_t_mg, sig_s_mg, nu_f_mg, chi_mg,
            boundary="vacuum",
            n_panels_per_region=1, p_order=3,
            n_angular=16, n_rho=16, n_surf_quad=16,
            dps=20, max_iter=300, tol=1e-10,
        )

        assert sol.n_groups == 2
        assert sol.k_eff > 0.0, f"k_eff = {sol.k_eff} must be positive"
        assert sol.k_eff < 10.0, f"k_eff = {sol.k_eff} is physically implausible"
        assert np.all(np.isfinite(sol.phi_values))
        # Fundamental eigenmode: all flux entries should have a
        # consistent sign after the power iteration.
        assert np.all(sol.phi_values >= 0) or np.all(sol.phi_values <= 0), (
            f"Flux sign pattern is not monochromatic: {sol.phi_values}"
        )
# ═══════════════════════════════════════════════════════════════════════
# Tier 2 — 2G parity against native slab driver (Phase G.5 tie-back)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.l1
@pytest.mark.slow
@pytest.mark.verifies("peierls-unified")
class TestMGSlabPolarMatchesNativeSlabMG:
    r"""Phase G.5 tie-back (Issue #130): the unified multi-group slab
    path (:func:`solve_peierls_mg(SLAB_POLAR_1D, ...)`) must agree
    with :func:`peierls_slab.solve_peierls_eigenvalue` on k_eff and
    flux shape for the shipped 2G XS set.

    This is the **definitive cross-check** on the sig_s convention:
    if the two drivers agree on the 2G eigenvalue, the per-region
    ``sig_s[g_src, g_dst]`` indexing in :func:`solve_peierls_mg` is
    correct (the native slab driver predates Issue #104 and has
    always used the same indexing pattern internally).

    Marked ``@pytest.mark.slow`` — the unified adaptive path is ~100×
    slower than the native E₁ Nyström at matched precision. 1G
    baseline cost at ``N = 3`` is ~30 s; 2G ~60 s.
    """

    def test_2g_vacuum_slab_matches_native_eigenvalue(self):
        from orpheus.derivations.peierls_geometry import (
            SLAB_POLAR_1D, solve_peierls_mg,
        )
        from orpheus.derivations.peierls_slab import solve_peierls_eigenvalue

        # Use a simple fabricated 2G XS set — values chosen to give a
        # healthy, well-conditioned eigenproblem (k_eff ~ O(1)) and to
        # exercise downscatter (sig_s[0, 1] = 0.1, fast → thermal).
        L = 1.0
        sig_t_region = np.array([0.8, 1.2])            # (ng,)
        sig_s_region = np.array([[0.3, 0.2],           # (ng, ng)
                                 [0.0, 0.7]])          # convention: [src, dst]
        nu_sig_f_region = np.array([0.1, 0.6])         # (ng,)
        chi_region = np.array([0.9, 0.1])              # (ng,)

        n_panels, p_order, dps = 1, 3, 20

        # Unified MG path
        sol_unified = solve_peierls_mg(
            SLAB_POLAR_1D,
            radii=np.array([L]),
            sig_t=sig_t_region[np.newaxis, :],          # (1, ng)
            sig_s=sig_s_region[np.newaxis, :, :],       # (1, ng, ng)
            nu_sig_f=nu_sig_f_region[np.newaxis, :],
            chi=chi_region[np.newaxis, :],
            boundary="vacuum",
            n_panels_per_region=n_panels,
            p_order=p_order,
            dps=dps,
            tol=1e-10,
        )

        # Native slab MG driver (block-Toeplitz E₁ Nyström)
        sol_native = solve_peierls_eigenvalue(
            sig_t_regions=[sig_t_region],
            sig_s_matrices=[sig_s_region],
            nu_sig_f_all=[nu_sig_f_region],
            chi_all=[chi_region],
            thicknesses=[L],
            n_panels_per_region=n_panels,
            p_order=p_order,
            precision_digits=dps,
            boundary="vacuum",
        )

        k_uni, k_nat = float(sol_unified.k_eff), float(sol_native.k_eff)
        rel = abs(k_uni - k_nat) / abs(k_nat)

        # Target: 1e-8. The two drivers use different quadrature
        # (adaptive tanh-sinh vs E₁ Nyström) so agreement is limited
        # by the worse of the two at matched nominal precision.
        assert rel < 1e-8, (
            f"2G unified slab vs native disagreement: "
            f"unified={k_uni:.12f}, native={k_nat:.12f}, rel={rel:.3e}\n"
            f"This is a conventions-mismatch smoking-gun — if >1e-4, "
            f"check the sig_s ordering in solve_peierls_mg."
        )

        # Flux shape per group — both solvers produce shape (N, 2).
        phi_u = np.asarray(sol_unified.phi_values)
        phi_n = np.asarray(sol_native.phi_values)
        assert phi_u.shape == phi_n.shape, (
            f"Shape mismatch: unified {phi_u.shape}, native {phi_n.shape}"
        )
        for g in range(2):
            pu = phi_u[:, g] / np.max(np.abs(phi_u[:, g]))
            pn = phi_n[:, g] / np.max(np.abs(phi_n[:, g]))
            if np.dot(pu, pn) < 0:
                pn = -pn
            max_rel = float(np.max(np.abs(pu - pn)))
            assert max_rel < 1e-6, (
                f"Flux shape group {g}: ||φ_unified - φ_native||_∞ = "
                f"{max_rel:.3e}"
            )


# ═══════════════════════════════════════════════════════════════════════
# Tier 2 — Registration smoke test for 2G hollow cyl / sph
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.l1
@pytest.mark.slow
@pytest.mark.verifies("peierls-unified")
class TestMG2GHollowRegistration:
    r"""The 2G hollow cyl / sph continuous references registered in
    :func:`peierls_cases._class_a_cases` are **buildable** and
    produce finite, positive k_eff and finite flux.

    This is a slow integration test: each reference rebuilds the
    continuous case (adaptive ``mpmath.quad`` per K element, 2
    groups). Expected wall time ~2 min per reference; 6 references
    ⇒ ~12 min. Marked ``@pytest.mark.slow``.

    The test does NOT check k_eff parity against an independent
    solver — that's outside the scope of commit 2. Parity against
    ``cp_cylinder`` / ``cp_sphere`` discrete MG solvers is future
    work (Issue #104 AC requires 1 % agreement at thick R).
    """

    @pytest.mark.parametrize("r0_over_R", [0.1, 0.2, 0.3])
    def test_hollow_cyl_2g_builds(self, r0_over_R):
        from orpheus.derivations.peierls_cylinder import (
            _build_peierls_cylinder_hollow_f4_case,
        )
        # Tight quadrature — default quadrature would make this test
        # prohibitive. The goal is "does the builder run to completion
        # and produce physical-looking output", not full precision.
        ref = _build_peierls_cylinder_hollow_f4_case(
            r0_over_R=r0_over_R, ng_key="2g",
            n_panels_per_region=1, p_order=3,
            n_beta=12, n_rho=12, n_phi=12,
            precision_digits=15,
        )
        assert ref.problem.n_groups == 2
        assert ref.k_eff is not None and ref.k_eff > 0.0
        assert np.isfinite(ref.k_eff) and ref.k_eff < 100.0
        # The 2G name layout is peierls_cyl1D_hollow_{ng}eg_{n_regions}rg_r0_{r0_tag}
        assert "2eg" in ref.name
        assert ref.operator_form == "integral-peierls"

    @pytest.mark.parametrize("r0_over_R", [0.1, 0.2, 0.3])
    def test_hollow_sph_2g_builds(self, r0_over_R):
        from orpheus.derivations.peierls_sphere import (
            _build_peierls_sphere_hollow_f4_case,
        )
        ref = _build_peierls_sphere_hollow_f4_case(
            r0_over_R=r0_over_R, ng_key="2g",
            n_panels_per_region=1, p_order=3,
            n_theta=12, n_rho=12, n_phi=12,
            precision_digits=15,
        )
        assert ref.problem.n_groups == 2
        assert ref.k_eff is not None and ref.k_eff > 0.0
        assert np.isfinite(ref.k_eff) and ref.k_eff < 100.0
        assert "2eg" in ref.name
        assert ref.operator_form == "integral-peierls"


# ═══════════════════════════════════════════════════════════════════════
# Tier 2b — Issue #130 Phase G.5 slab routing infrastructure
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.foundation
class TestSlabViaUnifiedRoutingInfrastructure:
    r"""Issue #130 Phase G.5 routing-switch infrastructure checks.

    These tests verify the dispatch plumbing:

    - ``_SLAB_VIA_UNIFIED`` defaults to True (as of 2026-04-24 after
      Issue #131 resolved the 1.5 % gap; bit-exact parity restored).
    - ``ORPHEUS_SLAB_VIA_E1=1`` env-var forces the native path for
      bisection.
    - ``_build_peierls_slab_case_via_unified`` produces a valid
      :class:`ContinuousReferenceSolution` with the same name as the
      native-path reference.

    The numerical parity gate between the two routes lives in
    :class:`TestSlabViaUnifiedDiscrepancyDiagnostic`.
    """

    def test_default_flag_is_unified(self):
        """Default routing is the unified multi-group path
        (``_SLAB_VIA_UNIFIED is True``). Activated 2026-04-24 after
        Issue #131 resolved the closed-form gap."""
        import importlib
        import os

        from orpheus.derivations import peierls_cases as pc

        # Clear any env-var the test runner might have set, reload,
        # then confirm the default is True.
        old = os.environ.pop("ORPHEUS_SLAB_VIA_E1", None)
        try:
            importlib.reload(pc)
            assert pc._SLAB_VIA_UNIFIED is True, (
                "_SLAB_VIA_UNIFIED must default to True now that "
                "Phase G.5 parity is bit-exact (see Issue #131). "
                "If this test fails, someone regressed the default."
            )
        finally:
            if old is not None:
                os.environ["ORPHEUS_SLAB_VIA_E1"] = old
            importlib.reload(pc)

    def test_env_var_forces_native(self):
        """``ORPHEUS_SLAB_VIA_E1=1`` forces the native path on
        import. The only documented way to route back to the legacy
        E₁ Nyström for bisection."""
        import importlib
        import os

        from orpheus.derivations import peierls_cases as pc

        old = os.environ.get("ORPHEUS_SLAB_VIA_E1")
        os.environ["ORPHEUS_SLAB_VIA_E1"] = "1"
        try:
            importlib.reload(pc)
            assert pc._SLAB_VIA_UNIFIED is False
        finally:
            if old is None:
                os.environ.pop("ORPHEUS_SLAB_VIA_E1", None)
            else:
                os.environ["ORPHEUS_SLAB_VIA_E1"] = old
            importlib.reload(pc)

    def test_unified_builder_produces_valid_reference(self):
        """The unified-path slab builder runs to completion on the
        1G 1-region fixture and emits a well-formed
        ``ContinuousReferenceSolution``. Uses tight quadrature to
        keep the cost bounded (~30 s)."""
        from orpheus.derivations.peierls_cases import (
            _build_peierls_slab_case_via_unified,
        )

        ref = _build_peierls_slab_case_via_unified(
            ng_key="1g", n_regions=1,
            n_panels_per_region=1, p_order=3, precision_digits=15,
        )
        assert ref.name == "peierls_slab_1eg_1rg"
        assert ref.problem.n_groups == 1
        assert ref.operator_form == "integral-peierls"
        assert ref.k_eff is not None and ref.k_eff > 0.0
        assert np.isfinite(ref.k_eff) and ref.k_eff < 100.0


@pytest.mark.l1
@pytest.mark.slow
@pytest.mark.verifies("peierls-unified")
class TestSlabViaUnifiedDiscrepancyDiagnostic:
    r"""Phase G.5 (Issue #130) parity gate: the unified multi-group
    slab build must agree with the native E₁ Nyström reference on
    the shipped ``peierls_slab_2eg_2rg`` fixture to tight tolerance.

    **Originally a diagnostic** of the 1.5 % gap documented in Issue
    #131 (2026-04-24). That issue was resolved the same day by
    replacing the finite-N GL quadrature in the multi-region slab
    branches of ``compute_P_esc_{outer,inner}`` /
    ``compute_G_bc_{outer,inner}`` with the closed-form
    :math:`\tfrac{1}{2} E_2(\tau_{\rm total})` and
    :math:`2\,E_2(\tau_{\rm total})` expressions (the µ-integral is
    closed-form for piecewise-constant :math:`\Sigma_t`).

    After the fix the two paths agree **bit-exactly** (``rel_diff
    = 5.4e-16``) at ``n_panels_per_region=2, p_order=3, dps=20``.
    The tight assertion ``rel_diff < 1e-10`` now gates future
    regressions. Kept ``@pytest.mark.slow`` because the unified
    adaptive-mpmath path still costs ~900 s at this N.
    """

    def test_2eg_2rg_parity_bit_exact(self):
        import numpy as _np
        from orpheus.derivations._xs_library import LAYOUTS, get_xs
        from orpheus.derivations.cp_slab import _THICKNESSES
        from orpheus.derivations.peierls_geometry import (
            SLAB_POLAR_1D, solve_peierls_mg,
        )
        from orpheus.derivations.peierls_slab import (
            solve_peierls_eigenvalue,
        )

        # Fixture setup
        n_regions = 2
        ng_key = "2g"
        layout = LAYOUTS[n_regions]
        thicknesses = _THICKNESSES[n_regions]
        xs_list = [get_xs(region, ng_key) for region in layout]

        sig_t_list = [xs["sig_t"] for xs in xs_list]
        sig_s_list = [xs["sig_s"] for xs in xs_list]
        nu_list = [xs["nu"] * xs["sig_f"] for xs in xs_list]
        chi_list = [xs["chi"] for xs in xs_list]

        sig_t_mg = _np.stack(sig_t_list)
        sig_s_mg = _np.stack(sig_s_list)
        nu_sig_f_mg = _np.stack(nu_list)
        chi_mg = _np.stack(chi_list)
        radii_mg = _np.cumsum(_np.asarray(thicknesses, dtype=float))

        n_panels, p_order, dps = 2, 3, 20

        # Native path — fast E₁ Nyström reference.
        sol_nat = solve_peierls_eigenvalue(
            sig_t_regions=sig_t_list,
            sig_s_matrices=sig_s_list,
            nu_sig_f_all=nu_list,
            chi_all=chi_list,
            thicknesses=thicknesses,
            n_panels_per_region=n_panels, p_order=p_order,
            precision_digits=dps, boundary="white",
        )

        # Unified multi-group adaptive-mpmath.quad path.
        sol_mg = solve_peierls_mg(
            SLAB_POLAR_1D, radii_mg,
            sig_t=sig_t_mg, sig_s=sig_s_mg,
            nu_sig_f=nu_sig_f_mg, chi=chi_mg,
            boundary="white_f4",
            n_panels_per_region=n_panels, p_order=p_order,
            dps=dps, tol=1e-12,
        )

        rel = abs(sol_mg.k_eff - sol_nat.k_eff) / abs(sol_nat.k_eff)
        # Bit-exact target. Post-Issue-#131 fix measurement:
        # rel_diff = 5.4e-16. Tolerance of 1e-10 leaves ~5 orders of
        # margin for potential numerical noise in different
        # environments while still catching a regression of > 1 ppb.
        assert rel < 1e-10, (
            f"Phase G.5 slab parity regressed: rel_diff={rel:.3e} "
            f"(expected ≈ 5e-16 after Issue #131 fix). "
            f"Unified={sol_mg.k_eff:.14f}, native={sol_nat.k_eff:.14f}. "
            f"Investigate before landing more MG code — "
            f"compute_P_esc_{{outer,inner}} / compute_G_bc_{{outer,inner}} "
            f"multi-region slab branches are suspect."
        )


# ═══════════════════════════════════════════════════════════════════════
# Tier 3 — Issue #131 regime-isolation parity (promoted from
# `derivations/diagnostics/diag_slab_issue131_probe_{a,b}_*.py`, authored
# by numerics-investigator 2026-04-23).
#
# These tests gate two regimes the flagship
# ``test_2eg_2rg_parity_bit_exact`` does NOT cover:
#   • 1G 2-region vacuum  — simplest multi-region case, isolates the
#     volume-kernel ray walker and material-interface breakpoints from
#     the MG × closure interaction.
#   • 2G 2-region vacuum  — isolates MG × multi-region from the F.4
#     white closure (the flagship runs white_f4 on the unified side).
#
# Keeping both as permanent guards means a future regression can be
# pinpointed to exactly one axis (regions vs. groups vs. closure).
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.l1
@pytest.mark.slow
@pytest.mark.verifies("peierls-unified")
class TestSlabMultiRegionVacuumParity:
    r"""Unified vs native Peierls parity on multi-region slabs with
    **vacuum BC** (strips the F.4 closure from the comparison).

    Complements :class:`TestSlabViaUnifiedDiscrepancyDiagnostic`, which
    gates the ``white_f4`` closure in the 2G 2-region fixture. These
    two methods isolate multi-region handling itself:

    * Probe A — 1G 2-region, synthetic XS. If this ever fails, the
      bug is in the ray walker / interface breakpoints, *not* in MG.
    * Probe B — 2G 2-region, shipped fixture. Any failure here with
      Probe A passing pinpoints the MG × multi-region interaction
      (assembly, χ indexing, sig_s transpose across region boundary).

    Originally ``diag_slab_issue131_probe_{a,b}_*.py``; promoted per
    the policy in ``tests/derivations/_promotion_policy.md``.
    """

    def test_1g_2rg_vacuum_parity(self):
        """Issue #131 Probe A — 1G 2-region vacuum unified vs native."""
        from orpheus.derivations.peierls_geometry import (
            SLAB_POLAR_1D,
            solve_peierls_mg,
        )
        from orpheus.derivations.peierls_slab import solve_peierls_eigenvalue

        thicknesses = [0.5, 0.5]
        sig_t_A = np.array([0.5])
        sig_t_B = np.array([0.6])
        nu_sf_A = np.array([0.3])
        nu_sf_B = np.array([0.0])
        sig_s_A = np.array([[0.3]])
        sig_s_B = np.array([[0.5]])
        chi_A = np.array([1.0])
        chi_B = np.array([1.0])

        sol_nat = solve_peierls_eigenvalue(
            sig_t_regions=[sig_t_A, sig_t_B],
            sig_s_matrices=[sig_s_A, sig_s_B],
            nu_sig_f_all=[nu_sf_A, nu_sf_B],
            chi_all=[chi_A, chi_B],
            thicknesses=thicknesses,
            n_panels_per_region=2,
            p_order=3,
            precision_digits=20,
            boundary="vacuum",
        )

        sig_t_mg = np.stack([sig_t_A, sig_t_B])
        sig_s_mg = np.stack([sig_s_A, sig_s_B])
        nu_sig_f_mg = np.stack([nu_sf_A, nu_sf_B])
        chi_mg = np.stack([chi_A, chi_B])
        radii_mg = np.cumsum(np.asarray(thicknesses, dtype=float))

        sol_mg = solve_peierls_mg(
            SLAB_POLAR_1D,
            radii_mg,
            sig_t=sig_t_mg,
            sig_s=sig_s_mg,
            nu_sig_f=nu_sig_f_mg,
            chi=chi_mg,
            boundary="vacuum",
            n_panels_per_region=2,
            p_order=3,
            dps=20,
            tol=1e-12,
        )

        rel = abs(sol_mg.k_eff - sol_nat.k_eff) / abs(sol_nat.k_eff)
        assert rel < 1e-8, (
            f"1G 2-region vacuum parity: rel_diff={rel:.3e} "
            f"(native={sol_nat.k_eff:.12f}, unified={sol_mg.k_eff:.12f}). "
            f"Volume kernel / ray-walker regression — multi-region "
            f"interface handling suspect."
        )

    def test_2g_2rg_vacuum_parity(self):
        """Issue #131 Probe B — 2G 2-region vacuum unified vs native.

        Runs the shipped ``peierls_slab_2eg_2rg`` XS fixture with
        vacuum BC on both paths. With the white_f4 gate in
        :class:`TestSlabViaUnifiedDiscrepancyDiagnostic`, a failure
        here while that passes (or vice versa) cleanly isolates MG ×
        multi-region handling from the white-BC closure.
        """
        from orpheus.derivations._xs_library import LAYOUTS, get_xs
        from orpheus.derivations.cp_slab import _THICKNESSES
        from orpheus.derivations.peierls_geometry import (
            SLAB_POLAR_1D,
            solve_peierls_mg,
        )
        from orpheus.derivations.peierls_slab import solve_peierls_eigenvalue

        n_regions = 2
        ng_key = "2g"
        layout = LAYOUTS[n_regions]
        thicknesses = _THICKNESSES[n_regions]
        xs_list = [get_xs(region, ng_key) for region in layout]

        sig_t_list = [xs["sig_t"] for xs in xs_list]
        sig_s_list = [xs["sig_s"] for xs in xs_list]
        nu_list = [xs["nu"] * xs["sig_f"] for xs in xs_list]
        chi_list = [xs["chi"] for xs in xs_list]

        sol_nat = solve_peierls_eigenvalue(
            sig_t_regions=sig_t_list,
            sig_s_matrices=sig_s_list,
            nu_sig_f_all=nu_list,
            chi_all=chi_list,
            thicknesses=thicknesses,
            n_panels_per_region=2,
            p_order=3,
            precision_digits=20,
            boundary="vacuum",
        )

        sig_t_mg = np.stack(sig_t_list)
        sig_s_mg = np.stack(sig_s_list)
        nu_sig_f_mg = np.stack(nu_list)
        chi_mg = np.stack(chi_list)
        radii_mg = np.cumsum(np.asarray(thicknesses, dtype=float))

        sol_mg = solve_peierls_mg(
            SLAB_POLAR_1D,
            radii_mg,
            sig_t=sig_t_mg,
            sig_s=sig_s_mg,
            nu_sig_f=nu_sig_f_mg,
            chi=chi_mg,
            boundary="vacuum",
            n_panels_per_region=2,
            p_order=3,
            dps=20,
            tol=1e-12,
        )

        rel = abs(sol_mg.k_eff - sol_nat.k_eff) / abs(sol_nat.k_eff)
        assert rel < 1e-8, (
            f"2G 2-region vacuum parity: rel_diff={rel:.3e} "
            f"(native={sol_nat.k_eff:.12f}, unified={sol_mg.k_eff:.12f}). "
            f"MG × multi-region regression with closure excluded — "
            f"check χ indexing, sig_s transpose, K assembly."
        )


# ═══════════════════════════════════════════════════════════════════════
# Foundation — software invariant: the multi-region slab branches of
# compute_P_esc_{outer,inner} / compute_G_bc_{outer,inner} must reduce
# to the single-region closed form when Σ_t is spatially uniform.
#
# Promoted from `derivations/diagnostics/diag_slab_issue131_probe_d_*`
# (Issue #131). This invariant gates the Issue-#131 fix MECHANISM
# (closed-form ½ E_2 / 2 E_2 in the multi-region branch) directly,
# orthogonal to the k_eff parity gates above which only see the net
# effect.
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.foundation
class TestMultiRegionEscapeReduction:
    r"""The piecewise-constant-Σ_t closed form for
    ``compute_P_esc_outer``, ``compute_P_esc_inner``,
    ``compute_G_bc_outer``, ``compute_G_bc_inner`` must reduce bit-
    exactly to the homogeneous (single-region) closed form when the
    cross sections are spatially uniform.

    This is a pure software invariant — no equation label, no k_eff
    involvement. Before Issue #131 the multi-region branches used
    finite-N GL µ-quadrature, which diverged from the single-region
    closed form by ~4e-3 at N=24 (the bug). The fix replaced them
    with ½ E_2(τ_total) / 2 E_2(τ_total) closed forms (the µ-integral
    is analytic for piecewise-constant Σ_t). Any regression of the
    branching logic — e.g. accidental fall-through to GL quadrature,
    wrong τ accumulation, off-by-one on region index — will surface
    here as a large reduction error on this trivial check.
    """

    def test_pesc_gbc_multiregion_reduces_to_homogeneous(self):
        """σ_t uniform across regions ⇒ multi-region branch == homogeneous."""
        from orpheus.derivations.peierls_geometry import (
            SLAB_POLAR_1D,
            compute_G_bc_inner,
            compute_G_bc_outer,
            compute_P_esc_inner,
            compute_P_esc_outer,
            composite_gl_r,
        )

        sig_t_val = 2.0  # Region-B-thermal-worst-case from 2eg_2rg fixture.
        thicknesses = [0.5, 0.5]
        radii_2reg = np.cumsum(thicknesses)
        radii_1reg = np.array([1.0])
        sig_t_2reg = np.array([sig_t_val, sig_t_val])
        sig_t_1reg = np.array([sig_t_val])

        n_panels, p_order, dps = 2, 3, 20
        r_nodes, _r_wts, _panels = composite_gl_r(
            np.asarray(radii_2reg, dtype=float),
            n_panels, p_order, dps=dps,
        )
        r_nodes_arr = np.asarray(r_nodes, dtype=float)

        # The four primitives — all four multi-region branches must
        # match the single-region closed form.
        cases = [
            ("P_esc_outer", compute_P_esc_outer, dict(n_angular=24)),
            ("P_esc_inner", compute_P_esc_inner, dict(n_angular=24)),
            ("G_bc_outer", compute_G_bc_outer, dict(n_surf_quad=24)),
            ("G_bc_inner", compute_G_bc_inner, dict(n_surf_quad=24)),
        ]
        for name, fn, kwargs in cases:
            P_hom = fn(
                SLAB_POLAR_1D, r_nodes_arr, radii_1reg, sig_t_1reg,
                dps=dps, **kwargs,
            )
            P_mr = fn(
                SLAB_POLAR_1D, r_nodes_arr, radii_2reg, sig_t_2reg,
                dps=dps, **kwargs,
            )
            err = float(np.max(np.abs(np.asarray(P_hom) - np.asarray(P_mr))))
            assert err < 1e-12, (
                f"{name}: multi-region branch (σ_t uniform) diverges "
                f"from single-region closed form by {err:.3e}. "
                f"Issue #131 regression — multi-region branches must "
                f"use closed-form ½ E_2 / 2 E_2, not GL µ-quadrature."
            )
