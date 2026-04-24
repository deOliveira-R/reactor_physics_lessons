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
