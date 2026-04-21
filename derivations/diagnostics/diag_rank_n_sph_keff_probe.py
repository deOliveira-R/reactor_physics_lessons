"""Diagnostic: rank-N hollow-sphere k_eff probe across conventions.

Created by numerics-investigator on 2026-04-21.

Monkey-patches ``pg.build_closure_operator`` to test alternative
assembly recipes at N=2 on a hollow sphere with white BC. Records
k_eff residuals versus k_inf = nu*Sigma_f / Sigma_a = 1.5.

If this test catches a real bug, promote to
``tests/derivations/test_peierls_rank2_bc.py`` — once the fix lands,
the "Bmu-inverse-W" or "(2n+1)-weighted-W" recipe should land at
<= 0.1% residual.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/workspaces/ORPHEUS")
sys.path.insert(0, "/workspaces/ORPHEUS/derivations/diagnostics")

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from orpheus.derivations._kernels import _shifted_legendre_eval
from orpheus.derivations import peierls_geometry as pg
from orpheus.derivations.peierls_geometry import (
    CurvilinearGeometry, BoundaryClosureOperator,
    compute_hollow_sph_transmission_rank_n,
)

try:
    from diag_rank_n_sph_normalisation_probe import build_pg, half_range_gram
except ImportError:
    # test-run from a different cwd — fall back.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "diag_probe",
        "/workspaces/ORPHEUS/derivations/diagnostics/"
        "diag_rank_n_sph_normalisation_probe.py",
    )
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    build_pg = _m.build_pg
    half_range_gram = _m.half_range_gram


K_INF = 1.5  # homogeneous cell: nu*Sigma_f / (Sigma_t - Sigma_s) = 0.75/0.5


def _setup():
    R = 5.0
    r_in = 0.3 * R
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_in)
    sig_t = np.array([1.0])
    sig_s = np.array([0.5])
    nu_sig_f = np.array([0.75])
    W_full = compute_hollow_sph_transmission_rank_n(
        r_in, R, np.array([R]), sig_t, n_bc_modes=2, dps=15,
    )
    return geom, R, r_in, sig_t, sig_s, nu_sig_f, W_full


def build_custom_closure_factory(mu_power_P, mu_power_G, recipe, N, W_full, orig):
    def _build(geometry, r_nodes, r_wts, radii, sig_t, *,
               n_angular=32, n_surf_quad=32, dps=25,
               n_bc_modes=1, reflection="marshak"):
        if n_bc_modes == 1:
            return orig(geometry, r_nodes, r_wts, radii, sig_t,
                        n_angular=n_angular, n_surf_quad=n_surf_quad,
                        dps=dps, n_bc_modes=1, reflection=reflection)
        P_mat, G_mat, _, _ = build_pg(geometry, r_nodes, r_wts, radii,
                                       sig_t, N,
                                       mu_power_P=mu_power_P,
                                       mu_power_G=mu_power_G,
                                       n_angular=n_angular, dps=dps)
        I_2N = np.eye(2 * N)
        D_gel = np.diag([1.0, 3.0] * 2)
        if recipe == "I-W_inv":
            R_mat = np.linalg.inv(I_2N - W_full)
        elif recipe == "I-W_inv_D":
            R_mat = np.linalg.inv(I_2N - W_full) @ D_gel
        elif recipe == "I-BmuInvW_inv":
            Bmu = half_range_gram(N, "mu")
            Bmu_block = np.block([[Bmu, np.zeros((N, N))],
                                   [np.zeros((N, N)), Bmu]])
            R_mat = np.linalg.inv(I_2N - np.linalg.inv(Bmu_block) @ W_full)
        else:
            raise ValueError(recipe)
        return BoundaryClosureOperator(P=P_mat, R=R_mat, G=G_mat)
    return _build


def run_keff_probe():
    geom, R, r_in, sig_t, sig_s, nu_sig_f, W_full = _setup()

    orig = pg.build_closure_operator
    print(f"Hollow sphere R={R}, r_0/R=0.3, homogeneous.")
    print(f"  k_inf = {K_INF}")

    # N=1 baseline:
    try:
        sol = pg.solve_peierls_1g(
            geom, np.array([R]), sig_t, sig_s, nu_sig_f,
            boundary="white",
            n_panels_per_region=2, p_order=4,
            n_angular=24, n_rho=24, n_surf_quad=24, dps=15,
            n_bc_modes=1,
        )
        print(f"  N=1 (shipped rank-2): k_eff = {sol.k_eff:.6f}, "
              f"err = {abs(sol.k_eff - K_INF)/K_INF*100:.3f}%")
    finally:
        pg.build_closure_operator = orig

    # N=2 convention scan:
    print()
    print("N=2 convention scan:")
    N = 2
    conventions = [
        (0, 0, "I-W_inv"),         # current Phase F.5 (code's C convention)
        (0, 0, "I-W_inv_D"),       # Gelbard (2n+1) per face
        (0, 0, "I-BmuInvW_inv"),   # measure-converted
        (1, 1, "I-W_inv"),         # mu-weighted P and G
        (1, 0, "I-W_inv"),         # mu-weighted P only
        (0, 1, "I-W_inv"),         # mu-weighted G only
    ]
    for mP, mG, recipe in conventions:
        pg.build_closure_operator = build_custom_closure_factory(
            mP, mG, recipe, N, W_full, orig,
        )
        try:
            sol = pg.solve_peierls_1g(
                geom, np.array([R]), sig_t, sig_s, nu_sig_f,
                boundary="white",
                n_panels_per_region=2, p_order=4,
                n_angular=24, n_rho=24, n_surf_quad=24, dps=15,
                n_bc_modes=N,
            )
            err = abs(sol.k_eff - K_INF) / K_INF * 100
            print(f"  mu_P={mP}, mu_G={mG}, R={recipe:18}: "
                  f"k={sol.k_eff:.5f}, err={err:7.3f}%")
        except Exception as e:
            print(f"  mu_P={mP}, mu_G={mG}, R={recipe:18}: "
                  f"FAIL {type(e).__name__}: {str(e)[:60]}")
        finally:
            pg.build_closure_operator = orig


def test_N1_baseline_residual():
    """Regression pin: the Phase F.4 rank-2 N=1 k_eff residual at
    R=5, r_0/R=0.3, homogeneous sigma_t=1/sigma_s=0.5/nuSigf=0.75 is
    approximately 3% — the scalar baseline the N>=2 closure must match
    or improve.
    """
    geom, R, _, sig_t, sig_s, nu_sig_f, _ = _setup()
    sol = pg.solve_peierls_1g(
        geom, np.array([R]), sig_t, sig_s, nu_sig_f,
        boundary="white",
        n_panels_per_region=2, p_order=4,
        n_angular=24, n_rho=24, n_surf_quad=24, dps=15,
        n_bc_modes=1,
    )
    err = abs(sol.k_eff - K_INF) / K_INF
    assert err < 0.05, f"N=1 baseline residual {err*100:.3f}% > 5%"
    # Pin: 3.031% +- 0.1%
    assert abs(err * 100 - 3.031) < 0.1, (
        f"N=1 residual drift from Phase F.4 value: got {err*100:.3f}%, "
        f"expected 3.031%"
    )


@pytest.mark.skip(reason="Convention under investigation — Issue #119")
def test_N2_improves_over_N1():
    """Target regression for Phase F.5 Issue #119 fix.

    After fix, the correct N=2 convention should produce k_eff with
    residual <= 0.1% (well below the 3.031% N=1 baseline).
    """
    geom, R, _, sig_t, sig_s, nu_sig_f, _ = _setup()
    sol = pg.solve_peierls_1g(
        geom, np.array([R]), sig_t, sig_s, nu_sig_f,
        boundary="white",
        n_panels_per_region=2, p_order=4,
        n_angular=24, n_rho=24, n_surf_quad=24, dps=15,
        n_bc_modes=2,
    )
    err = abs(sol.k_eff - K_INF) / K_INF * 100
    assert err < 0.1, (
        f"N=2 residual {err:.3f}% > 0.1% — Phase F.5 closure still "
        f"incorrect. See Issue #119."
    )


if __name__ == "__main__":
    run_keff_probe()
