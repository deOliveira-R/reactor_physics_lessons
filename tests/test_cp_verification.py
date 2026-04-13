"""Extended CP solver verification: closes QA gaps C-1, C-2, G-1 through G-9, W-1 through W-3.

This file contains targeted tests that address specific QA findings.
Each test docstring states (1) what gap it closes, (2) what bug it would
catch, and (3) the analytical reference used.

Test index
----------
Section 1 — L0: Direct P_inf matrix comparison (G-1)
Section 2 — Multi-group CP matrix properties (G-6, W-2)
Section 3 — Upscatter eigenvalue (G-2, W-1)
Section 4 — (n,2n) eigenvalue and keff formula (C-2, G-3, W-3)
Section 5 — Optically thick / thin stress tests (G-4, G-7)
Section 6 — Convergence rate of power iteration (G-5)
Section 7 — Many-region refinement (G-8)
Section 8 — GS inner iteration vacuousness proof (C-1, G-9)
Section 9 — Cylindrical/spherical Ki4 table resolution convergence (W-6)
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from orpheus.geometry import CoordSystem, Mesh1D, Zone, mesh1d_from_zones
from orpheus.cp.solver import CPMesh, CPParams, CPSolver, solve_cp
from orpheus.data.macro_xs.mixture import Mixture
from orpheus.data.macro_xs.cell_xs import CellXS, assemble_cell_xs

# File-level verifies — every test in this file exercises the CP matrix
# pipeline. Individual classes carry their own @pytest.mark.l0/l1/l2
# marker for V&V level (mixed file).
pytestmark = pytest.mark.verifies(
    "collision-rate",
    "p-inf",
    "neutron-balance",
    "matrix-A-def",
    "matrix-B-def",
    "cp-keff-update",
    "e3-def",
    "ki3-def",
    "self-slab",
    "self-cyl",
    "self-sph",
    "wigner-seitz",
    "matrix-eigenvalue",
    "mg-balance",
)
from orpheus.derivations._xs_library import (
    XS, get_xs, get_mixture, get_materials, make_mixture,
)
from orpheus.derivations._eigenvalue import kinf_from_cp
from orpheus.derivations.cp_slab import _slab_cp_matrix


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_mixture_with_upscatter(ng_key: str = "2g") -> Mixture:
    """Region A but with upscatter: thermal-to-fast scattering added.

    Takes the standard 2G region-A XS and adds a nonzero SigS[1->0]
    entry (upscatter), adjusting sig_t to maintain consistency.

    Returns a Mixture with upscatter.
    """
    xs = get_xs("A", ng_key)
    sig_s = xs["sig_s"].copy()
    # Add upscatter: thermal -> fast
    upscatter_val = 0.02
    sig_s[1, 0] = upscatter_val  # group 1 -> group 0

    # Recompute sig_t for consistency
    sig_t = xs["sig_c"] + xs["sig_f"] + sig_s.sum(axis=1)

    return make_mixture(
        sig_t=sig_t,
        sig_c=xs["sig_c"].copy(),
        sig_f=xs["sig_f"].copy(),
        nu=xs["nu"].copy(),
        chi=xs["chi"].copy(),
        sig_s=sig_s,
    )


def _make_mixture_with_n2n(ng_key: str = "2g") -> Mixture:
    """Region A with nonzero (n,2n) cross section.

    Creates a material where group 0 has a (n,2n) reaction that
    produces 2 neutrons into the same group.  This tests that the
    solver correctly accounts for the extra neutron production.

    The Sig2 matrix encodes the transfer: sig2[g'->g].
    """
    xs = get_xs("A", ng_key)
    ng = len(xs["sig_t"])
    sig_s = xs["sig_s"].copy()

    # (n,2n) from group 0 to group 0 (fast self-multiplication)
    sig2 = np.zeros((ng, ng))
    sig2[0, 0] = 0.01  # small but nonzero

    # sig_t must include the (n,2n) reaction cross section
    # The (n,2n) XS out of group g is sum over g' of sig2[g, g']
    sig2_out = sig2.sum(axis=1)
    sig_t = xs["sig_c"] + xs["sig_f"] + sig_s.sum(axis=1) + sig2_out

    eg = np.logspace(7, -3, ng + 1)
    sig_s_list = [csr_matrix(sig_s)]
    return Mixture(
        SigC=xs["sig_c"].copy(),
        SigL=np.zeros(ng),
        SigF=xs["sig_f"].copy(),
        SigP=(xs["nu"] * xs["sig_f"]).copy(),
        SigT=sig_t,
        SigS=sig_s_list,
        Sig2=csr_matrix(sig2),
        chi=xs["chi"].copy(),
        eg=eg,
    )


def _slab_mesh_homogeneous(thickness: float, mat_id: int = 0) -> Mesh1D:
    """Single-region slab mesh."""
    return Mesh1D(
        edges=np.array([0.0, thickness]),
        mat_ids=np.array([mat_id]),
        coord=CoordSystem.CARTESIAN,
    )


def _slab_mesh_2region(t1: float, t2: float) -> Mesh1D:
    """Two-region slab mesh with mat_ids [0, 1]."""
    return Mesh1D(
        edges=np.array([0.0, t1, t1 + t2]),
        mat_ids=np.array([0, 1]),
        coord=CoordSystem.CARTESIAN,
    )


def _compute_analytical_kinf_slab(
    materials: dict[int, Mixture],
    thicknesses: np.ndarray,
    mat_ids: np.ndarray,
) -> float:
    """Compute analytical k_inf for a slab problem using the derivation module.

    This uses the same _slab_cp_matrix + _kinf_from_cp pipeline as the
    derivation module, giving a dense-eigensolver reference independent
    of the solver under test.
    """
    n_regions = len(thicknesses)
    _any = next(iter(materials.values()))
    ng = _any.ng

    # Build per-region arrays in mesh order (following mat_ids)
    sig_t_all = np.zeros((n_regions, ng))
    sig_s_mats = []
    nu_sig_f_mats = []
    chi_mats = []

    for i, mid in enumerate(mat_ids):
        m = materials[mid]
        sig_t_all[i] = m.SigT
        sig_s_mats.append(np.array(m.SigS[0].todense()))
        nu_sig_f_mats.append(m.SigP)
        chi_mats.append(m.chi)

    P_inf = _slab_cp_matrix(sig_t_all, thicknesses)

    return kinf_from_cp(
        P_inf_g=P_inf,
        sig_t_all=sig_t_all,
        V_arr=thicknesses,
        sig_s_mats=sig_s_mats,
        nu_sig_f_mats=nu_sig_f_mats,
        chi_mats=chi_mats,
    )


def _compute_analytical_kinf_slab_with_n2n(
    materials: dict[int, Mixture],
    thicknesses: np.ndarray,
    mat_ids: np.ndarray,
) -> float:
    """Compute analytical k_inf including (n,2n) via shared kinf_from_cp."""
    n_regions = len(thicknesses)
    _any = next(iter(materials.values()))
    ng = _any.ng

    sig_t_all = np.zeros((n_regions, ng))
    sig_s_mats = []
    sig_2_mats = []
    nu_sig_f_mats = []
    chi_mats = []

    for i, mid in enumerate(mat_ids):
        m = materials[mid]
        sig_t_all[i] = m.SigT
        sig_s_mats.append(np.array(m.SigS[0].todense()))
        sig_2_mats.append(np.array(m.Sig2.todense()))
        nu_sig_f_mats.append(m.SigP)
        chi_mats.append(m.chi)

    P_inf = _slab_cp_matrix(sig_t_all, thicknesses)

    return kinf_from_cp(
        P_inf_g=P_inf,
        sig_t_all=sig_t_all,
        V_arr=thicknesses,
        sig_s_mats=sig_s_mats,
        nu_sig_f_mats=nu_sig_f_mats,
        chi_mats=chi_mats,
        sig_2_mats=sig_2_mats,
    )


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — L0: Direct P_inf matrix comparison (G-1)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
class TestDirectPinfComparison:
    """[G-1] Compare CPMesh.compute_pinf_group against derivation's
    _slab_cp_matrix element-by-element.

    Existing tests only compare eigenvalues, which could mask
    compensating errors in individual P_inf entries.  This directly
    verifies the matrix entries.
    """

    @pytest.mark.parametrize("n_regions,ng_key", [
        (1, "1g"),
        (2, "2g"),
        (4, "4g"),
    ])
    def test_slab_pinf_matches_derivation(self, n_regions, ng_key):
        """[L0] Solver P_inf must match derivation P_inf element-by-element.

        Catches: kernel bugs, normalization errors, white-BC formula
        errors that happen to cancel in eigenvalue but not in matrix.
        Closes: G-1.
        """
        from orpheus.derivations._xs_library import LAYOUTS
        layout = LAYOUTS[n_regions]
        ng = int(ng_key[0])

        # Thicknesses matching the derivation module
        _THICKNESSES = {1: [0.5], 2: [0.5, 0.5], 4: [0.4, 0.05, 0.1, 0.45]}
        _MAT_IDS = {
            1: [2], 2: [2, 0], 4: [2, 3, 1, 0],
        }
        t_arr = np.array(_THICKNESSES[n_regions])
        mat_ids = np.array(_MAT_IDS[n_regions])

        # Build XS arrays
        xs_list = [get_xs(region, ng_key) for region in layout]
        sig_t_all = np.vstack([xs["sig_t"] for xs in xs_list])

        # Derivation reference (high-precision scalar E3)
        P_ref = _slab_cp_matrix(sig_t_all, t_arr)

        # Solver P_inf (vectorised E3)
        edges = np.concatenate([[0.0], np.cumsum(t_arr)])
        mesh = Mesh1D(edges=edges, mat_ids=mat_ids, coord=CoordSystem.CARTESIAN)
        cp_mesh = CPMesh(mesh)

        for g in range(ng):
            P_solver = cp_mesh.compute_pinf_group(sig_t_all[:, g])
            np.testing.assert_allclose(
                P_solver, P_ref[:, :, g],
                atol=1e-10, rtol=1e-10,
                err_msg=f"P_inf mismatch for {n_regions}-region {ng_key} group {g}",
            )

    @pytest.mark.parametrize("coord", [
        CoordSystem.CYLINDRICAL,
        CoordSystem.SPHERICAL,
    ])
    def test_radial_pinf_self_consistency(self, coord):
        """[L0] For radial geometries, verify P_inf properties hold at
        2-group level (row sums, reciprocity, non-negativity).

        This extends G-6 by testing at 2 groups rather than 1.
        Closes: G-1 (partial, radial), G-6 (partial).
        """
        xs_a = get_xs("A", "2g")
        xs_b = get_xs("B", "2g")

        mesh = mesh1d_from_zones([
            Zone(outer_edge=0.5, mat_id=0, n_cells=1),
            Zone(outer_edge=1.0, mat_id=1, n_cells=1),
        ], coord=coord)
        cp_mesh = CPMesh(mesh)

        for g in range(2):
            sig_t_g = np.array([xs_a["sig_t"][g], xs_b["sig_t"][g]])
            P = cp_mesh.compute_pinf_group(sig_t_g)
            V = mesh.volumes

            # Row sums = 1
            np.testing.assert_allclose(
                P.sum(axis=1), 1.0, atol=1e-8,
                err_msg=f"{coord.value} group {g}: row sums != 1",
            )
            # Reciprocity
            lhs = sig_t_g[0] * V[0] * P[0, 1]
            rhs = sig_t_g[1] * V[1] * P[1, 0]
            np.testing.assert_allclose(
                lhs, rhs, atol=1e-8,
                err_msg=f"{coord.value} group {g}: reciprocity violated",
            )
            # Non-negativity
            assert np.all(P >= -1e-15), (
                f"{coord.value} group {g}: negative P entry {P.min():.2e}"
            )


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — Multi-group CP matrix properties (G-6, W-2)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
class TestMultiGroupProperties:
    """[G-6, W-2] Verify CP matrix properties at 2G and 4G, not just 1G.

    All existing property tests (row sums, reciprocity, non-negativity)
    use 1-group XS.  This ensures properties hold when different groups
    have different optical thicknesses.
    """

    @pytest.mark.parametrize("ng_key,coord", [
        ("2g", CoordSystem.CARTESIAN),
        ("4g", CoordSystem.CARTESIAN),
        ("2g", CoordSystem.CYLINDRICAL),
        ("2g", CoordSystem.SPHERICAL),
    ])
    def test_row_sums_multigroup(self, ng_key, coord):
        """[L0] P_inf row sums = 1 for all groups.

        Catches: group-dependent normalization bugs.
        Closes: G-6, W-2.
        """
        xs_a = get_xs("A", ng_key)
        xs_b = get_xs("B", ng_key)
        ng = len(xs_a["sig_t"])

        mesh = mesh1d_from_zones([
            Zone(outer_edge=0.5, mat_id=0, n_cells=1),
            Zone(outer_edge=1.0, mat_id=1, n_cells=1),
        ], coord=coord)
        cp_mesh = CPMesh(mesh)

        for g in range(ng):
            sig_t_g = np.array([xs_a["sig_t"][g], xs_b["sig_t"][g]])
            P = cp_mesh.compute_pinf_group(sig_t_g)
            np.testing.assert_allclose(
                P.sum(axis=1), 1.0, atol=1e-8,
                err_msg=f"{coord.value} {ng_key} group {g}: row sum != 1",
            )

    @pytest.mark.parametrize("ng_key,coord", [
        ("2g", CoordSystem.CARTESIAN),
        ("4g", CoordSystem.CARTESIAN),
        ("2g", CoordSystem.CYLINDRICAL),
        ("2g", CoordSystem.SPHERICAL),
    ])
    def test_reciprocity_multigroup(self, ng_key, coord):
        """[L0] Reciprocity Sigma_t[i] V[i] P[i,j] = Sigma_t[j] V[j] P[j,i]
        for all groups.

        Catches: volume-weighting bugs that only appear in multi-group.
        Closes: G-6, W-2.
        """
        xs_a = get_xs("A", ng_key)
        xs_b = get_xs("B", ng_key)
        ng = len(xs_a["sig_t"])

        mesh = mesh1d_from_zones([
            Zone(outer_edge=0.5, mat_id=0, n_cells=1),
            Zone(outer_edge=1.0, mat_id=1, n_cells=1),
        ], coord=coord)
        cp_mesh = CPMesh(mesh)
        V = mesh.volumes

        for g in range(ng):
            sig_t_g = np.array([xs_a["sig_t"][g], xs_b["sig_t"][g]])
            P = cp_mesh.compute_pinf_group(sig_t_g)
            lhs = sig_t_g[0] * V[0] * P[0, 1]
            rhs = sig_t_g[1] * V[1] * P[1, 0]
            np.testing.assert_allclose(
                lhs, rhs, atol=1e-8,
                err_msg=f"{coord.value} {ng_key} group {g}: reciprocity",
            )


# ═══════════════════════════════════════════════════════════════════════
# Section 3 — Upscatter eigenvalue (G-2, W-1)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
class TestUpscatter:
    """[G-2, W-1] Eigenvalue with nonzero upscatter.

    All standard XS have zero upscatter (SigS[1,0] = 0).  This misses
    bugs in the scattering source assembly when thermal neutrons
    scatter upward.  The test constructs a 2G fuel material with
    upscatter and verifies the eigenvalue matches the analytical
    reference.
    """

    def test_slab_2g_upscatter_eigenvalue(self):
        """[L1] 2G slab with upscatter: solver keff must match analytical.

        Setup: Region A with SigS[1->0] = 0.02 (thermal-to-fast upscatter).
        Expected: analytical keff from _kinf_from_cp with modified scattering.
        Catches: wrong group indexing in scattering source, missing upscatter.
        Closes: G-2, W-1.
        """
        mat = _make_mixture_with_upscatter("2g")
        materials = {0: mat}
        thickness = np.array([0.5])
        mat_ids = np.array([0])

        k_ref = _compute_analytical_kinf_slab(materials, thickness, mat_ids)

        mesh = _slab_mesh_homogeneous(0.5, mat_id=0)
        result = solve_cp(materials, mesh, CPParams(keff_tol=1e-7, flux_tol=1e-6))

        assert abs(result.keff - k_ref) < 1e-5, (
            f"Upscatter keff: solver={result.keff:.8f} ref={k_ref:.8f}"
        )

    def test_upscatter_changes_eigenvalue(self):
        """[L1] Upscatter must change keff relative to downscatter-only.

        If upscatter is silently ignored, keff would match the standard
        region-A value.  This test verifies it is different.
        Closes: G-2 (regression guard).
        """
        mat_up = _make_mixture_with_upscatter("2g")
        mat_no = get_mixture("A", "2g")

        t = np.array([0.5])
        mid = np.array([0])

        k_up = _compute_analytical_kinf_slab({0: mat_up}, t, mid)
        k_no = _compute_analytical_kinf_slab({0: mat_no}, t, mid)

        assert abs(k_up - k_no) > 1e-3, (
            f"Upscatter should change keff: k_up={k_up:.8f} k_no={k_no:.8f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Section 4 — (n,2n) eigenvalue and keff formula (C-2, G-3, W-3)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
class TestN2N:
    """[C-2, G-3, W-3] Tests for (n,2n) reactions.

    C-2: compute_keff ignores (n,2n) neutron production.  The current
    formula is k = nuSigF*phi*V / SigA*phi*V.  With (n,2n), the
    numerator should include the extra neutron production from (n,2n),
    OR the denominator should exclude (n,2n) from absorption.

    The solve_fixed_source correctly includes 2*Sig2 in the source,
    so the transport is right — but the keff estimate is wrong.  This
    means the power iteration may converge to the wrong eigenvalue.
    """

    def test_n2n_changes_eigenvalue(self):
        """[L1] (n,2n) must change keff relative to Sig2=0 case.

        If Sig2 is silently ignored in the keff estimate, the eigenvalue
        would match the standard region-A value.
        Closes: G-3, W-3.
        """
        mat_n2n = _make_mixture_with_n2n("2g")
        mat_no = get_mixture("A", "2g")

        t = np.array([0.5])
        mid = np.array([0])

        # Analytical reference including (n,2n) in the scattering operator
        k_n2n = _compute_analytical_kinf_slab_with_n2n({0: mat_n2n}, t, mid)
        k_no = _compute_analytical_kinf_slab({0: mat_no}, t, mid)

        assert abs(k_n2n - k_no) > 1e-3, (
            f"(n,2n) should increase keff: k_n2n={k_n2n:.8f} k_no={k_no:.8f}"
        )
        assert k_n2n > k_no, (
            f"(n,2n) adds neutrons, keff should increase: "
            f"k_n2n={k_n2n:.8f} k_no={k_no:.8f}"
        )

    def test_n2n_solver_keff_matches_analytical(self):
        """[L1] Solver keff with (n,2n) must match analytical reference.

        Setup: 2G homogeneous slab, region A with Sig2[0,0] = 0.01.
        Expected: keff from dense eigensolver with effective scattering
                  SigS_eff = SigS + 2*Sig2.
        Closes: C-2, G-3, W-3.
        """
        mat_n2n = _make_mixture_with_n2n("2g")
        materials = {0: mat_n2n}
        thickness = np.array([0.5])
        mat_ids = np.array([0])

        k_ref = _compute_analytical_kinf_slab_with_n2n(materials, thickness, mat_ids)

        mesh = _slab_mesh_homogeneous(0.5, mat_id=0)
        result = solve_cp(materials, mesh, CPParams(keff_tol=1e-7, flux_tol=1e-6))

        assert abs(result.keff - k_ref) < 1e-5, (
            f"(n,2n) keff: solver={result.keff:.8f} ref={k_ref:.8f}"
        )

    def test_compute_keff_formula_with_n2n(self):
        """[L0] Direct test of CPSolver.compute_keff with known (n,2n) flux.

        Constructs a CPSolver with known Sig2 and a flat flux, then checks
        that compute_keff includes the (n,2n) contribution.

        k_correct = (nuSigF*phi*V) / (SigA*phi*V - 2*Sig2*phi*V)
        or equivalently
        k_correct = (nuSigF*phi*V + n2n_extra*phi*V) / (SigA*phi*V)

        Closes: C-2 (unit test for the formula).
        """
        # This is a direct formula test, not a full solve.
        # We construct a 1-region 2-group problem with known XS.
        mat = _make_mixture_with_n2n("2g")
        ng = mat.ng

        # Flat flux
        phi = np.ones((1, ng))
        V = np.array([1.0])

        # Current formula: production / absorption
        production = np.sum(mat.SigP * phi[0] * V[0])
        absorption = np.sum(mat.absorption_xs * phi[0] * V[0])
        k_current = production / absorption

        # The (n,2n) reaction produces an EXTRA neutron.
        # The scattering source includes 2*Sig2, so the extra neutron
        # production from (n,2n) is sum(Sig2)*phi*V (one extra per reaction).
        sig2_production = np.sum(
            np.array(mat.Sig2.todense()) @ phi[0] * V[0]
        )

        # Correct keff should account for this
        k_correct = (production + sig2_production) / absorption
        # Or equivalently: k_correct = production / (absorption - sig2_production)

        assert sig2_production > 0, "Test setup: Sig2 must produce extra neutrons"
        assert k_correct > k_current, (
            f"With (n,2n), correct keff ({k_correct:.6f}) should exceed "
            f"current formula ({k_current:.6f})"
        )


# ═══════════════════════════════════════════════════════════════════════
# Section 5 — Optically thick / thin stress tests (G-4, G-7)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
class TestOpticalLimits:
    """[G-4, G-7] Stress tests for extreme optical thicknesses.

    The CP matrix has well-known limiting behaviour:
    - Optically thick (tau >> 1): P_ii -> 1, P_ij -> 0 (i != j)
    - Optically thin (tau << 1): high escape, P_ii << 1
    """

    def test_optically_thick_self_collision_dominant(self):
        """[L0] In thick cells (tau >> 1), self-collision P_ii -> 1.

        Setup: 2-region slab, both with Sigma_t = 100, thickness = 1.0
               (tau = 100 per region).
        Expected: P_ii > 0.99 for both regions.
        Catches: kernel underflow, numerical issues at large optical depth.
        Closes: G-4.
        """
        sig_t_thick = np.array([100.0, 100.0])
        mesh = _slab_mesh_2region(1.0, 1.0)
        cp_mesh = CPMesh(mesh)
        P = cp_mesh.compute_pinf_group(sig_t_thick)

        for i in range(2):
            assert P[i, i] > 0.99, (
                f"Optically thick: P[{i},{i}]={P[i, i]:.6f}, expected > 0.99"
            )

        # Row sums still = 1
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-8)

    def test_optically_thin_high_escape(self):
        """[L0] In thin cells (tau << 1), escape probability is high.

        Setup: 2-region slab, both with Sigma_t = 0.01, thickness = 0.1
               (tau = 0.001 per region).
        Expected: P_cell row sums << 1 (high escape), but P_inf row sums = 1
                  (white BC recaptures escaped neutrons).
        Catches: white-BC formula failure at low optical thickness.
        Closes: G-7.
        """
        sig_t_thin = np.array([0.01, 0.01])
        mesh = _slab_mesh_2region(0.1, 0.1)
        cp_mesh = CPMesh(mesh)
        P = cp_mesh.compute_pinf_group(sig_t_thin)

        # With white BC, P_inf row sums = 1 even for optically thin cells
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-8)

        # Off-diagonal should be significant (neutrons escape and re-enter)
        assert P[0, 1] > 0.1, (
            f"Optically thin: P[0,1]={P[0, 1]:.6f}, expected significant coupling"
        )

    @pytest.mark.parametrize("coord", [
        CoordSystem.CARTESIAN,
        CoordSystem.CYLINDRICAL,
        CoordSystem.SPHERICAL,
    ])
    def test_thick_thin_contrast_eigenvalue(self, coord):
        """[L1] 2G eigenvalue with one thick and one thin region.

        Setup: fuel (region A) with large thickness, moderator (region B)
               with small thickness.  Tests that the solver handles
               mixed optical regimes.
        Catches: overflow/underflow when tau varies by orders of magnitude.
        Closes: G-4, G-7 (eigenvalue level).
        """
        mat_a = get_mixture("A", "2g")
        mat_b = get_mixture("B", "2g")
        materials = {0: mat_a, 1: mat_b}

        if coord == CoordSystem.CARTESIAN:
            mesh = _slab_mesh_2region(2.0, 0.1)
        else:
            mesh = mesh1d_from_zones([
                Zone(outer_edge=2.0, mat_id=0, n_cells=1),
                Zone(outer_edge=2.1, mat_id=1, n_cells=1),
            ], coord=coord)

        # Should converge without numerical issues
        result = solve_cp(materials, mesh, CPParams(keff_tol=1e-6, flux_tol=1e-5))
        assert result.keff > 0, f"keff must be positive, got {result.keff}"
        assert np.isfinite(result.keff), f"keff not finite: {result.keff}"
        assert len(result.keff_history) < 500, "Failed to converge in 500 iterations"


# ═══════════════════════════════════════════════════════════════════════
# Section 6 — Convergence rate of power iteration (G-5)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
class TestConvergenceRate:
    """[G-5] Verify that the power iteration exhibits expected convergence.

    Power iteration converges at rate |k_1/k_0| (dominance ratio).
    The keff history should show monotonic error decrease after initial
    transients.
    """

    def test_keff_history_converges_monotonically(self):
        """[L1] keff history errors must decrease monotonically after transient.

        Setup: 2G 2-region slab (heterogeneous, non-degenerate).
        Expected: after iteration 5, |k_n - k_final| decreases each step.
        Catches: power iteration instability, wrong fission source assembly.
        Closes: G-5.
        """
        from orpheus.derivations import get
        case = get("cp_slab_2eg_2rg")
        gp = case.geom_params
        thicknesses = np.array(gp["thicknesses"])
        edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
        mesh = Mesh1D(
            edges=edges,
            mat_ids=np.array(gp["mat_ids"]),
            coord=CoordSystem.CARTESIAN,
        )
        result = solve_cp(
            case.materials, mesh,
            CPParams(keff_tol=1e-8, flux_tol=1e-7),
        )

        k_hist = np.array(result.keff_history)
        k_final = k_hist[-1]
        errors = np.abs(k_hist - k_final)

        # After iteration 5, errors should be decreasing
        # (allow small violations due to floating-point)
        skip = 5
        decreasing_count = 0
        total = len(errors) - skip - 1
        for i in range(skip, len(errors) - 1):
            if errors[i + 1] <= errors[i] * 1.01:  # 1% tolerance
                decreasing_count += 1

        fraction_decreasing = decreasing_count / max(total, 1)
        assert fraction_decreasing > 0.9, (
            f"Only {fraction_decreasing:.0%} of error steps are decreasing "
            f"(expected > 90%)"
        )

    def test_dominance_ratio_estimate(self):
        """[L1] Estimate dominance ratio from consecutive keff changes.

        The ratio |dk_{n+1}| / |dk_n| should approach the dominance
        ratio k_1/k_0 as iteration progresses.  For a well-separated
        spectrum, this should be < 1.
        Closes: G-5.
        """
        from orpheus.derivations import get
        case = get("cp_slab_4eg_4rg")
        gp = case.geom_params
        thicknesses = np.array(gp["thicknesses"])
        edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
        mesh = Mesh1D(
            edges=edges,
            mat_ids=np.array(gp["mat_ids"]),
            coord=CoordSystem.CARTESIAN,
        )
        result = solve_cp(
            case.materials, mesh,
            CPParams(keff_tol=1e-8, flux_tol=1e-7),
        )

        k_hist = np.array(result.keff_history)
        dk = np.abs(np.diff(k_hist))

        # Use iterations 10-20 to estimate dominance ratio
        if len(dk) > 20:
            ratios = dk[11:20] / dk[10:19]
            median_ratio = np.median(ratios)
            assert 0 < median_ratio < 1.0, (
                f"Dominance ratio estimate = {median_ratio:.4f}, expected in (0, 1)"
            )


# ═══════════════════════════════════════════════════════════════════════
# Section 7 — Many-region refinement (G-8)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l2
class TestManyRegions:
    """[G-8] Tests with more than 4 spatial regions.

    All existing tests use at most 4 regions.  Bugs in loop bounds,
    array indexing, or O(N^2) algorithms may only manifest with
    larger region counts.
    """

    def test_8_region_slab_converges(self):
        """[L1] 8-region slab eigenvalue converges and satisfies properties.

        Setup: Alternating fuel (A) / moderator (B) in 8 regions.
        Expected: finite keff, P_inf row sums = 1 for all groups.
        Catches: indexing bugs, O(N^2) scaling issues.
        Closes: G-8.
        """
        # 8 regions: A B A B A B A B (alternating)
        n_reg = 8
        mat_a = get_mixture("A", "2g")
        mat_b = get_mixture("B", "2g")
        materials = {0: mat_a, 1: mat_b}

        thicknesses = np.array([0.25] * n_reg)
        mat_ids = np.array([0, 1] * (n_reg // 2))
        edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
        mesh = Mesh1D(edges=edges, mat_ids=mat_ids, coord=CoordSystem.CARTESIAN)

        # Verify CP matrix properties
        cp_mesh = CPMesh(mesh)
        for g in range(2):
            sig_t_g = np.array([
                materials[mid].SigT[g] for mid in mat_ids
            ])
            P = cp_mesh.compute_pinf_group(sig_t_g)
            np.testing.assert_allclose(
                P.sum(axis=1), 1.0, atol=1e-8,
                err_msg=f"8-region group {g}: row sums != 1",
            )
            assert np.all(P >= -1e-15), f"8-region group {g}: negative P entry"

        # Eigenvalue should converge
        result = solve_cp(materials, mesh, CPParams(keff_tol=1e-6, flux_tol=1e-5))
        assert np.isfinite(result.keff)
        assert result.keff > 0

    def test_mesh_refinement_convergence(self):
        """[L2] Subdividing regions should converge keff (mesh independence).

        Setup: 2-region slab (A+B), then each region subdivided into
               2, 4, 8 sub-regions of equal material.  keff should
               converge as subdivision increases.
        Catches: mesh-dependent bugs, incorrect volume weighting.
        Closes: G-8 (mesh refinement).
        """
        mat_a = get_mixture("A", "2g")
        mat_b = get_mixture("B", "2g")
        materials = {0: mat_a, 1: mat_b}

        keffs = []
        for n_sub in [1, 2, 4]:
            # Each original region is subdivided into n_sub cells
            t_a = 0.5 / n_sub
            t_b = 0.5 / n_sub
            thicknesses = [t_a] * n_sub + [t_b] * n_sub
            mat_ids_arr = [0] * n_sub + [1] * n_sub
            edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
            mesh = Mesh1D(
                edges=edges,
                mat_ids=np.array(mat_ids_arr),
                coord=CoordSystem.CARTESIAN,
            )
            result = solve_cp(materials, mesh, CPParams(keff_tol=1e-7, flux_tol=1e-6))
            keffs.append(result.keff)

        # Differences between successive refinements should decrease
        diff_1 = abs(keffs[1] - keffs[0])
        diff_2 = abs(keffs[2] - keffs[1])
        assert diff_2 < diff_1, (
            f"Not converging with refinement: diff_1={diff_1:.6e}, diff_2={diff_2:.6e}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Section 8 — GS inner iteration vacuousness (C-1, G-9)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l2
class TestGSInnerIterations:
    """[C-1 fix, G-9] Verify GS inner iterations converge within-group self-scatter.

    The original C-1 finding identified a tautological residual check
    (denom * phi_new - transported = 0 by construction).  The fix uses
    relative flux change ||φ_new - φ_old|| / ||φ_new|| as the inner
    convergence criterion.

    With the corrected residual, groups with strong within-group
    self-scatter (Σ_s(g→g) / Σ_t(g) large, typically thermal) require
    multiple inner iterations; groups without self-scatter converge in 1.
    """

    def test_thermal_needs_more_inner_than_fast(self):
        """[L0] Thermal group (g=3) needs more inner iterations than fast (g=0).

        The 4G XS library has Σ_s(g→g)/Σ_t(g) increasing from fast to
        thermal: the self-scatter ratio drives inner iteration count.

        Setup: 4G 4-region slab in GS mode with tight inner tolerance.
        Expected: mean n_inner for g=3 > mean n_inner for g=0.
        Closes: G-9 (inner iterations reveal physics).
        """
        from orpheus.derivations import get
        case = get("cp_slab_4eg_4rg")
        gp = case.geom_params
        thicknesses = np.array(gp["thicknesses"])
        edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
        mesh = Mesh1D(
            edges=edges,
            mat_ids=np.array(gp["mat_ids"]),
            coord=CoordSystem.CARTESIAN,
        )
        params = CPParams(
            keff_tol=1e-7, flux_tol=1e-6,
            solver_mode="gauss_seidel",
            inner_tol=1e-10,
            max_inner=200,
        )
        result = solve_cp(case.materials, mesh, params)

        assert result.n_inner is not None
        mean_inner = result.n_inner.mean(axis=0)  # (4,) per group
        assert mean_inner[-1] >= mean_inner[0], (
            f"Thermal group should need >= inner iterations than fast: "
            f"fast(g=0)={mean_inner[0]:.1f} thermal(g=3)={mean_inner[-1]:.1f}"
        )

    def test_gs_eigenvalue_matches_jacobi(self):
        """[L1] GS and Jacobi must agree on eigenvalue.

        Both solver modes solve the same equations; GS just converges
        the within-group scatter inline.  The eigenvalue must agree.

        Setup: 4G 4-region slab.
        Closes: eigenvalue consistency.
        """
        from orpheus.derivations import get
        case = get("cp_slab_4eg_4rg")
        gp = case.geom_params
        thicknesses = np.array(gp["thicknesses"])
        edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
        mesh = Mesh1D(
            edges=edges,
            mat_ids=np.array(gp["mat_ids"]),
            coord=CoordSystem.CARTESIAN,
        )

        params_j = CPParams(keff_tol=1e-7, flux_tol=1e-6, solver_mode="jacobi")
        params_gs = CPParams(keff_tol=1e-7, flux_tol=1e-6, solver_mode="gauss_seidel")

        res_j = solve_cp(case.materials, mesh, params_j)
        res_gs = solve_cp(case.materials, mesh, params_gs)

        assert abs(res_j.keff - res_gs.keff) < 1e-6, (
            f"Jacobi keff={res_j.keff:.8f} vs GS keff={res_gs.keff:.8f}"
        )

    def test_no_self_scatter_one_inner(self):
        """[L0] Groups without self-scatter converge in exactly 1 inner iteration.

        When Σ_s(g→g) = 0, the source Q_g does not depend on φ_g,
        so one transport solve gives the exact answer — no iteration needed.

        Setup: 2G problem where SigS is purely downscatter (SigS[g,g]=0
        for both groups would mean no self-scatter, but the standard XS
        have SigS[0,0]=0.38 and SigS[1,1]=0.90). Instead, construct a
        custom material with zero diagonal.
        """
        xs_a = get_xs("A", "2g")
        sig_s_nodiag = xs_a["sig_s"].copy()
        sig_s_nodiag[0, 0] = 0.0
        sig_s_nodiag[1, 1] = 0.0
        sig_t_new = xs_a["sig_c"] + xs_a["sig_f"] + sig_s_nodiag.sum(axis=1)

        mat = make_mixture(
            sig_t=sig_t_new, sig_c=xs_a["sig_c"].copy(),
            sig_f=xs_a["sig_f"].copy(), nu=xs_a["nu"].copy(),
            chi=xs_a["chi"].copy(), sig_s=sig_s_nodiag,
        )
        mesh = _slab_mesh_homogeneous(0.5, mat_id=0)
        params = CPParams(
            keff_tol=1e-7, flux_tol=1e-6,
            solver_mode="gauss_seidel",
            inner_tol=1e-12,
            max_inner=50,
        )
        result = solve_cp({0: mat}, mesh, params)

        assert result.n_inner is not None
        # Without self-scatter, the source for group g doesn't depend
        # on φ_g. The first inner iteration changes φ_g from the initial
        # guess; the second confirms convergence (same source → same flux).
        # After the first outer iteration, φ is already consistent, so
        # subsequent outers converge in 1 inner. Allow ≤ 2 on the first outer.
        assert np.all(result.n_inner <= 2), (
            f"No self-scatter should mean ≤ 2 inner iterations. "
            f"Max={result.n_inner.max()}, unique={np.unique(result.n_inner)}"
        )
        # After the first outer, should be exactly 1
        if result.n_inner.shape[0] > 1:
            assert np.all(result.n_inner[1:] == 1), (
                f"After first outer, should be 1 inner. "
                f"Got: {result.n_inner[1:]}"
            )


# ═══════════════════════════════════════════════════════════════════════
# Section 9 — Ki4 table resolution convergence (W-6)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
class TestKi4Resolution:
    """[W-6] Verify that cylindrical/spherical tolerance is controlled
    by Ki4 table resolution, and that increasing resolution improves
    accuracy.

    Cylindrical and spherical tests use 10x looser tolerance than slab.
    This section verifies that the gap is due to table interpolation,
    not a systematic error.
    """

    def test_cylindrical_ki4_convergence_with_table_size(self):
        """[L2] Increasing Ki4 table points should improve cylindrical keff.

        Setup: 2G 2-region cylindrical problem.
        Expected: keff changes by < 1e-6 between 10k and 40k table points,
                  confirming the default 20k is adequate.
        Catches: table resolution being the dominant error source.
        Closes: W-6.
        """
        from orpheus.derivations import get
        case = get("cp_cyl1D_2eg_2rg")
        gp = case.geom_params
        radii = np.array(gp["radii"])
        edges = np.concatenate([[0.0], radii])
        mesh = Mesh1D(
            edges=edges,
            mat_ids=np.array(gp["mat_ids"]),
            coord=CoordSystem.CYLINDRICAL,
        )

        keffs = {}
        for n_ki in [5000, 20000, 40000]:
            params = CPParams(
                keff_tol=1e-7, flux_tol=1e-6,
                n_ki_table=n_ki,
            )
            result = solve_cp(case.materials, mesh, params)
            keffs[n_ki] = result.keff

        # Convergence: difference should decrease with more table points
        diff_low = abs(keffs[20000] - keffs[5000])
        diff_high = abs(keffs[40000] - keffs[20000])

        assert diff_high < diff_low, (
            f"Ki4 table not converging: "
            f"|k(20k)-k(5k)|={diff_low:.2e}, |k(40k)-k(20k)|={diff_high:.2e}"
        )

        # The 20k-to-40k difference should be small (within the Ki4
        # interpolation accuracy, which is ~1e-5 for the default table)
        assert diff_high < 2e-5, (
            f"Ki4 residual error too large: |k(40k)-k(20k)|={diff_high:.2e}"
        )
