r"""Spatial homogenization gate — ``Solution.homogenize`` preserves rates.

Homogenization collapses a fine SN solution's per-cell cross sections onto
a coarse mesh by the flux·volume-weighted average

.. math::

    \Sigma_{R,g} = \frac{\sum_{i\in R} V_i\phi_{i,g}\Sigma_{i,g}}
                        {\sum_{i\in R} V_i\phi_{i,g}},

whose defining property is **reaction-rate preservation**:
:math:`\Sigma_{R,g}\,\Phi_{R,g} = \sum_{i\in R} V_i\Sigma_{i,g}\phi_{i,g}`
with :math:`\Phi_{R,g} = \sum_{i\in R} V_i\phi_{i,g}`.

THE gate (L0 term verification) checks this identity against a
**structurally-independent** reference: an explicit per-region Python loop
over the fine cells (``vv-principles`` L11 — NOT a re-call of the
production ``membership @ …`` matmul).  The matrix channels
(:math:`\Sigma_{s,\ell}`, :math:`\Sigma_{2n}`, indexed ``[g_from, g_to]``)
weight by the **source** group; a ``g_from``↔``g_to`` swap (vv Mode 2) is
caught because the reference loop weights by ``g_from`` explicitly.
Companion checks: every homogenized ``Mixture`` balances and its χ is a
simplex; the effective Σ is bracketed by the region's fine extremes; and
the identity / single-material controls pin the degenerate limits.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.common.xs_library import make_mixture
from orpheus.geometry import BC, CoordSystem, Mesh1D, Mesh2D
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.solver import solve_sn

pytestmark = [pytest.mark.l0, pytest.mark.cap("solve")]

NG = 2
_FINE_EDGES = np.linspace(0.0, 4.0, 9)            # 8 fine cells
_MAT_IDS = np.array([0, 0, 0, 1, 1, 0, 1, 1])     # heterogeneous, interleaved


def _balanced_fissile(sig_c, sig_f, nu, chi, sig_s, sig_s1=None):
    """A balanced fissile Mixture: SigT = SigC + SigF + rowsum(SigS0)."""
    sig_c = np.asarray(sig_c, float); sig_f = np.asarray(sig_f, float)
    sig_s = np.asarray(sig_s, float)
    sig_t = sig_c + sig_f + sig_s.sum(axis=1)
    return make_mixture(
        sig_t, sig_c, sig_f, np.asarray(nu, float), np.asarray(chi, float),
        sig_s, sig_s1=None if sig_s1 is None else np.asarray(sig_s1, float),
    )


@pytest.fixture(scope="module")
def materials():
    m0 = _balanced_fissile(
        [0.20, 0.30], [0.10, 0.20], [2.4, 2.4], [1.0, 0.0],
        [[0.60, 0.10], [0.0, 0.90]], sig_s1=[[0.05, 0.0], [0.0, 0.08]],
    )
    m1 = _balanced_fissile(
        [0.30, 0.40], [0.15, 0.25], [2.4, 2.4], [1.0, 0.0],
        [[0.50, 0.05], [0.0, 0.85]], sig_s1=[[0.03, 0.0], [0.0, 0.06]],
    )
    m0.assert_balanced(); m1.assert_balanced()
    return {0: m0, 1: m1}


@pytest.fixture(scope="module")
def solution(materials):
    fine = Mesh1D(
        edges=_FINE_EDGES, mat_ids=_MAT_IDS, coord=CoordSystem.CARTESIAN,
        bc_left=BC("reflective"), bc_right=BC("reflective"),
    )
    quad = Quadrature.gauss_legendre(n_ordinates=8)
    return solve_sn(materials, fine, quad, scattering_order=0)


def _coarse_two_region():
    """Two coarse cells [0,2],[2,4], each mixing materials 0 and 1."""
    return Mesh1D(
        edges=np.array([0.0, 2.0, 4.0]), mat_ids=np.array([0, 1]),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("reflective"), bc_right=BC("reflective"),
    )


def _fine_region_indices(coarse_edges):
    centers = 0.5 * (_FINE_EDGES[:-1] + _FINE_EDGES[1:])
    cof = np.clip(
        np.searchsorted(coarse_edges, centers, side="right") - 1,
        0, coarse_edges.size - 2,
    )
    return [np.where(cof == R)[0] for R in range(coarse_edges.size - 1)]


# ── THE gate: reaction-rate preservation ──────────────────────────────

@pytest.mark.verifies("sn-homogenization-rate-preservation")
def test_rate_preservation_vector_channels(solution, materials):
    """Σ_{R,g}·Φ_{R,g} == Σ_{i∈R} V_i Σ_{i,g} φ_{i,g} for every vector channel."""
    coarse = _coarse_two_region()
    mm = solution.homogenize(coarse)
    phi = solution.scalar_flux.values            # (ng, n_fine)
    V = np.asarray(solution.mesh.volumes)        # (n_fine,)
    regions = _fine_region_indices(coarse.edges)

    for channel in ("SigT", "SigC", "SigL", "SigF", "SigP"):
        for R, sel in enumerate(regions):
            sig_eff = getattr(mm.materials[R], channel)       # (ng,)
            for g in range(NG):
                phi_R = float((V[sel] * phi[g, sel]).sum())    # region flux integral
                rate_homog = sig_eff[g] * phi_R
                rate_ref = float(sum(
                    V[i] * phi[g, i] * getattr(materials[_MAT_IDS[i]], channel)[g]
                    for i in sel
                ))
                assert rate_homog == pytest.approx(rate_ref, abs=1e-12, rel=1e-12), (
                    f"{channel} rate not preserved in coarse {R}, group {g}"
                )


@pytest.mark.verifies("sn-homogenization-rate-preservation")
def test_rate_preservation_scattering_and_n2n(solution, materials):
    """Σ_{s,ℓ,R}[g',g]·Φ_{R,g'} == Σ_{i∈R} V_iφ_{i,g'} Σ_{s,ℓ,i}[g',g] (source-group
    weighted) for every Legendre order AND the (n,2n) matrix — catches a
    g_from↔g_to swap (vv Mode 2)."""
    coarse = _coarse_two_region()
    mm = solution.homogenize(coarse)
    phi = solution.scalar_flux.values
    V = np.asarray(solution.mesh.volumes)
    regions = _fine_region_indices(coarse.edges)
    n_leg = len(materials[0].SigS)

    def fine_mat(i, order):
        sig_s = materials[_MAT_IDS[i]].SigS
        return np.asarray(sig_s[order].todense()) if order < len(sig_s) \
            else np.zeros((NG, NG))

    for order in range(n_leg):
        for R, sel in enumerate(regions):
            sig_eff = np.asarray(mm.materials[R].SigS[order].todense())  # (ng,ng)
            for gf in range(NG):           # source group
                phi_R = float((V[sel] * phi[gf, sel]).sum())
                for gt in range(NG):       # sink group
                    rate_homog = sig_eff[gf, gt] * phi_R
                    rate_ref = float(sum(
                        V[i] * phi[gf, i] * fine_mat(i, order)[gf, gt] for i in sel
                    ))
                    assert rate_homog == pytest.approx(rate_ref, abs=1e-12, rel=1e-12), (
                        f"SigS[{order}][{gf},{gt}] rate not preserved in coarse {R}"
                    )

    # (n,2n) channel (all zero here, but the path + weighting must hold).
    for R, sel in enumerate(regions):
        sig2 = np.asarray(mm.materials[R].Sig2[0].todense())
        np.testing.assert_allclose(sig2, 0.0, atol=1e-14)


# ── Companion invariants ──────────────────────────────────────────────

@pytest.mark.verifies("sn-homogenization-balance-preservation")
def test_homogenized_materials_balance(solution):
    """Balance survives the collapse — every removal channel shares the weight."""
    mm = solution.homogenize(_coarse_two_region())
    for mix in mm.materials.values():
        mix.assert_balanced(atol=1e-9)


def test_homogenized_chi_is_simplex(solution):
    """χ_R is a probability simplex (convex avg of producing simplices)."""
    mm = solution.homogenize(_coarse_two_region())
    for mix in mm.materials.values():
        # producing region → sums to 1; the Mixture __post_init__ already
        # enforced the simplex/null law, so a positive check suffices here.
        assert mix.chi.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(mix.chi >= -1e-15)


def test_effective_xs_bracketed_by_fine_extremes(solution, materials):
    """Homogenized Σ_t is a flux·volume average → bracketed by the region's
    fine-cell extremes (a physical sanity check independent of the rate gate)."""
    coarse = _coarse_two_region()
    mm = solution.homogenize(coarse)
    for R, sel in enumerate(_fine_region_indices(coarse.edges)):
        fine_sigt = np.array([materials[_MAT_IDS[i]].SigT for i in sel])  # (n_i, ng)
        lo, hi = fine_sigt.min(axis=0), fine_sigt.max(axis=0)
        eff = mm.materials[R].SigT
        assert np.all(eff >= lo - 1e-12) and np.all(eff <= hi + 1e-12)


# ── Degenerate-limit controls ─────────────────────────────────────────

def test_identity_homogenization_recovers_per_cell_materials(solution, materials):
    """Homogenize onto the SAME fine mesh → each coarse cell is one fine cell,
    so the effective material equals that cell's original (avg over one cell)."""
    same = Mesh1D(
        edges=_FINE_EDGES, mat_ids=_MAT_IDS, coord=CoordSystem.CARTESIAN,
        bc_left=BC("reflective"), bc_right=BC("reflective"),
    )
    mm = solution.homogenize(same)
    for i in range(len(_MAT_IDS)):
        orig = materials[_MAT_IDS[i]]
        np.testing.assert_allclose(mm.materials[i].SigT, orig.SigT, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(mm.materials[i].SigS[0].todense()),
            np.asarray(orig.SigS[0].todense()), atol=1e-12,
        )


def test_single_material_region_recovers_material(materials):
    """A coarse cell containing only material m homogenizes to m (flux cancels)."""
    # Uniform single-material fine mesh → any coarse partition gives m back.
    fine = Mesh1D(
        edges=np.linspace(0.0, 3.0, 7), mat_ids=np.zeros(6, dtype=int),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("reflective"), bc_right=BC("reflective"),
    )
    quad = Quadrature.gauss_legendre(n_ordinates=8)
    sol = solve_sn({0: materials[0]}, fine, quad, scattering_order=0)
    mm = sol.homogenize(Mesh1D(
        edges=np.array([0.0, 1.5, 3.0]), mat_ids=np.array([0, 1]),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("reflective"), bc_right=BC("reflective"),
    ))
    for mix in mm.materials.values():
        np.testing.assert_allclose(mix.SigT, materials[0].SigT, atol=1e-12)


# ── Guard ─────────────────────────────────────────────────────────────

def test_outer_boundary_mismatch_raises(solution):
    bad = Mesh1D(
        edges=np.array([0.0, 2.0, 3.5]),  # outer 3.5 ≠ fine outer 4.0
        mat_ids=np.array([0, 1]), coord=CoordSystem.CARTESIAN,
        bc_left=BC("reflective"), bc_right=BC("reflective"),
    )
    with pytest.raises(ValueError, match="outer boundary"):
        solution.homogenize(bad)


# ── Derived-measure discriminator: φV-weighted, NOT dV (volume-only) ───

def test_homogenization_is_flux_weighted_not_volume_weighted(materials):
    """DISTINGUISH the φV measure from a dV (volume-only) average — the
    load-bearing guard on the L²(φV) derivation.  A coarse region spanning a
    strong flux gradient (vacuum→reflective) over two materials of different
    Σ_t makes the φV-weighted effective Σ_t and the dV-weighted one numerically
    distinct; production MUST equal the φV one.  Reds a regression that drops φ
    from the weight (volume-only averaging)."""
    fine = Mesh1D(
        edges=np.linspace(0.0, 2.0, 5), mat_ids=np.array([0, 0, 1, 1]),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("vacuum"), bc_right=BC("reflective"),   # strong flux tilt
    )
    quad = Quadrature.gauss_legendre(n_ordinates=8)
    sol = solve_sn(materials, fine, quad, scattering_order=0)
    coarse = Mesh1D(
        edges=np.array([0.0, 2.0]), mat_ids=np.array([0]),   # ONE region
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("vacuum"), bc_right=BC("reflective"),
    )
    mm = sol.homogenize(coarse)

    phi = sol.scalar_flux.values                 # (ng, n_fine)
    V = np.asarray(sol.mesh.volumes)
    sigt_fine = np.array([materials[m].SigT for m in fine.mat_ids])  # (n_fine, ng)

    discriminated = False
    for g in range(NG):
        w_fluxvol = V * phi[g]
        sig_phi = (w_fluxvol * sigt_fine[:, g]).sum() / w_fluxvol.sum()  # φV
        sig_vol = (V * sigt_fine[:, g]).sum() / V.sum()                  # dV (WRONG)
        eff = mm.materials[0].SigT[g]
        np.testing.assert_allclose(
            eff, sig_phi, rtol=1e-12,
            err_msg=f"g={g}: eff={eff} != φV-weighted {sig_phi}",
        )
        if abs(sig_phi - sig_vol) > 1e-3:        # this group discriminates
            discriminated = True
            assert abs(eff - sig_vol) > 1e-4, (
                f"g={g}: production matched the dV (volume-only) average!"
            )
    assert discriminated, (
        "fixture too flat: φ does not vary enough within the region to "
        "distinguish φV- from dV-weighting"
    )


def test_chi_is_production_weighted(solution, materials):
    """χ_R is the PRODUCTION-weighted convex average (weight
    p_i = Σ_g νΣ_{f,i,g} φ_{i,g} V_i), not flux- or volume-weighted —
    ``test_homogenized_chi_is_simplex`` is blind to the weight (any convex
    weight sums to 1)."""
    coarse = _coarse_two_region()
    mm = solution.homogenize(coarse)
    phi = solution.scalar_flux.values
    V = np.asarray(solution.mesh.volumes)
    for R, sel in enumerate(_fine_region_indices(coarse.edges)):
        production = np.array([
            float((materials[_MAT_IDS[i]].SigP * phi[:, i]).sum() * V[i])
            for i in sel
        ])
        chi_fine = np.array([materials[_MAT_IDS[i]].chi for i in sel])
        chi_ref = (production[:, None] * chi_fine).sum(axis=0) / production.sum()
        np.testing.assert_allclose(mm.materials[R].chi, chi_ref, rtol=1e-12)


# ── Mode-11: the φ-weighting is actually on the TEST side, on the call graph ──

def test_homogenize_routes_through_the_petrov_galerkin_frame(solution, monkeypatch):
    """Mode-11 sentinel: ``homogenize`` routes the φ-weighting through the TEST side.

    The PG re-frame's load-bearing claim is that φ moved OUT of the metric INTO the
    test basis.  A green rate gate is **vacuous** for that claim unless the new
    production readers are on the call graph: a bit-identity-preserving regression
    that kept the OLD Galerkin metric-fold (folding φ into the coefficient-space
    metric, test = plain trial indicator) would produce IDENTICAL numbers yet NEVER
    construct the weighted test basis.  So the sentinel monkeypatch-counts:

    * ``IndicatorBasis.evaluate`` — the TRIAL membership table (still a reader);
    * ``WeightedIndicatorBasis.analyze`` — the **TEST-side** reader (the load-bearing
      re-point: φ now lives on the test basis, not folded into the metric);
    * ``FrameBase.project`` — the NEW coefficient-extraction verb G⁻¹M;
    * ``CrossGramInverse.apply`` — the normalisation :math:`G^{-1}` inside it,
      the frame's own arrow ``test_space → basis_space`` (CS4c step 6 item
      6.2c-ii; until then ``project`` normalised through the probe SPACE's
      ``FunctionSpace.apply_inverse_metric``, which this sentinel counted).

    A φ-never-moved mutation (keep ``GalerkinFrame`` + the metric-fold) leaves
    ``test_analyze`` at 0 → RED, while the rate gate stays green — the exact
    structural-vs-value split Mode-11 exists to catch.
    """
    from orpheus.numerics.basis.indicator_basis import IndicatorBasis
    from orpheus.numerics.basis.weighted_indicator_basis import WeightedIndicatorBasis
    from orpheus.numerics.frame import CrossGramInverse, FrameBase

    counts = {"evaluate": 0, "test_analyze": 0, "project": 0, "inverse_metric": 0}

    def _counting(name, fn):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return fn(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(
        IndicatorBasis, "evaluate", _counting("evaluate", IndicatorBasis.evaluate),
    )
    monkeypatch.setattr(
        WeightedIndicatorBasis, "analyze",
        _counting("test_analyze", WeightedIndicatorBasis.analyze),
    )
    monkeypatch.setattr(
        FrameBase, "project", _counting("project", FrameBase.project),
    )
    monkeypatch.setattr(
        CrossGramInverse, "apply",
        _counting("inverse_metric", CrossGramInverse.apply),
    )

    solution.homogenize(_coarse_two_region())

    assert counts["evaluate"] > 0, "homogenize did NOT tabulate the trial membership"
    assert counts["test_analyze"] > 0, (
        "homogenize did NOT route the analysis through the WEIGHTED test basis — "
        "the flux is not on the test side (it may still be folded into the metric)"
    )
    assert counts["project"] > 0, "homogenize did NOT use the FrameBase.project verb"
    assert counts["inverse_metric"] > 0, (
        "homogenize did NOT normalise through the frame's gram_inverse arrow"
    )


# ════════════════════════════════════════════════════════════════════
# P6 (#281) — the adjoint-weighted (eigenvalue-consistent) collapse.
# Spec: .claude/plans/archive/p6_adjoint_verification_spec.md §4 (delta-refreshed);
# rules: orpheus/derivations/common/homogenization.py (T1/T1b/T2/T3).
# ════════════════════════════════════════════════════════════════════


def _nonselfadjoint_materials():
    """Strongly non-self-adjoint pair: asymmetric SigS + absorber gradient,
    χ ∦ νΣf across materials — φ* and φ differ materially in SHAPE."""
    m0 = _balanced_fissile(
        [0.20, 0.30], [0.10, 0.20], [2.4, 2.4], [1.0, 0.0],
        [[0.60, 0.10], [0.00, 0.90]],
    )
    m1 = _balanced_fissile(
        [0.35, 0.45], [0.05, 0.30], [2.4, 2.4], [1.0, 0.0],
        [[0.50, 0.02], [0.00, 0.85]],
    )
    return {0: m0, 1: m1}


@pytest.fixture(scope="module")
def tilted_pair():
    """(materials, fwd, adj) on the 8-cell vacuum→reflective slab (C1/C3/Cχ)."""
    from orpheus.sn.solver import solve_sn_adjoint

    materials = _nonselfadjoint_materials()
    fine = Mesh1D(
        edges=_FINE_EDGES, mat_ids=_MAT_IDS, coord=CoordSystem.CARTESIAN,
        bc_left=BC("vacuum"), bc_right=BC("reflective"),
    )
    quad = Quadrature.gauss_legendre(n_ordinates=8)
    fwd = solve_sn(materials, fine, quad, scattering_order=0)
    adj = solve_sn_adjoint(materials, fine, quad, scattering_order=0)
    return materials, fwd, adj


def _flat_pair(sol, adj):
    """(phi, phi_star, rho, V, mat_ids) in the (n_fine, ng) 'ij' order."""
    ng = sol.mesh.ng
    phi = np.asarray(sol.scalar_flux.values, float).reshape(ng, -1).T
    phis = np.asarray(adj.scalar_flux.values, float).reshape(ng, -1).T
    w = np.asarray(sol.mesh.quad.weights, float)
    psi = np.asarray(sol.angular_flux.interior.values, float)
    psis = np.asarray(adj.angular_flux.interior.values, float)
    rho = np.einsum("n,n...->...", w, psis * psi).reshape(ng, -1).T
    V = np.asarray(sol.mesh.volumes).ravel()
    return phi, phis, rho, V


class TestAdjointDegeneratePins:
    """§4.0 tooth 1: the no-arg default ≡ the explicit ``adjoint=None`` at
    0-ULP on every channel (tooth 2 — no shared drift — is the existing
    forward suite above, whose hand-loop rate gates stay green)."""

    def test_homogenize_no_arg_equals_explicit_none_bitwise(self, solution):
        coarse = _coarse_two_region()
        a = solution.homogenize(coarse)
        b = solution.homogenize(coarse, adjoint=None)
        for R in a.materials:
            ma, mb = a.materials[R], b.materials[R]
            for ch in ("SigT", "SigC", "SigL", "SigF", "SigP", "chi"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(ma, ch)), np.asarray(getattr(mb, ch)),
                    err_msg=f"{ch} not bit-identical between no-arg and None",
                )
            for la, lb in zip(ma.SigS, mb.SigS):
                np.testing.assert_array_equal(
                    np.asarray(la.todense()), np.asarray(lb.todense()),
                )


@pytest.mark.verifies("sn-homogenization-adjoint-weighted")
@pytest.mark.verifies("sn-homogenization-bilinear")
class TestC1AdjointWeightedDiscriminator:
    """C1: every channel class equals its B1-derived hand rule (structurally
    independent per-region Python loops), differs from the forward
    degenerate, and the fixture proves itself non-dud (normalized shapes).

    Honest scope (qa NIT-4): the elementwise FOLDS (ρ = Σ_n w ψ*ψ, ι, p)
    share their numpy idiom between these hand loops and production — the
    COLLAPSE (region membership, the /Σ ratio, axis order) is what is
    structurally independent here; the folds' formula-correctness is owned
    by the derivation module's exact theorems, and C3/C5 capture the
    actual weight arrays production constructs."""

    def test_dud_guard_importance_shape_differs_materially(self, tilted_pair):
        """The bilinear is φ*-scale-invariant — compare normalized SHAPES."""
        _, fwd, adj = tilted_pair
        phi, phis, _, _ = _flat_pair(fwd, adj)
        a = phis / np.linalg.norm(phis)
        b = phi / np.linalg.norm(phi)
        assert not np.allclose(a, b, rtol=1e-2, atol=1e-3), (
            "fixture too self-adjoint: φ*/‖φ*‖ ≈ φ/‖φ‖ — C1 proves nothing"
        )

    def test_vector_channels_match_pair_weight_rule(self, tilted_pair):
        """SigC/SigL/SigF: Σ_R = Σ V φ*Σφ / Σ V φ*φ (T1) per (R, g)."""
        materials, fwd, adj = tilted_pair
        mm = fwd.homogenize(_coarse_two_region(), adjoint=adj)
        mm_f = fwd.homogenize(_coarse_two_region())
        phi, phis, _, V = _flat_pair(fwd, adj)
        discriminated = False
        for R, sel in enumerate(_fine_region_indices(np.array([0.0, 2.0, 4.0]))):
            for ch in ("SigC", "SigL", "SigF"):
                fine_ch = np.array(
                    [getattr(materials[_MAT_IDS[i]], ch) for i in sel]
                )
                for g in range(NG):
                    w = V[sel] * phis[sel, g] * phi[sel, g]
                    ref = float((w * fine_ch[:, g]).sum() / w.sum())
                    got = float(getattr(mm.materials[R], ch)[g])
                    np.testing.assert_allclose(got, ref, rtol=1e-12)
                    if abs(got - float(getattr(mm_f.materials[R], ch)[g])) > 1e-6:
                        discriminated = True
        assert discriminated, "adjoint-weighted ≡ forward everywhere: dud fixture"

    def test_sigma_t_matches_angular_pairing_rule(self, tilted_pair):
        """SigT: the T1b collision rule — weight ρ = Σ_n w ψ*ψ (user-ruled
        exact angular pairing), NOT the scalar φ*⊙φ."""
        materials, fwd, adj = tilted_pair
        mm = fwd.homogenize(_coarse_two_region(), adjoint=adj)
        phi, phis, rho, V = _flat_pair(fwd, adj)
        for R, sel in enumerate(_fine_region_indices(np.array([0.0, 2.0, 4.0]))):
            sigt = np.array([materials[_MAT_IDS[i]].SigT for i in sel])
            for g in range(NG):
                w_ang = V[sel] * rho[sel, g]
                ref = float((w_ang * sigt[:, g]).sum() / w_ang.sum())
                np.testing.assert_allclose(mm.materials[R].SigT[g], ref, rtol=1e-12)
                # Activation guard (qa SHOULD-1 fix): the fixture must make
                # the ρ-rule genuinely differ from the scalar φ*⊙φ rule
                # (measured gap ~6.5e-4), else the SigT gate loses its
                # T1b-vs-scalar discriminating power. An honest, reddenable
                # guard — the prior spelling was a tautology (P or ¬P).
                w_sc = V[sel] * phis[sel, g] * phi[sel, g]
                ref_scalar = float((w_sc * sigt[:, g]).sum() / w_sc.sum())
                assert not np.isclose(ref, ref_scalar, rtol=1e-8), (
                    f"R={R}, g={g}: the angular ρ rule coincides with the "
                    f"scalar pair rule — fixture too isotropic to pin T1b"
                )

    def test_matrix_channels_match_per_pair_rule(self, tilted_pair):
        """SigS[ℓ]: the T2 per-pair sink×source rule per (R, g', g)."""
        materials, fwd, adj = tilted_pair
        mm = fwd.homogenize(_coarse_two_region(), adjoint=adj)
        phi, phis, _, V = _flat_pair(fwd, adj)
        for R, sel in enumerate(_fine_region_indices(np.array([0.0, 2.0, 4.0]))):
            S = np.array(
                [np.asarray(materials[_MAT_IDS[i]].SigS[0].todense()) for i in sel]
            )
            got = np.asarray(mm.materials[R].SigS[0].todense())
            for gf in range(NG):
                for gt in range(NG):
                    w = V[sel] * phis[sel, gt] * phi[sel, gf]
                    ref = float((w * S[:, gf, gt]).sum() / w.sum())
                    np.testing.assert_allclose(got[gf, gt], ref, rtol=1e-12)

    def test_fission_dyad_matches_mixed_fold_and_canonical_chi(self, tilted_pair):
        """SigP: the T3 mixed-fold rule (ι numerator / ι̃ denominator);
        χ: the canonical adjoint-weighted-emission convex average."""
        materials, fwd, adj = tilted_pair
        mm = fwd.homogenize(_coarse_two_region(), adjoint=adj)
        phi, phis, _, V = _flat_pair(fwd, adj)
        for R, sel in enumerate(_fine_region_indices(np.array([0.0, 2.0, 4.0]))):
            chi_f = np.array([materials[_MAT_IDS[i]].chi for i in sel])
            nsf = np.array([materials[_MAT_IDS[i]].SigP for i in sel])
            iota = (phis[sel] * chi_f).sum(axis=1)
            p = (nsf * phi[sel]).sum(axis=1)
            q = V[sel] * iota * p
            chi_ref = (q[:, None] * chi_f).sum(axis=0) / q.sum()
            np.testing.assert_allclose(mm.materials[R].chi, chi_ref, rtol=1e-12)
            iota_t = (phis[sel] * chi_ref[None, :]).sum(axis=1)
            for gp in range(NG):
                num = float((V[sel] * iota * nsf[:, gp] * phi[sel, gp]).sum())
                den = float((V[sel] * iota_t * phi[sel, gp]).sum())
                np.testing.assert_allclose(
                    mm.materials[R].SigP[gp], num / den, rtol=1e-12,
                )

    def test_worth_exact_collapse_breaks_balance_as_derived(self, tilted_pair):
        """T4 pinned live: the adjoint-collapsed Mixture does NOT satisfy the
        total-XS balance identity (the classical reactivity-vs-rates
        property) — while the forward collapse does. Never assert_balanced
        on an adjoint-collapsed Mixture; this gate pins the imbalance as the
        DERIVED property, not an accident."""
        _, fwd, adj = tilted_pair
        mm_a = fwd.homogenize(_coarse_two_region(), adjoint=adj)
        mm_f = fwd.homogenize(_coarse_two_region())
        for R in range(2):
            mm_f.materials[R].assert_balanced(atol=1e-9)   # forward: balanced
            m = mm_a.materials[R]
            resid = np.abs(
                m.SigT - (
                    m.SigC + m.SigL + m.SigF
                    + np.array(m.SigS[0].sum(axis=1)).ravel()
                    + np.array(m.Sig2[0].sum(axis=1)).ravel()
                )
            ).max()
            assert resid > 1e-9, (
                f"adjoint-collapsed region {R} unexpectedly balanced "
                f"(resid={resid}) — check the worth-exact taxonomy wiring"
            )


class TestC3WeightCaptureSentinel:
    """C3 (Mode-11 CAPTURE upgrade): the weights actually handed to the
    test-basis constructions ARE the derived ones — the pair product φ*⊙φ
    (not bare φ*, not φ — the ``frame.rst:3458`` trap's committed catcher),
    the angular ρ, and the emission fold ι·p. Fires regardless of how close
    φ* is to φ."""

    def test_bilinear_frames_receive_the_derived_weights(self, tilted_pair, monkeypatch):
        from orpheus.numerics.basis.weighted_indicator_basis import (
            WeightedIndicatorBasis,
        )

        _, fwd, adj = tilted_pair
        phi, phis, rho, _ = _flat_pair(fwd, adj)
        captured: list[np.ndarray] = []
        orig = WeightedIndicatorBasis.__init__

        def spy(self, basis, weight, *a, **k):
            captured.append(np.asarray(weight, dtype=float))
            return orig(self, basis, weight, *a, **k)

        monkeypatch.setattr(WeightedIndicatorBasis, "__init__", spy)
        fwd.homogenize(_coarse_two_region(), adjoint=adj)

        assert len(captured) >= 3, "the three bilinear frames were not built"

        def _seen(expected):
            return any(
                c.shape == np.shape(expected) and np.array_equal(c, expected)
                for c in captured
            )

        assert _seen(phis * phi), (
            "no frame received the PAIR weight φ*⊙φ — bare φ* or φ wired?"
        )
        assert _seen(rho), "no frame received the angular collision weight ρ"
        assert not _seen(phis), "a frame received BARE φ* (the φ→φ* trap)"
        # The forward path must NOT have been silently taken:
        assert not _seen(phi), (
            "a frame received the forward weight φ — adjoint dropped"
        )


class TestCxChiSimplexPositiveControl:
    """Cχ: the canonical χ rule stays a simplex — constructing the
    adjoint-collapsed Mixture must NOT raise (EmissionSpectrum validates via
    explicit ValueError — Mode-8-safe), and χ sums to 1 on producing
    regions."""

    def test_adjoint_collapsed_chi_is_valid_simplex(self, tilted_pair):
        _, fwd, adj = tilted_pair
        mm = fwd.homogenize(_coarse_two_region(), adjoint=adj)   # must not raise
        for mix in mm.materials.values():
            assert mix.chi.sum() == pytest.approx(1.0, abs=1e-12)
            assert np.all(mix.chi >= -1e-15)


# ── n-D activation: 2-D rate preservation (the dropped-guard capability) ──

@pytest.mark.verifies("sn-homogenization-rate-preservation")
def test_homogenize_2d_rate_preservation(materials):
    """2-D homogenize preserves reaction rates — the n-D path the dropped guard opens.

    P3 drops the ``ndim != 1`` guard; this exercises the n-D membership
    (``ravel_multi_index`` ``"ij"``) and the ``(ng, nx, ny) → (n_fine, ng)`` flatten
    end-to-end through a REAL 2-D ``solve_sn``, checked against a structurally-
    independent per-region Python loop (``vv-principles`` L11).  A flux tilt
    (vacuum-x / reflective elsewhere) keeps φ non-flat so the φ-weighting is genuinely
    activated.
    """
    mat_map = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]], dtype=int,
    )
    fine = Mesh2D(
        edges_x=np.linspace(0.0, 4.0, 5), edges_y=np.linspace(0.0, 4.0, 5),
        mat_map=mat_map, coord=CoordSystem.CARTESIAN,
        bc_xmin=BC("vacuum"), bc_xmax=BC("reflective"),
        bc_ymin=BC("reflective"), bc_ymax=BC("reflective"),
    )
    quad = Quadrature.level_symmetric(sn_order=4)
    sol = solve_sn(materials, fine, quad, scattering_order=0)

    coarse = Mesh2D(
        edges_x=np.array([0.0, 2.0, 4.0]), edges_y=np.array([0.0, 2.0, 4.0]),
        mat_map=np.zeros((2, 2), dtype=int), coord=CoordSystem.CARTESIAN,
        bc_xmin=BC("vacuum"), bc_xmax=BC("reflective"),
        bc_ymin=BC("reflective"), bc_ymax=BC("reflective"),
    )
    mm = sol.homogenize(coarse)
    assert mm.ndim == 2 and len(mm.materials) == 4

    phi = sol.scalar_flux.values                  # (ng, nx, ny)
    ng, nx, ny = phi.shape
    V = np.asarray(sol.mesh.volumes).ravel()      # (n_fine,) "ij"
    phi_flat = phi.reshape(ng, -1)                # (ng, n_fine) "ij"
    mat_flat = fine.mat_map.ravel()
    fx, fy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    # Coarse cell flat index of each fine cell — the IndicatorBasis "ij" raveling.
    coarse_of_fine = ((fx // 2) * 2 + (fy // 2)).ravel()

    for channel in ("SigT", "SigC", "SigF"):
        for R in range(4):
            sel = np.where(coarse_of_fine == R)[0]
            sig_eff = getattr(mm.materials[R], channel)
            for g in range(ng):
                phi_R = float((V[sel] * phi_flat[g, sel]).sum())
                rate_homog = sig_eff[g] * phi_R
                rate_ref = float(sum(
                    V[i] * phi_flat[g, i]
                    * getattr(materials[mat_flat[i]], channel)[g]
                    for i in sel
                ))
                assert rate_homog == pytest.approx(rate_ref, abs=1e-12, rel=1e-12), (
                    f"2-D {channel} rate not preserved in coarse {R}, group {g}"
                )
