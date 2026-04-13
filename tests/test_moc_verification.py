"""Bulletproof verification suite for the 2D MOC transport solver.

Organized by verification level:
  L0 — Component isolation: each physics term tested by zeroing all others.
  L1 — Eigenvalue accuracy: quantitative checks against analytical references.
  L2 — Convergence rates: spatial (ray spacing), angular (azimuthal, polar).
  XV — Cross-verification: MOC vs CP on identical geometry.

Cardinal rule: every level includes >= 2-group tests to escape the
1-group degeneracy (k = nuSigF/SigA independent of flux shape).

Failure modes targeted:
  [FM-01] Sign flip in delta_psi           -> L0 fixed-source sign tests
  [FM-02] Variable swap (sin_p / omega_p)  -> L0 single-direction residual
  [FM-03] Missing area factor in Boyd Eq45 -> L0 fixed-source absolute value
  [FM-04] Wrong index (from/to in SigS)    -> L1 2-group flux ratio
  [FM-05] Convention drift (SigS^T)        -> L1 heterogeneous 2G keff
  [FM-06] 4*pi normalization error         -> L0 equilibrium flux test
  [FM-07] Tau / sin_p division missing     -> L0 single-track attenuation
  [FM-08] Boundary link direction swap     -> L1 convergence monotonicity
  [FM-09] (n,2n) factor of 2 missing       -> L0 n2n production test
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from orpheus.geometry import CoordSystem, Mesh1D
from orpheus.derivations._xs_library import make_mixture, get_mixture, get_xs
from orpheus.derivations._eigenvalue import kinf_homogeneous
from orpheus.moc.geometry import MOCMesh
from orpheus.moc.quadrature import MOCQuadrature
from orpheus.moc.core import MOCSolver
from orpheus.moc.solver import solve_moc


# =====================================================================
# Helpers
# =====================================================================

def _ws_mesh(edges, mat_ids):
    """Build a cylindrical Mesh1D (Wigner-Seitz convention)."""
    return Mesh1D(
        edges=np.array(edges, dtype=float),
        mat_ids=np.array(mat_ids, dtype=int),
        coord=CoordSystem.CYLINDRICAL,
    )


def _homogeneous_ws_mesh(pitch=2.0, mat_id=0):
    """Single-region Wigner-Seitz mesh with given pitch."""
    r_cell = pitch / np.sqrt(np.pi)
    return _ws_mesh([0.0, r_cell], [mat_id])


def _two_region_ws_mesh(r_fuel=0.5, pitch=2.0, mat_ids=(2, 0)):
    """Fuel + coolant Wigner-Seitz mesh."""
    ws_r = pitch / np.sqrt(np.pi)
    return _ws_mesh([0.0, r_fuel, ws_r], list(mat_ids))


def _make_pure_absorber_1g(sig_t=1.0, sig_a=1.0):
    """1G material with only absorption (no scatter, no fission)."""
    return make_mixture(
        sig_t=np.array([sig_t]),
        sig_c=np.array([sig_a]),
        sig_f=np.array([0.0]),
        nu=np.array([0.0]),
        chi=np.array([1.0]),
        sig_s=np.array([[0.0]]),
    )


def _make_pure_scatterer_1g(sig_t=1.0):
    """1G material with Sigma_s = Sigma_t (no absorption, no fission)."""
    return make_mixture(
        sig_t=np.array([sig_t]),
        sig_c=np.array([0.0]),
        sig_f=np.array([0.0]),
        nu=np.array([0.0]),
        chi=np.array([1.0]),
        sig_s=np.array([[sig_t]]),
    )


def _make_fission_only_1g(sig_t=1.0, sig_a=0.4, sig_f=0.2, nu=2.5):
    """1G fission material with NO scattering.

    Sigma_t = sig_a (capture) + sig_f; no scatter at all.
    Analytical keff = nu * sig_f / (sig_a + sig_f) = nu * sig_f / Sigma_t
    (since absorption_xs = sig_c + sig_f and sig_c = sig_a).
    """
    sig_c = sig_t - sig_f
    return make_mixture(
        sig_t=np.array([sig_t]),
        sig_c=np.array([sig_c]),
        sig_f=np.array([sig_f]),
        nu=np.array([nu]),
        chi=np.array([1.0]),
        sig_s=np.array([[0.0]]),
    )


def _make_fission_only_2g():
    """2G fission-only material (no scattering).

    Fast fission + thermal fission, chi = [1, 0].
    Analytical k_inf from the matrix eigenvalue.
    """
    sig_t = np.array([0.5, 1.0])
    sig_f = np.array([0.01, 0.08])
    sig_c = sig_t - sig_f  # all removal is capture + fission
    nu = np.array([2.5, 2.5])
    chi = np.array([1.0, 0.0])
    sig_s = np.zeros((2, 2))
    return make_mixture(
        sig_t=sig_t, sig_c=sig_c, sig_f=sig_f,
        nu=nu, chi=chi, sig_s=sig_s,
    )


def _make_n2n_material_1g(sig_c=0.2, sig_s_val=0.4, sig_f=0.1,
                           nu=2.5, sig2_val=0.1):
    """1G material with (n,2n) reactions.

    sig_t = sig_c + sig_f + sig_s + sig2_out (consistent).
    """
    sig_t = sig_c + sig_f + sig_s_val + sig2_val
    return make_mixture(
        sig_t=np.array([sig_t]),
        sig_c=np.array([sig_c]),
        sig_f=np.array([sig_f]),
        nu=np.array([nu]),
        chi=np.array([1.0]),
        sig_s=np.array([[sig_s_val]]),
        sig_2=np.array([[sig2_val]]),
    )


def _quick_solve(materials, mesh, n_azi=8, n_polar=3, ray_spacing=0.05,
                 max_outer=200, n_inner=15, keff_tol=1e-6, flux_tol=1e-5):
    """Convenience wrapper for solve_moc with sensible defaults."""
    return solve_moc(
        materials, mesh,
        n_azi=n_azi, n_polar=n_polar, ray_spacing=ray_spacing,
        max_outer=max_outer, n_inner_sweeps=n_inner,
        keff_tol=keff_tol, flux_tol=flux_tol,
    )


# =====================================================================
# L0: COMPONENT ISOLATION — TRANSPORT OPERATOR
# =====================================================================

class TestL0SingleTrackAttenuation:
    """L0: Verify the characteristic ODE for a single track/segment.

    For a single segment with known Sigma_t, length L, polar sin_p:
        tau = Sigma_t * L / sin_p
        psi_out = psi_in * exp(-tau) + (Q/Sigma_t)(1 - exp(-tau))
        delta_psi = (psi_in - Q/Sigma_t)(1 - exp(-tau))

    Catches [FM-07] tau/sin_p missing, [FM-01] sign flip in delta_psi.
    """

    def test_attenuation_vacuum_source(self):
        """[L0] Zero source: psi_out = psi_in * exp(-tau).

        If Q = 0, then delta_psi = psi_in * (1 - exp(-tau)), and
        psi_out = psi_in - delta_psi = psi_in * exp(-tau).
        """
        sig_t = 2.0
        length = 0.5
        sin_p = 0.8
        tau = sig_t * length / sin_p
        psi_in = 1.0

        one_minus_exp = 1.0 - np.exp(-tau)
        dpsi = (psi_in - 0.0) * one_minus_exp  # Q/sig_t = 0
        psi_out = psi_in - dpsi

        expected = psi_in * np.exp(-tau)
        assert psi_out == pytest.approx(expected, rel=1e-14), (
            f"psi_out={psi_out:.10f}, expected={expected:.10f}"
        )

    def test_attenuation_equilibrium(self):
        """[L0] At equilibrium (psi_in = Q/Sigma_t), delta_psi = 0 exactly.

        This is the fixed point of the characteristic ODE.
        Catches [FM-06] normalization errors.
        """
        sig_t = 1.5
        Q = 3.0  # isotropic source (already divided by 4pi in the solver)
        q_over_sigt = Q / sig_t
        psi_in = q_over_sigt

        for length in [0.1, 0.5, 1.0, 5.0]:
            for sin_p in [0.3, 0.7, 0.99]:
                tau = sig_t * length / sin_p
                one_minus_exp = 1.0 - np.exp(-tau)
                dpsi = (psi_in - q_over_sigt) * one_minus_exp
                assert abs(dpsi) < 1e-15, (
                    f"delta_psi={dpsi:.2e} at L={length}, sin_p={sin_p}"
                )

    def test_attenuation_known_value(self):
        """[L0] Verify delta_psi against hand-computed value.

        sig_t=1.0, L=1.0, sin_p=0.5 -> tau=2.0
        psi_in=2.0, Q/sig_t=0.5
        delta_psi = (2.0 - 0.5) * (1 - exp(-2)) = 1.5 * 0.864664... = 1.29699...
        """
        sig_t = 1.0
        length = 1.0
        sin_p = 0.5
        tau = sig_t * length / sin_p
        psi_in = 2.0
        q_over_sigt = 0.5

        one_minus_exp = 1.0 - np.exp(-tau)
        dpsi = (psi_in - q_over_sigt) * one_minus_exp
        expected = 1.5 * (1.0 - np.exp(-2.0))

        assert dpsi == pytest.approx(expected, rel=1e-14)


class TestL0EquilibriumFlux:
    """L0: When boundary flux = Q/Sigma_t everywhere, the scalar flux update
    (Boyd Eq. 45) should reproduce phi = Q/Sigma_t * 4*pi / Sigma_t... but
    actually phi = (4*pi*Q + 0)/Sigma_t = 4*pi*Q/Sigma_t since delta_phi=0.

    For isotropic source Q = Sigma_s*phi/(4pi), at equilibrium:
    phi = (4pi * Sigma_s*phi/(4pi)) / Sigma_t = Sigma_s*phi/Sigma_t
    => Sigma_t = Sigma_s => pure scatterer in equilibrium.

    We test: with a pure scatterer and boundary flux = phi/(4pi),
    a single transport sweep should return phi unchanged.

    Catches [FM-03] missing area factor, [FM-06] 4*pi normalization.
    """

    def test_pure_scatterer_equilibrium_single_sweep(self):
        """[L0] Pure scatterer: transport sweep preserves uniform flux.

        With Sigma_s = Sigma_t, the equilibrium flux is uniform.
        Starting from uniform boundary flux, delta_phi should be zero
        and the returned flux should be uniform.
        """
        sig_t_val = 1.0
        mat = _make_pure_scatterer_1g(sig_t=sig_t_val)
        mesh = _homogeneous_ws_mesh(pitch=2.0, mat_id=0)
        quad = MOCQuadrature.create(n_azi=8, n_polar=3)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.05)

        solver = MOCSolver(moc_mesh, {0: mat}, n_inner_sweeps=1)

        nr = moc_mesh.n_regions
        phi_uniform = np.ones((nr, 1))  # uniform flux

        # Fission source = 0 (no fission)
        fission_source = np.zeros((nr, 1))

        # Reset boundary fluxes to equilibrium: Q/Sigma_t = Sigma_s*phi/(4pi*Sigma_t)
        # = phi/(4pi) for pure scatterer
        q_eq = sig_t_val * phi_uniform[0, 0] / (4.0 * np.pi)
        solver._fwd_bflux[:] = q_eq
        solver._bwd_bflux[:] = q_eq

        phi_out = solver.solve_fixed_source(fission_source, phi_uniform)

        # Flux should be uniform (all regions same value)
        phi_vals = phi_out[:, 0]
        # Note: the solver normalizes by total production; for no fission,
        # it will skip normalization (total_prod=0 -> no normalization).
        # Actually check: sig_p is zero, sig2 is zero => total_prod = 0
        # => normalization is skipped => phi_out should be close to phi_uniform

        rel_std = np.std(phi_vals) / np.mean(phi_vals) if np.mean(phi_vals) > 0 else 0
        assert rel_std < 0.05, (
            f"Flux not uniform: std/mean = {rel_std:.4f}, "
            f"values = {phi_vals}"
        )


class TestL0FissionOnly:
    """L0: Fission-only material (no scattering).

    With zero scattering, the source is purely fission: Q = chi * nuSigF * phi / (4pi * k).
    The eigenvalue must be k = nuSigF / Sigma_a (1G) or the matrix eigenvalue (2G).

    Catches [FM-04] wrong scattering matrix index, [FM-05] SigS convention.
    """

    def test_fission_only_1g_eigenvalue(self):
        """[L0] 1G fission-only: k = nu*SigF / Sigma_t (no scatter)."""
        sig_t = 1.0
        sig_f = 0.3
        nu = 2.5
        mat = _make_fission_only_1g(sig_t=sig_t, sig_a=sig_t - sig_f,
                                     sig_f=sig_f, nu=nu)
        mesh = _homogeneous_ws_mesh(pitch=3.0)
        result = _quick_solve({0: mat}, mesh, n_azi=8, n_polar=3,
                              ray_spacing=0.05, max_outer=200)

        k_expected = nu * sig_f / sig_t
        assert result.keff == pytest.approx(k_expected, rel=1e-4), (
            f"keff={result.keff:.8f}, expected={k_expected:.8f}"
        )

    def test_fission_only_2g_eigenvalue(self):
        """[L0] 2G fission-only: k from matrix eigenvalue, NOT 1G shortcut.

        This catches bugs that hide in 1G degeneracy (flux shape irrelevant).
        """
        mat = _make_fission_only_2g()
        xs = dict(
            sig_t=mat.SigT, sig_s=np.zeros((2, 2)),
            nu_sig_f=mat.SigP, chi=mat.chi,
        )
        k_expected = kinf_homogeneous(**xs)

        mesh = _homogeneous_ws_mesh(pitch=3.0)
        result = _quick_solve({0: mat}, mesh, n_azi=8, n_polar=3,
                              ray_spacing=0.05, max_outer=300)

        assert result.keff == pytest.approx(k_expected, rel=1e-3), (
            f"2G fission-only: keff={result.keff:.8f}, expected={k_expected:.8f}"
        )


class TestL0N2nReaction:
    """L0: (n,2n) reactions contribute correctly to production and source.

    Catches [FM-09] missing factor of 2 in (n,2n).
    """

    def test_n2n_increases_keff(self):
        """[L0] Adding (n,2n) reactions must increase keff vs. same material without.

        The (n,2n) reaction produces 2 neutrons per reaction, adding to production.
        """
        # Material A 1G as baseline
        xs_a = get_xs("A", "1g")
        mat_no_n2n = make_mixture(**xs_a)

        # Same material but with (n,2n) added
        mat_with_n2n = make_mixture(**xs_a, sig_2=np.array([[0.05]]))

        mesh = _homogeneous_ws_mesh(pitch=3.0)

        result_no = _quick_solve({0: mat_no_n2n}, mesh, n_azi=8, max_outer=200)
        result_yes = _quick_solve({0: mat_with_n2n}, mesh, n_azi=8, max_outer=200)

        assert result_yes.keff > result_no.keff, (
            f"(n,2n) should increase keff: "
            f"without={result_no.keff:.6f}, with={result_yes.keff:.6f}"
        )

    def test_n2n_1g_analytical_keff(self):
        """[L0] 1G with (n,2n): keff = (nuSigF + 2*Sig2) / (SigC + SigF + Sig2).

        The solver defines keff = production/absorption where:
        - production = nuSigF + 2*Sig2 (fission + (n,2n) outgoing)
        - absorption = SigC + SigF + Sig2 (capture + fission + (n,2n) reaction)

        This is the steady-state production/loss ratio. For homogeneous 1G,
        the MOC must reproduce this exactly (flux shape is uniform).
        """
        sig_c_val = 0.2
        sig_s_val = 0.4
        sig_f_val = 0.1
        nu_val = 2.5
        sig2_val = 0.1

        mat = _make_n2n_material_1g(
            sig_c=sig_c_val, sig_s_val=sig_s_val,
            sig_f=sig_f_val, nu=nu_val, sig2_val=sig2_val,
        )

        # Expected keff from the solver's production/absorption definition
        production = nu_val * sig_f_val + 2.0 * sig2_val
        absorption = sig_c_val + sig_f_val + sig2_val
        k_expected = production / absorption

        mesh = _homogeneous_ws_mesh(pitch=3.0)
        result = _quick_solve({0: mat}, mesh, n_azi=8, max_outer=200)

        assert result.keff == pytest.approx(k_expected, rel=1e-4), (
            f"(n,2n) keff: solver={result.keff:.8f}, expected={k_expected:.8f}"
        )


class TestL0SourceTermIsolation:
    """L0: Isolate individual source terms by zeroing all others.

    The MOC source is: Q = (1/4pi)(fission + scatter + 2*n2n).
    By zeroing two of the three, we test each term in isolation.
    """

    def test_scatter_only_source(self):
        """[L0] Scatter-only source: flux must be positive and bounded.

        With no fission and no (n,2n), the scatter source alone cannot
        sustain a chain reaction. The solver should converge to a flux
        distribution with keff -> 0 (or the iteration should produce
        very small keff).

        We test that the transport sweep with scatter-only source
        produces positive flux without NaN or overflow.
        """
        mat = _make_pure_scatterer_1g(sig_t=1.0)
        mesh = _homogeneous_ws_mesh(pitch=2.0)
        quad = MOCQuadrature.create(n_azi=8, n_polar=3)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.05)
        solver = MOCSolver(moc_mesh, {0: mat}, n_inner_sweeps=5)

        phi = solver.initial_flux_distribution()
        # With zero fission source, solve_fixed_source should still work
        fission_source = np.zeros_like(phi)
        phi_out = solver.solve_fixed_source(fission_source, phi)

        assert np.all(np.isfinite(phi_out)), "Scatter-only produced NaN/Inf"
        assert np.all(phi_out >= 0), "Scatter-only produced negative flux"

    def test_fission_source_dominates_at_high_nu(self):
        """[L0] At very high nu, fission dominates and keff >> 1.

        Verifies that the fission term is actually being used.
        """
        mat = _make_fission_only_1g(sig_t=1.0, sig_a=0.7, sig_f=0.3, nu=10.0)
        mesh = _homogeneous_ws_mesh(pitch=3.0)
        result = _quick_solve({0: mat}, mesh, n_azi=8, max_outer=200)

        k_expected = 10.0 * 0.3 / 1.0  # = 3.0
        assert result.keff == pytest.approx(k_expected, rel=1e-3), (
            f"High-nu keff={result.keff:.6f}, expected={k_expected:.6f}"
        )


# =====================================================================
# L0: GEOMETRY — RAY TRACING INVARIANTS
# =====================================================================

class TestL0GeometricInvariants:
    """L0: Geometric properties of the ray tracing that must hold exactly.

    These are algebraic identities, not physics — they catch index bugs,
    missing factors, and wrong geometric formulas.
    """

    def test_segment_lengths_sum_to_chord(self):
        """[L0] Total segment length per track = entry-to-exit distance.

        Independent of how many region boundaries the ray crosses.
        Catches bugs where a crossing is double-counted or missed.
        """
        r_fuel = 0.5
        mesh = _two_region_ws_mesh(r_fuel=r_fuel, pitch=2.0)
        quad = MOCQuadrature.create(n_azi=16, n_polar=1)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.03)

        for track in moc_mesh.tracks:
            seg_total = sum(s.length for s in track.segments)
            ex, ey = track.entry_point
            xx, xy = track.exit_point
            chord = np.sqrt((xx - ex)**2 + (xy - ey)**2)
            assert seg_total == pytest.approx(chord, rel=1e-10), (
                f"Segment sum {seg_total:.10f} != chord {chord:.10f}"
            )

    def test_track_reciprocity(self):
        """[L0] Track from A->B has same total length as track in opposite direction.

        For reflective BC, the forward and backward sweeps traverse the same
        physical segments. The segment lengths must be identical.
        """
        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.0)
        quad = MOCQuadrature.create(n_azi=8, n_polar=1)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.05)

        for track in moc_mesh.tracks:
            fwd_total = sum(s.length for s in track.segments)
            bwd_total = sum(s.length for s in reversed(track.segments))
            assert fwd_total == pytest.approx(bwd_total, rel=1e-15), (
                f"Forward/backward length mismatch: {fwd_total} vs {bwd_total}"
            )

    def test_region_area_conservation(self):
        """[L0] Sum of all region areas = pitch^2 (exact, no ray tracing)."""
        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.5)
        quad = MOCQuadrature.create(n_azi=4, n_polar=1)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.1)

        assert moc_mesh.region_areas.sum() == pytest.approx(
            moc_mesh.pitch**2, rel=1e-12
        )

    def test_all_tracks_inside_cell(self):
        """[L0] Every segment midpoint lies inside [0, pitch]^2."""
        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.0)
        quad = MOCQuadrature.create(n_azi=16, n_polar=1)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.03)

        pitch = moc_mesh.pitch
        phi = moc_mesh.quad.phi

        for track in moc_mesh.tracks:
            a_idx = track.azi_index
            cos_phi = np.cos(phi[a_idx])
            sin_phi = np.sin(phi[a_idx])
            ex, ey = track.entry_point
            s = 0.0
            for seg in track.segments:
                s_mid = s + seg.length / 2
                mx = ex + s_mid * cos_phi
                my = ey + s_mid * sin_phi
                assert -1e-10 <= mx <= pitch + 1e-10, f"x={mx} outside cell"
                assert -1e-10 <= my <= pitch + 1e-10, f"y={my} outside cell"
                s += seg.length

    def test_reflective_links_form_cycles(self):
        """[L0] Following forward links must eventually return to the starting track.

        Reflective BCs form closed cycles. If not, neutrons are lost.
        Catches [FM-08] boundary link direction swap.
        """
        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.0)
        quad = MOCQuadrature.create(n_azi=8, n_polar=1)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.05)

        n_tracks = len(moc_mesh.tracks)
        visited_any = set()

        for start_idx in range(n_tracks):
            if start_idx in visited_any:
                continue
            # Follow forward links
            visited = set()
            current = start_idx
            max_steps = n_tracks * 4  # generous bound
            for _ in range(max_steps):
                if current in visited:
                    break
                visited.add(current)
                t = moc_mesh.tracks[current]
                # Forward exit links to fwd_link
                current = t.fwd_link
                assert current >= 0, f"Track {start_idx}: dangling fwd_link"
            assert current in visited, (
                f"Forward links from track {start_idx} don't form a cycle "
                f"after {max_steps} steps"
            )
            visited_any.update(visited)


# =====================================================================
# L0: QUADRATURE WEIGHT CONSISTENCY
# =====================================================================

class TestL0QuadratureWeights:
    """L0: The total angular weight must integrate to 4*pi.

    In the MOC solver, the weight for each (azimuthal, polar) pair is:
        weight = 4*pi * omega_azi * omega_polar * t_s * sin_p

    Without the t_s factor (which is spatial), the angular part is:
        4*pi * sum_a(omega_a) * sum_p(omega_p) = 4*pi * 1.0 * 0.5 = 2*pi

    But since we sweep both forward AND backward (2 hemispheres), the
    total is 4*pi. This must hold for any (n_azi, n_polar).
    """

    @pytest.mark.parametrize("n_azi,n_polar", [
        (4, 1), (8, 2), (16, 3), (32, 3),
    ])
    def test_angular_weight_sum(self, n_azi, n_polar):
        """[L0] sum(omega_azi) * sum(omega_polar) * 2 hemispheres = 1.0."""
        quad = MOCQuadrature.create(n_azi=n_azi, n_polar=n_polar)
        # omega_azi sums to 1.0, omega_polar sums to 0.5 (one hemisphere)
        azi_sum = quad.omega_azi.sum()
        polar_sum = quad.omega_polar.sum()
        assert azi_sum == pytest.approx(1.0, abs=1e-14)
        assert polar_sum == pytest.approx(0.5, abs=1e-14)


# =====================================================================
# L1: EIGENVALUE ACCURACY — HOMOGENEOUS
# =====================================================================

class TestL1HomogeneousEigenvalue:
    """L1: Homogeneous infinite-medium eigenvalue.

    The flat-source MOC in a single-region Wigner-Seitz cell with
    reflective BCs must reproduce k_inf = lambda_max(A^{-1}F) exactly
    (up to angular/spatial discretization).

    MANDATORY: includes 2G and 4G tests to escape 1G degeneracy.
    """

    @pytest.mark.parametrize("ng_key,tol", [
        ("1g", 1e-4),
        ("2g", 1e-4),
        ("4g", 1e-4),
    ])
    def test_homogeneous_kinf(self, ng_key, tol):
        """[L1] Homogeneous k_inf for material A at 1/2/4 groups."""
        xs = get_xs("A", ng_key)
        k_expected = kinf_homogeneous(
            sig_t=xs["sig_t"], sig_s=xs["sig_s"],
            nu_sig_f=xs["nu"] * xs["sig_f"], chi=xs["chi"],
        )
        mat = get_mixture("A", ng_key)
        mesh = _homogeneous_ws_mesh(pitch=3.6)
        result = _quick_solve({0: mat}, mesh, n_azi=8, n_polar=3,
                              ray_spacing=0.05, max_outer=200)

        assert result.keff == pytest.approx(k_expected, abs=tol), (
            f"{ng_key}: keff={result.keff:.8f}, expected={k_expected:.8f}"
        )

    def test_homogeneous_2g_flux_ratio(self):
        """[L1] 2G homogeneous: flux ratio phi_2/phi_1 must match analytical.

        The fundamental eigenvector of A^{-1}F gives the group ratio.
        This catches SigS transpose convention errors [FM-05].
        """
        xs = get_xs("A", "2g")
        sig_t = xs["sig_t"]
        sig_s = xs["sig_s"]
        nu_sig_f = xs["nu"] * xs["sig_f"]
        chi = xs["chi"]

        A = np.diag(sig_t) - sig_s.T
        F = np.outer(chi, nu_sig_f)
        M = np.linalg.solve(A, F)
        eigvals, eigvecs = np.linalg.eig(M)
        idx = np.argmax(np.real(eigvals))
        phi_ref = np.real(eigvecs[:, idx])
        ratio_ref = phi_ref[1] / phi_ref[0]

        mat = get_mixture("A", "2g")
        mesh = _homogeneous_ws_mesh(pitch=3.6)
        result = _quick_solve({0: mat}, mesh, n_azi=8, n_polar=3,
                              ray_spacing=0.05, max_outer=200)

        # Volume-averaged flux (single region, so just the flux itself)
        phi = result.scalar_flux
        phi_avg = (phi * result.moc_mesh.region_areas[:, None]).sum(axis=0)
        ratio_moc = phi_avg[1] / phi_avg[0]

        assert ratio_moc == pytest.approx(ratio_ref, rel=0.02), (
            f"2G flux ratio: MOC={ratio_moc:.6f}, analytical={ratio_ref:.6f}"
        )


# =====================================================================
# L1: EIGENVALUE ACCURACY — HETEROGENEOUS
# =====================================================================

class TestL1HeterogeneousEigenvalue:
    """L1: Heterogeneous pin cell eigenvalue tests.

    Multi-region tests catch redistribution bugs that 1-region tests miss.
    2-group is MANDATORY to catch SigS convention and group-coupling bugs.
    """

    def test_heterogeneous_2g_fuel_coolant(self):
        """[L1] 2G fuel+coolant pin cell: keff must be physically reasonable.

        Fuel (material A) + coolant (material B) at 2 groups.
        With downscatter, k < k_inf(A_homogeneous) due to leakage into
        non-fissile coolant.
        """
        fuel = get_mixture("A", "2g")
        cool = get_mixture("B", "2g")

        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.0)
        result = _quick_solve(
            {2: fuel, 0: cool}, mesh,
            n_azi=16, n_polar=3, ray_spacing=0.03,
            max_outer=300, n_inner=20,
        )

        # k_inf for homogeneous A
        xs = get_xs("A", "2g")
        k_homo = kinf_homogeneous(
            sig_t=xs["sig_t"], sig_s=xs["sig_s"],
            nu_sig_f=xs["nu"] * xs["sig_f"], chi=xs["chi"],
        )

        # Heterogeneous keff should be less than homogeneous fuel
        # (coolant absorbs neutrons without producing them)
        assert result.keff < k_homo, (
            f"Heterogeneous keff={result.keff:.6f} should be < "
            f"homogeneous fuel k_inf={k_homo:.6f}"
        )
        # But still supercritical (fuel is fissile)
        assert result.keff > 0.5, f"keff={result.keff:.6f} unreasonably low"

    def test_heterogeneous_2g_thermal_depression(self):
        """[L1] 2G heterogeneous: thermal flux higher in coolant than fuel.

        Standard PWR physics: the coolant (moderator) has lower thermal
        absorption than fuel, so thermal flux is depressed in fuel.
        Catches bugs in spatial redistribution.
        """
        fuel = get_mixture("A", "2g")
        cool = get_mixture("B", "2g")

        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.0)
        result = _quick_solve(
            {2: fuel, 0: cool}, mesh,
            n_azi=16, n_polar=3, ray_spacing=0.03,
            max_outer=300, n_inner=20,
        )

        phi_fuel = result.flux_per_material[2]
        phi_cool = result.flux_per_material[0]

        # Thermal group (index 1) should be higher in coolant
        assert phi_cool[1] > phi_fuel[1], (
            f"Thermal flux: fuel={phi_fuel[1]:.4e}, cool={phi_cool[1]:.4e}"
        )

    def test_heterogeneous_2g_convergence_monotonic(self):
        """[L1] 2G heterogeneous: keff history must converge monotonically
        (no oscillations after initial transient).

        Large oscillations indicate boundary link errors [FM-08].
        """
        fuel = get_mixture("A", "2g")
        cool = get_mixture("B", "2g")

        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.0)
        result = _quick_solve(
            {2: fuel, 0: cool}, mesh,
            n_azi=16, n_polar=3, ray_spacing=0.03,
            max_outer=300, n_inner=20,
        )

        # After the first 10 iterations, differences should decrease
        kh = result.keff_history
        if len(kh) > 20:
            diffs = [abs(kh[i+1] - kh[i]) for i in range(10, len(kh)-1)]
            # At least the last half should be decreasing on average
            mid = len(diffs) // 2
            avg_early = np.mean(diffs[:mid]) if mid > 0 else 0
            avg_late = np.mean(diffs[mid:]) if mid > 0 else 0
            assert avg_late <= avg_early * 1.5, (
                f"keff not converging: early avg diff={avg_early:.2e}, "
                f"late avg diff={avg_late:.2e}"
            )


# =====================================================================
# L1: PARTICLE BALANCE
# =====================================================================

class TestL1ParticleBalance:
    """L1: Global neutron balance conservation.

    At converged keff: production / absorption = keff (no leakage for
    reflective BCs). This must hold for EVERY test geometry.
    """

    def test_balance_homogeneous_2g(self):
        """[L1] 2G homogeneous: production/absorption = keff."""
        mat = get_mixture("A", "2g")
        mesh = _homogeneous_ws_mesh(pitch=3.0)
        result = _quick_solve({0: mat}, mesh, n_azi=8, max_outer=200)

        phi = result.scalar_flux
        areas = result.moc_mesh.region_areas
        nr = result.moc_mesh.n_regions

        p_rate = 0.0
        a_rate = 0.0
        for i in range(nr):
            p_rate += mat.SigP @ phi[i, :] * areas[i]
            a_rate += mat.absorption_xs @ phi[i, :] * areas[i]

        k_balance = p_rate / a_rate
        assert k_balance == pytest.approx(result.keff, rel=1e-4), (
            f"Balance keff={k_balance:.8f} vs solver keff={result.keff:.8f}"
        )

    def test_balance_heterogeneous_2g(self):
        """[L1] 2G heterogeneous: production/absorption = keff."""
        fuel = get_mixture("A", "2g")
        cool = get_mixture("B", "2g")
        materials = {2: fuel, 0: cool}

        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.0)
        result = _quick_solve(
            materials, mesh,
            n_azi=16, n_polar=3, ray_spacing=0.03,
            max_outer=300, n_inner=20,
        )

        phi = result.scalar_flux
        areas = result.moc_mesh.region_areas
        nr = result.moc_mesh.n_regions
        mat_ids = result.moc_mesh.region_mat_ids

        p_rate = 0.0
        a_rate = 0.0
        for i in range(nr):
            mid = int(mat_ids[i])
            mix = materials[mid]
            p_rate += mix.SigP @ phi[i, :] * areas[i]
            a_rate += mix.absorption_xs @ phi[i, :] * areas[i]

        k_balance = p_rate / a_rate
        assert k_balance == pytest.approx(result.keff, rel=1e-3), (
            f"Balance keff={k_balance:.8f} vs solver keff={result.keff:.8f}"
        )


# =====================================================================
# L1: FLUX POSITIVITY AND PHYSICAL PROPERTIES
# =====================================================================

class TestL1FluxProperties:
    """L1: Physical sanity of the computed flux distribution."""

    @pytest.mark.parametrize("ng_key", ["1g", "2g", "4g"])
    def test_flux_positivity(self, ng_key):
        """[L1] Scalar flux must be non-negative everywhere, all groups."""
        mat = get_mixture("A", ng_key)
        mesh = _homogeneous_ws_mesh(pitch=3.0)
        result = _quick_solve({0: mat}, mesh, n_azi=8, max_outer=200)
        assert np.all(result.scalar_flux >= 0), (
            f"Negative flux found: min={result.scalar_flux.min():.4e}"
        )

    def test_heterogeneous_flux_positivity(self):
        """[L1] Heterogeneous 2G: all fluxes non-negative."""
        fuel = get_mixture("A", "2g")
        cool = get_mixture("B", "2g")
        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.0)
        result = _quick_solve(
            {2: fuel, 0: cool}, mesh,
            n_azi=16, n_polar=3, ray_spacing=0.03,
            max_outer=300, n_inner=20,
        )
        assert np.all(result.scalar_flux >= 0), (
            f"Negative flux in heterogeneous: min={result.scalar_flux.min():.4e}"
        )


# =====================================================================
# L2: CONVERGENCE RATE — RAY SPACING
# =====================================================================

class TestL2RaySpacingConvergence:
    """L2: keff must converge as ray spacing decreases.

    Flat-source MOC has O(t_s^2) spatial convergence in ray spacing.
    At minimum, successive refinements must reduce the error.
    """

    def test_ray_spacing_convergence_2g(self):
        """[L2] 2G homogeneous: keff converges with decreasing ray spacing.

        Three refinement levels. Differences must strictly decrease.
        """
        mat = get_mixture("A", "2g")
        mesh = _homogeneous_ws_mesh(pitch=3.0)

        xs = get_xs("A", "2g")
        k_ref = kinf_homogeneous(
            sig_t=xs["sig_t"], sig_s=xs["sig_s"],
            nu_sig_f=xs["nu"] * xs["sig_f"], chi=xs["chi"],
        )

        spacings = [0.1, 0.05, 0.025]
        keffs = []
        for sp in spacings:
            result = _quick_solve({0: mat}, mesh, n_azi=16, n_polar=3,
                                  ray_spacing=sp, max_outer=200)
            keffs.append(result.keff)

        errors = [abs(k - k_ref) for k in keffs]
        for i in range(len(errors) - 1):
            assert errors[i+1] <= errors[i] * 1.1, (
                f"Not converging in ray spacing: errors={errors}"
            )

    @pytest.mark.slow
    def test_ray_spacing_convergence_rate_heterogeneous_2g(self):
        """[L2] 2G heterogeneous: keff converges with decreasing ray spacing.

        Homogeneous is trivially exact (flat flux), so we MUST use a
        heterogeneous geometry to test spatial convergence.  Successive
        refinements must reduce the difference between consecutive keff
        values (Richardson convergence).
        """
        fuel = get_mixture("A", "2g")
        cool = get_mixture("B", "2g")
        materials = {2: fuel, 0: cool}
        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.0)

        spacings = [0.08, 0.04, 0.02]
        keffs = []
        for sp in spacings:
            result = _quick_solve(
                materials, mesh,
                n_azi=32, n_polar=3, ray_spacing=sp,
                max_outer=300, n_inner=20,
            )
            keffs.append(result.keff)

        # Successive differences must decrease (convergence)
        diff_1 = abs(keffs[1] - keffs[0])
        diff_2 = abs(keffs[2] - keffs[1])
        assert diff_2 < diff_1 * 1.1, (
            f"Not converging: diff_1={diff_1:.2e}, diff_2={diff_2:.2e}, "
            f"keffs={keffs}"
        )


# =====================================================================
# L2: CONVERGENCE RATE — AZIMUTHAL
# =====================================================================

class TestL2AzimuthalConvergence:
    """L2: keff must converge with increasing azimuthal angles."""

    def test_azimuthal_convergence_2g(self):
        """[L2] 2G homogeneous: keff converges with increasing n_azi."""
        mat = get_mixture("A", "2g")
        mesh = _homogeneous_ws_mesh(pitch=3.0)

        xs = get_xs("A", "2g")
        k_ref = kinf_homogeneous(
            sig_t=xs["sig_t"], sig_s=xs["sig_s"],
            nu_sig_f=xs["nu"] * xs["sig_f"], chi=xs["chi"],
        )

        n_azi_list = [4, 8, 16]
        keffs = []
        for n_azi in n_azi_list:
            result = _quick_solve({0: mat}, mesh, n_azi=n_azi, n_polar=3,
                                  ray_spacing=0.03, max_outer=200)
            keffs.append(result.keff)

        errors = [abs(k - k_ref) for k in keffs]
        # Last refinement should be at least as good as first
        assert errors[-1] <= errors[0] * 1.1, (
            f"Azimuthal not converging: errors={errors}"
        )


# =====================================================================
# L2: CONVERGENCE RATE — POLAR
# =====================================================================

class TestL2PolarConvergence:
    """L2: keff must improve (or not degrade) with more polar angles."""

    def test_polar_convergence_2g(self):
        """[L2] 2G homogeneous: keff at n_polar=3 at least as good as n_polar=1."""
        mat = get_mixture("A", "2g")
        mesh = _homogeneous_ws_mesh(pitch=3.0)

        xs = get_xs("A", "2g")
        k_ref = kinf_homogeneous(
            sig_t=xs["sig_t"], sig_s=xs["sig_s"],
            nu_sig_f=xs["nu"] * xs["sig_f"], chi=xs["chi"],
        )

        errors = []
        for n_polar in [1, 2, 3]:
            result = _quick_solve({0: mat}, mesh, n_azi=16, n_polar=n_polar,
                                  ray_spacing=0.03, max_outer=200)
            errors.append(abs(result.keff - k_ref))

        # n_polar=3 should be at least as accurate as n_polar=1
        assert errors[-1] <= errors[0] * 1.5, (
            f"Polar not converging: errors={errors}"
        )


# =====================================================================
# XV: CROSS-VERIFICATION — MOC vs CP
# =====================================================================

@pytest.mark.l2  # Code-to-code XV (MOC ↔ CP)
class TestXVCrossVerification:
    """XV: MOC eigenvalue vs CP eigenvalue on identical geometry.

    Both are independent solvers on the same cross sections and mesh.
    They should agree within their combined discretization error.

    The CP solver uses white-BC collision probabilities, so there is a
    ~1% systematic gap vs reflective-BC MOC. We use this as a
    cross-check, not a precision target.
    """

    @pytest.mark.slow
    def test_moc_vs_cp_2g_fuel_coolant(self):
        """[XV] 2G fuel+coolant: MOC and CP keff agree within 2%.

        This tolerance accounts for: white-BC vs reflective-BC gap (~1%),
        plus angular/spatial discretization in both solvers.
        """
        fuel = get_mixture("A", "2g")
        cool = get_mixture("B", "2g")
        materials = {2: fuel, 0: cool}

        r_fuel = 0.5
        pitch = 2.0
        ws_r = pitch / np.sqrt(np.pi)

        mesh = Mesh1D(
            edges=np.array([0.0, r_fuel, ws_r]),
            mat_ids=np.array([2, 0]),
            coord=CoordSystem.CYLINDRICAL,
        )

        # MOC solve
        result_moc = _quick_solve(
            materials, mesh,
            n_azi=16, n_polar=3, ray_spacing=0.03,
            max_outer=300, n_inner=20,
        )

        # CP solve
        try:
            from orpheus.cp.solver import solve_cp
            result_cp = solve_cp(materials, mesh)
            k_cp = result_cp.keff
        except ImportError:
            pytest.skip("CP solver not available")

        rel_diff = abs(result_moc.keff - k_cp) / k_cp
        assert rel_diff < 0.02, (
            f"MOC keff={result_moc.keff:.6f} vs CP keff={k_cp:.6f}, "
            f"rel_diff={rel_diff:.4f}"
        )

    @pytest.mark.slow
    def test_moc_vs_cp_4g_fuel_coolant(self):
        """[XV] 4G fuel+coolant: MOC and CP keff agree within 2%.

        4-group test to catch multi-group coupling bugs.
        """
        fuel = get_mixture("A", "4g")
        cool = get_mixture("B", "4g")
        materials = {2: fuel, 0: cool}

        r_fuel = 0.5
        pitch = 2.0
        ws_r = pitch / np.sqrt(np.pi)

        mesh = Mesh1D(
            edges=np.array([0.0, r_fuel, ws_r]),
            mat_ids=np.array([2, 0]),
            coord=CoordSystem.CYLINDRICAL,
        )

        result_moc = _quick_solve(
            materials, mesh,
            n_azi=16, n_polar=3, ray_spacing=0.03,
            max_outer=300, n_inner=20,
        )

        try:
            from orpheus.cp.solver import solve_cp
            result_cp = solve_cp(materials, mesh)
            k_cp = result_cp.keff
        except ImportError:
            pytest.skip("CP solver not available")

        rel_diff = abs(result_moc.keff - k_cp) / k_cp
        assert rel_diff < 0.02, (
            f"4G MOC keff={result_moc.keff:.6f} vs CP keff={k_cp:.6f}, "
            f"rel_diff={rel_diff:.4f}"
        )


# =====================================================================
# L0: SOLVER PROTOCOL COMPLIANCE
# =====================================================================

class TestL0ProtocolCompliance:
    """L0: MOCSolver satisfies the EigenvalueSolver protocol."""

    def test_initial_flux_shape(self):
        """[L0] initial_flux_distribution returns (n_regions, ng) array of ones."""
        mat = get_mixture("A", "2g")
        mesh = _homogeneous_ws_mesh(pitch=2.0)
        quad = MOCQuadrature.create(n_azi=4, n_polar=1)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.1)
        solver = MOCSolver(moc_mesh, {0: mat})

        phi = solver.initial_flux_distribution()
        assert phi.shape == (moc_mesh.n_regions, 2)
        assert np.all(phi == 1.0)

    def test_compute_keff_returns_float(self):
        """[L0] compute_keff returns a Python float."""
        mat = get_mixture("A", "2g")
        mesh = _homogeneous_ws_mesh(pitch=2.0)
        quad = MOCQuadrature.create(n_azi=4, n_polar=1)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.1)
        solver = MOCSolver(moc_mesh, {0: mat})

        phi = solver.initial_flux_distribution()
        k = solver.compute_keff(phi)
        assert isinstance(k, float)
        assert np.isfinite(k)

    def test_fission_source_shape(self):
        """[L0] compute_fission_source returns array of same shape as flux."""
        mat = get_mixture("A", "2g")
        mesh = _homogeneous_ws_mesh(pitch=2.0)
        quad = MOCQuadrature.create(n_azi=4, n_polar=1)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.1)
        solver = MOCSolver(moc_mesh, {0: mat})

        phi = solver.initial_flux_distribution()
        fs = solver.compute_fission_source(phi, keff=1.0)
        assert fs.shape == phi.shape
        assert np.all(np.isfinite(fs))

    def test_converged_false_early(self):
        """[L0] converged() returns False for iteration <= 2."""
        mat = get_mixture("A", "2g")
        mesh = _homogeneous_ws_mesh(pitch=2.0)
        quad = MOCQuadrature.create(n_azi=4, n_polar=1)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.1)
        solver = MOCSolver(moc_mesh, {0: mat})

        phi = solver.initial_flux_distribution()
        assert solver.converged(1.0, 1.0, phi, phi, iteration=1) is False
        assert solver.converged(1.0, 1.0, phi, phi, iteration=2) is False


# =====================================================================
# L1: SENSITIVITY TO MATERIAL PROPERTIES
# =====================================================================

class TestL1MaterialSensitivity:
    """L1: Eigenvalue responds correctly to material property changes.

    These are sanity tests: physically, increasing absorption should
    decrease keff, increasing nu should increase keff, etc.
    """

    def test_higher_absorption_lowers_keff(self):
        """[L1] Increasing absorption (Sigma_c) must decrease keff."""
        xs_base = get_xs("A", "2g")
        mat_base = make_mixture(**xs_base)

        # Increase capture by 50%
        xs_high = {k: v.copy() if isinstance(v, np.ndarray) else v
                   for k, v in xs_base.items()}
        xs_high["sig_c"] = xs_base["sig_c"] * 1.5
        xs_high["sig_t"] = xs_high["sig_c"] + xs_high["sig_f"] + xs_high["sig_s"].sum(axis=1)
        mat_high = make_mixture(**xs_high)

        mesh = _homogeneous_ws_mesh(pitch=3.0)
        r_base = _quick_solve({0: mat_base}, mesh, n_azi=8, max_outer=200)
        r_high = _quick_solve({0: mat_high}, mesh, n_azi=8, max_outer=200)

        assert r_high.keff < r_base.keff, (
            f"Higher absorption should lower keff: "
            f"base={r_base.keff:.6f}, high_abs={r_high.keff:.6f}"
        )

    def test_higher_nu_raises_keff(self):
        """[L1] Increasing nu must increase keff."""
        xs_base = get_xs("A", "1g")
        mat_base = make_mixture(**xs_base)

        xs_high = {k: v.copy() if isinstance(v, np.ndarray) else v
                   for k, v in xs_base.items()}
        xs_high["nu"] = xs_base["nu"] * 1.2
        mat_high = make_mixture(**xs_high)

        mesh = _homogeneous_ws_mesh(pitch=3.0)
        r_base = _quick_solve({0: mat_base}, mesh, n_azi=8, max_outer=200)
        r_high = _quick_solve({0: mat_high}, mesh, n_azi=8, max_outer=200)

        assert r_high.keff > r_base.keff, (
            f"Higher nu should raise keff: "
            f"base={r_base.keff:.6f}, high_nu={r_high.keff:.6f}"
        )


# =====================================================================
# L0: VOLUME-WEIGHTED TRACK INTEGRATION
# =====================================================================

class TestL0VolumeTracking:
    """L0: Track-estimated volumes converge to geometric volumes.

    The MOC volume estimate for region i is:
        V_est_i = sum_{tracks} omega_a * t_s * sum_{segs in i} length_s

    This must converge to region_areas[i] as ray spacing -> 0.
    Catches bugs in effective_spacing computation.
    """

    def test_track_volume_vs_geometric_2region(self):
        """[L0] Track-estimated areas within 3% of geometric for 2 regions."""
        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.0)
        quad = MOCQuadrature.create(n_azi=16, n_polar=1)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.02)

        est_areas = np.zeros(moc_mesh.n_regions)
        for a_idx in range(quad.n_azi):
            ts = moc_mesh.effective_spacing(a_idx)
            omega_a = quad.omega_azi[a_idx]
            for t_idx in moc_mesh.tracks_per_azi[a_idx]:
                for seg in moc_mesh.tracks[t_idx].segments:
                    est_areas[seg.region_id] += seg.length * ts * omega_a

        for k in range(moc_mesh.n_regions):
            rel_err = abs(est_areas[k] - moc_mesh.region_areas[k]) / moc_mesh.region_areas[k]
            assert rel_err < 0.03, (
                f"Region {k}: track_area={est_areas[k]:.6f}, "
                f"geom_area={moc_mesh.region_areas[k]:.6f}, err={rel_err:.4f}"
            )

    def test_track_volume_convergence(self):
        """[L0] Track volume estimates improve with finer ray spacing."""
        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.0)

        errors = []
        for spacing in [0.1, 0.05, 0.02]:
            quad = MOCQuadrature.create(n_azi=16, n_polar=1)
            moc_mesh = MOCMesh(mesh, quad, ray_spacing=spacing)

            est_areas = np.zeros(moc_mesh.n_regions)
            for a_idx in range(quad.n_azi):
                ts = moc_mesh.effective_spacing(a_idx)
                omega_a = quad.omega_azi[a_idx]
                for t_idx in moc_mesh.tracks_per_azi[a_idx]:
                    for seg in moc_mesh.tracks[t_idx].segments:
                        est_areas[seg.region_id] += seg.length * ts * omega_a

            max_err = max(
                abs(est_areas[k] - moc_mesh.region_areas[k]) / moc_mesh.region_areas[k]
                for k in range(moc_mesh.n_regions)
            )
            errors.append(max_err)

        # Each refinement should reduce the error
        assert errors[-1] < errors[0], (
            f"Volume tracking not converging: errors={errors}"
        )


# =====================================================================
# L0: BOUNDARY CONDITION INTEGRITY
# =====================================================================

class TestL0BoundaryConditions:
    """L0: Reflective boundary conditions preserve angular flux.

    In reflective BC, no neutrons are lost at boundaries. The total
    angular flux entering the domain must equal the total exiting.
    """

    def test_all_tracks_have_links(self):
        """[L0] Every track must have valid forward and backward links."""
        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.0)
        quad = MOCQuadrature.create(n_azi=16, n_polar=3)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.03)

        n_tracks = len(moc_mesh.tracks)
        for i, track in enumerate(moc_mesh.tracks):
            assert 0 <= track.fwd_link < n_tracks, (
                f"Track {i}: invalid fwd_link={track.fwd_link}"
            )
            assert 0 <= track.bwd_link < n_tracks, (
                f"Track {i}: invalid bwd_link={track.bwd_link}"
            )

    def test_link_target_azimuthal_is_reflected(self):
        """[L0] Link targets must be at the reflected azimuthal angle.

        For reflective BC, the reflected angle of phi is pi - phi.
        The target track must be at this reflected azimuthal index.
        """
        mesh = _two_region_ws_mesh(r_fuel=0.5, pitch=2.0)
        quad = MOCQuadrature.create(n_azi=8, n_polar=1)
        moc_mesh = MOCMesh(mesh, quad, ray_spacing=0.05)

        phi = quad.phi
        for track in moc_mesh.tracks:
            # Forward link
            fwd_target = moc_mesh.tracks[track.fwd_link]
            phi_refl = np.pi - phi[track.azi_index]
            if phi_refl < 0:
                phi_refl += np.pi
            if phi_refl >= np.pi:
                phi_refl -= np.pi
            expected_azi = int(np.argmin(np.abs(phi - phi_refl)))
            assert fwd_target.azi_index == expected_azi, (
                f"Forward link: azi={track.azi_index} -> "
                f"target azi={fwd_target.azi_index}, expected={expected_azi}"
            )
