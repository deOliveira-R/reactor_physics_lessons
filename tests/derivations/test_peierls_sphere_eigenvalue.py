"""L1 eigenvalue tests for the spherical Peierls solver — vacuum BC.

Mirror of ``test_peierls_cylinder_eigenvalue.py``. Vacuum-BC
:math:`k_{\\rm eff}(R)` is a clean per-size tie-point: no closure
approximation enters, so :math:`k_{\\rm eff}(R \\to \\infty) \\to
k_\\infty` and :math:`k_{\\rm eff}(R_c) = 1` at the critical radius.

We exercise:

1. **Thick limit**: at :math:`R = 30` MFP, :math:`k_{\\rm eff}` is
   within :math:`\\sim 1 \\%` of :math:`k_\\infty = \\nu\\Sigma_f /
   \\Sigma_a` (pure leakage deficit).
2. **Monotone growth**: :math:`k_{\\rm eff}(R)` increases with
   :math:`R` (geometric buckling :math:`(\\pi/R)^2` shrinks).
3. **Quadrature convergence**: at fixed :math:`R`, refining
   :math:`(n_\\theta, n_\\rho)` makes :math:`k_{\\rm eff}` converge.

A sphere critical-radius tie-point is documented for future
enhancement; Case & Zweifel 1967 has tabulated values but we do not
transcribe digits manually (Cardinal Rule L4 — programmatic sources
only). Ship test-list placeholders instead."""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_sphere import (
    solve_peierls_sphere_1g,
)


# ═══════════════════════════════════════════════════════════════════════
# Vacuum BC thick limit
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
@pytest.mark.verifies("peierls-unified")
class TestVacuumBCThickLimit:
    """As R → ∞, the bare-sphere k_eff approaches k_inf."""

    @staticmethod
    def _k_inf(sig_t, sig_s, nu_sig_f):
        return nu_sig_f / (sig_t - sig_s)

    def test_k_eff_approaches_k_inf_at_thirty_mfp(self):
        """k_eff(R = 30 MFP) is within 1 % of k_inf = 1.5."""
        sig_t, sig_s, nu_sig_f = 1.0, 0.5, 0.75
        k_inf = self._k_inf(sig_t, sig_s, nu_sig_f)
        sol = solve_peierls_sphere_1g(
            np.array([30.0]),
            np.array([sig_t]), np.array([sig_s]), np.array([nu_sig_f]),
            boundary="vacuum",
            n_panels_per_region=3, p_order=5,
            n_theta=24, n_rho=24, dps=25,
        )
        assert abs(sol.k_eff - k_inf) / k_inf < 1e-2, (
            f"k_eff(R=30 MFP) = {sol.k_eff:.4f} is more than 1 % from "
            f"k_inf = {k_inf:.4f}"
        )

    def test_k_eff_monotone_in_R(self):
        """k_eff(R) grows with R (smaller geometric buckling)."""
        sig_t, sig_s, nu_sig_f = 1.0, 0.5, 0.75
        R_values = [1.5, 3.0, 6.0, 12.0, 24.0]
        k_values = []
        for R in R_values:
            sol = solve_peierls_sphere_1g(
                np.array([R]),
                np.array([sig_t]), np.array([sig_s]),
                np.array([nu_sig_f]),
                boundary="vacuum",
                n_panels_per_region=2, p_order=5,
                n_theta=20, n_rho=20, dps=22,
            )
            k_values.append(sol.k_eff)
        diffs = np.diff(k_values)
        assert np.all(diffs > -1e-4), (
            f"k_eff(R) not monotone: values={k_values}, diffs={diffs}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Quadrature convergence
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
@pytest.mark.verifies("peierls-unified")
class TestQuadratureConvergence:
    """Refining the (θ, ρ) quadrature orders drives k_eff toward a
    fixed value (spectral-like convergence of the polar integrand)."""

    def test_k_eff_converges_under_refinement(self):
        sig_t, sig_s, nu_sig_f = 1.0, 0.5, 0.75
        R = 4.0
        radii = np.array([R])
        k_values = []
        for n_q in (12, 20, 28):
            sol = solve_peierls_sphere_1g(
                radii,
                np.array([sig_t]), np.array([sig_s]),
                np.array([nu_sig_f]),
                boundary="vacuum",
                n_panels_per_region=2, p_order=5,
                n_theta=n_q, n_rho=n_q, dps=22,
            )
            k_values.append(sol.k_eff)
        # The change between the last two refinements is smaller than
        # between the first two — convergence proxy.
        assert abs(k_values[2] - k_values[1]) < abs(k_values[1] - k_values[0])
        # And the converged value is finite, positive, below k_inf.
        assert 0.0 < k_values[-1] < 1.5


# ═══════════════════════════════════════════════════════════════════════
# White BC thick-limit sanity (Phase-4.2-style quick scan)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
@pytest.mark.verifies("peierls-unified")
class TestWhiteBCThickLimit:
    """The rank-1 white-BC closure is a flat-source result re-applied
    at the pointwise level; its error shrinks in the thick limit.
    At R = 10 MFP, k_eff matches k_inf to ~0.5 %."""

    def test_k_eff_matches_k_inf_at_ten_mfp(self):
        sig_t, sig_s, nu_sig_f = 1.0, 0.5, 0.75
        k_inf = nu_sig_f / (sig_t - sig_s)
        sol = solve_peierls_sphere_1g(
            np.array([10.0]),
            np.array([sig_t]), np.array([sig_s]), np.array([nu_sig_f]),
            boundary="white",
            n_panels_per_region=2, p_order=5,
            n_theta=20, n_rho=20, n_phi=32, dps=25,
        )
        err = abs(sol.k_eff - k_inf) / k_inf
        assert err < 1e-2, (
            f"R=10 MFP white-BC k_eff = {sol.k_eff:.4f} off k_inf = "
            f"{k_inf:.4f} by {err*100:.2f} % (rank-1 closure expected "
            f"~0.5 % error — cf. cylinder's ~1 % at the same size)"
        )
