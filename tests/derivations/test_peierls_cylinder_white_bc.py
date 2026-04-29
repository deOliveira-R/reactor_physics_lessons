"""L1 tests for the rank-1 white-BC closure of the Peierls cylinder solver.

C7 of the Phase-4.2 campaign. Exercises the ``boundary='white'``
path of :func:`solve_peierls_cylinder_1g` and the two helpers
:func:`~orpheus.derivations.peierls_geometry.compute_P_esc` and
:func:`~orpheus.derivations.peierls_geometry.compute_G_bc`.

.. important::

   The rank-1 white-BC closure is the CP-flat-source-equivalent
   (Mark / isotropic re-entry) closure at the pointwise-Nyström
   level. It is an **approximation** that degrades as the cell
   becomes thinner — the Wigner-Seitz exact identity
   ``k_eff(white) = k_inf`` holds only asymptotically. See the
   caveat block in
   :func:`~orpheus.derivations.peierls_geometry.build_white_bc_correction`
   for the error table. Tests here gate on the **thick-limit**
   behaviour, where the closure is quantitatively accurate.

Structural checks validated here:

1. ``P_esc`` behaves correctly: decreasing with :math:`r_i \\to R`
   (neutrons born near the boundary escape with high probability)
   and bounded by :math:`[0, 1]`.
2. ``G_bc`` is positive and non-zero for every interior node.
3. White-BC :math:`k_{\\rm eff}` exceeds vacuum-BC :math:`k_{\\rm eff}`
   at every finite :math:`R` (less leakage ⇒ higher eigenvalue).
4. White-BC :math:`k_{\\rm eff}` converges to
   :math:`k_\\infty = \\nu\\Sigma_f / \\Sigma_a` in the thick-cylinder
   limit, at a rate consistent with the rank-1 closure error.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_cylinder import (
    GEOMETRY,
    solve_peierls_cylinder_1g,
)
from orpheus.derivations.peierls_geometry import (
    compute_G_bc,
    compute_P_esc,
)


_SIG_T = np.array([1.0])
_SIG_S = np.array([0.5])
_NU_SIG_F = np.array([0.75])
_K_INF = _NU_SIG_F[0] / (_SIG_T[0] - _SIG_S[0])  # = 1.5


@pytest.mark.l0
@pytest.mark.verifies("peierls-equation")
class TestPescProperties:
    """``P_esc(r_i)`` — uncollided escape probability from interior."""

    def test_bounded_in_unit_interval(self):
        r_nodes = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        P_esc = compute_P_esc(
            GEOMETRY, r_nodes, np.array([1.0]), np.array([1.0]),
            n_angular=16, dps=20,
        )
        assert np.all(P_esc >= 0.0)
        assert np.all(P_esc <= 1.0)

    def test_increases_toward_boundary(self):
        """For uniform Σ_t, points closer to the surface escape more
        easily (shorter mean chord to surface)."""
        r_nodes = np.linspace(0.05, 0.95, 7)
        P_esc = compute_P_esc(
            GEOMETRY, r_nodes, np.array([1.0]), np.array([1.0]),
            n_angular=16, dps=20,
        )
        diffs = np.diff(P_esc)
        assert np.all(diffs > -1e-6), (
            f"P_esc not monotone in r: diffs = {diffs}"
        )


@pytest.mark.l0
@pytest.mark.verifies("peierls-equation")
class TestGbcProperties:
    """``G_bc(r_i)`` — flux at :math:`r_i` from unit uniform surface current."""

    def test_positive_everywhere(self):
        r_nodes = np.array([0.1, 0.5, 0.95])
        G_bc = compute_G_bc(
            GEOMETRY, r_nodes, np.array([1.0]), np.array([1.0]),
            n_surf_quad=16, dps=20,
        )
        assert np.all(G_bc > 0.0)

    def test_increases_toward_boundary(self):
        """Uniform surface source contributes more to points nearer
        the boundary (unattenuated path length is shorter)."""
        r_nodes = np.linspace(0.05, 0.95, 7)
        G_bc = compute_G_bc(
            GEOMETRY, r_nodes, np.array([1.0]), np.array([1.0]),
            n_surf_quad=16, dps=20,
        )
        diffs = np.diff(G_bc)
        assert np.all(diffs > -1e-6), (
            f"G_bc not monotone in r: diffs = {diffs}"
        )


@pytest.mark.l1
@pytest.mark.verifies("peierls-equation", "one-group-kinf")
class TestWhiteBCEigenvalue:
    """White-BC eigenvalue approaches k_inf in the thick limit,
    and exceeds vacuum-BC everywhere."""

    def test_k_eff_thick_approaches_kinf(self):
        """At R = 10 MFP, the rank-1 white-BC closure gives k_eff
        within 2 % of k_inf = 1.5."""
        sol = solve_peierls_cylinder_1g(
            radii=np.array([10.0]),
            sig_t=_SIG_T, sig_s=_SIG_S, nu_sig_f=_NU_SIG_F,
            boundary="white",
            n_panels_per_region=3, p_order=5,
            n_beta=20, n_rho=20, n_phi=20, dps=20,
        )
        err = abs(sol.k_eff - _K_INF) / _K_INF
        assert err < 2e-2, (
            f"Thick-limit k_eff(white, R=10) = {sol.k_eff:.6f}, "
            f"err vs k_inf = {err:.3e} (>2%)"
        )

    def test_white_exceeds_vacuum(self):
        """White BC suppresses leakage ⇒ k_eff(white) > k_eff(vacuum)
        at every finite R."""
        for R in (2.0, 5.0, 10.0):
            sol_vac = solve_peierls_cylinder_1g(
                radii=np.array([R]),
                sig_t=_SIG_T, sig_s=_SIG_S, nu_sig_f=_NU_SIG_F,
                boundary="vacuum",
                n_panels_per_region=3, p_order=5,
                n_beta=18, n_rho=18, dps=20,
            )
            sol_wht = solve_peierls_cylinder_1g(
                radii=np.array([R]),
                sig_t=_SIG_T, sig_s=_SIG_S, nu_sig_f=_NU_SIG_F,
                boundary="white",
                n_panels_per_region=3, p_order=5,
                n_beta=18, n_rho=18, n_phi=18, dps=20,
            )
            assert sol_wht.k_eff > sol_vac.k_eff, (
                f"At R={R}: white k_eff = {sol_wht.k_eff:.6f} should "
                f"exceed vacuum k_eff = {sol_vac.k_eff:.6f}"
            )

    @pytest.mark.slow
    def test_k_eff_monotone_approach_to_kinf(self):
        """k_eff(white) → k_inf monotonically from below as R increases."""
        keffs = []
        for R in (3.0, 5.0, 10.0, 20.0):
            sol = solve_peierls_cylinder_1g(
                radii=np.array([R]),
                sig_t=_SIG_T, sig_s=_SIG_S, nu_sig_f=_NU_SIG_F,
                boundary="white",
                n_panels_per_region=3, p_order=5,
                n_beta=16, n_rho=16, n_phi=16, dps=20,
            )
            keffs.append(sol.k_eff)
        # All below k_inf
        assert all(k < _K_INF for k in keffs), keffs
        # Monotone increasing
        assert all(keffs[i+1] >= keffs[i] - 1e-5 for i in range(len(keffs)-1)), keffs
        # Last point within 2 % of k_inf
        assert abs(keffs[-1] - _K_INF) / _K_INF < 2e-2
