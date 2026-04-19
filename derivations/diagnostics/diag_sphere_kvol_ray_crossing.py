"""Diagnostic: sphere Peierls K-matrix, ray-crosses-panel integration bug.

Created by numerics-investigator on 2026-04-18.

Hypothesis under test
---------------------
``peierls_geometry.build_volume_kernel`` uses fixed-order Gauss-Legendre
over (omega, rho) for the double integral in the unified polar form.
Along a ray of given cos(omega), the source position
r'(rho) = sqrt(r_i^2 + 2 r_i rho cos(omega) + rho^2) traces out a
non-linear arc in r' that crosses one or more panel boundaries of the
piecewise Lagrange basis. At each crossing L_j(r'(rho)) has a derivative
discontinuity (a kink) — fixed GL cannot integrate through these kinks
without subdividing.

The mpmath-adaptive reference ``curvilinear_K_vol_element`` DOES
subdivide (see ``rho_crossings_for_ray``). A second, mathematically
independent reference based on the shell-average point-kernel identity

    <exp(-Sigma_t d)/(4 pi d^2)>_shell =
        1 / (8 pi r_i r') * [E_1(Sigma_t|r_i-r'|) - E_1(Sigma_t(r_i+r'))]

reduces the 3-D volume integral to a 1-D integral in r' and produces
element-wise agreement with the polar reference to < 1e-4 on typical
entries (limited by adaptive tolerance at the panel kinks). Both
references disagree with production by ~0.5 to ~5%, establishing that
production — not either reference — is wrong.

Problem: R=1, Sigma_t=1, 2 panels, p_order=3 (6 nodes). K[3,3]
production = 0.2033, polar-ref = 0.2058, shell-ref = 0.2056. Rel diff
prod vs shell ~1.1%.

If this diagnostic catches a real bug, promote to
``tests/derivations/test_peierls_sphere.py`` (or ``test_peierls_reference.py``).
"""
from __future__ import annotations

import mpmath
import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, build_volume_kernel, lagrange_basis_on_panels,
)
from orpheus.derivations.peierls_reference import curvilinear_K_vol_element
from orpheus.derivations.peierls_slab import (
    _gl_nodes_weights, _map_to_interval,
)


def _build_sphere_K(R, sig_t_val, n_panels, p_order, *,
                    n_angular=32, n_rho=32, dps=30):
    """Build nodes and production K for a homogeneous sphere of radius R."""
    with mpmath.workdps(dps):
        gl_ref, gl_wt = _gl_nodes_weights(p_order, dps)
        x_all, w_all, pbs = [], [], []
        pw = mpmath.mpf(R) / n_panels
        for pidx in range(n_panels):
            pa = mpmath.mpf(pidx) * pw
            pb = pa + pw
            i_start = len(x_all)
            xs, ws = _map_to_interval(gl_ref, gl_wt, pa, pb)
            x_all.extend(xs)
            w_all.extend(ws)
            pbs.append((pa, pb, i_start, len(x_all)))

    x_nodes_f = np.array([float(x) for x in x_all])
    pbs_f = [(float(pa), float(pb), i0, i1) for pa, pb, i0, i1 in pbs]
    radii = np.array([R])
    sig_t_arr = np.array([sig_t_val])

    K_prod = build_volume_kernel(
        SPHERE_1D, x_nodes_f, pbs_f, radii, sig_t_arr,
        n_angular=n_angular, n_rho=n_rho, dps=dps,
    )
    return K_prod, x_nodes_f, pbs_f, radii, sig_t_arr


def _shell_avg_K(i, j, x_nodes, pbs, R, sig_t, *, dps=30):
    """INDEPENDENT reference via shell-averaged point kernel.

    For a volumetric source on a shell of radius r' seen by an observer
    at radius r_i, the angular-averaged flux kernel (isotropic emission
    of unit strength per unit volume in the shell) is

        avg(exp(-Sigma_t d)/(4 pi d^2))
            = 1/(8 pi r_i r') * [E_1(Sigma_t|r_i-r'|) - E_1(Sigma_t(r_i+r'))]

    Integrating over r' weighted by the 4 pi r'^2 volume element and
    multiplying by the Lagrange basis L_j gives

        K[i, j] = Sigma_t(r_i) * (1 / (2 r_i)) * int_0^R
                    r' * [E_1(...) - E_1(...)] * L_j(r') dr'

    Subdividing the 1-D integral at r_i and at panel boundaries handles
    the log singularity at r' = r_i (E_1 diverges) and the kinks of L_j.
    """
    r_i = mpmath.mpf(x_nodes[i])
    s = mpmath.mpf(sig_t)

    def integrand(rp):
        rp = mpmath.mpf(rp)
        if rp == 0 or rp == r_i:
            return mpmath.mpf(0)
        tau1 = s * abs(r_i - rp)
        tau2 = s * (r_i + rp)
        term = mpmath.expint(1, tau1) - mpmath.expint(1, tau2)
        L_vals = lagrange_basis_on_panels(x_nodes, pbs, float(rp))
        return rp * term * mpmath.mpf(float(L_vals[j]))

    # Breakpoints: 0, panel boundaries, r_i, R
    breaks_set = {mpmath.mpf(0), mpmath.mpf(R), r_i}
    for pa, pb, _, _ in pbs:
        breaks_set.add(mpmath.mpf(pa))
        breaks_set.add(mpmath.mpf(pb))
    breaks = sorted(breaks_set)

    with mpmath.workdps(dps):
        val = mpmath.quad(integrand, breaks)
    return s / (2 * r_i) * val


@pytest.mark.l1
def test_sphere_K_production_disagrees_with_two_independent_references():
    """Shell-average form and polar adaptive form MUST agree with each
    other (they are two distinct but equivalent integrals of the same
    operator). Production must agree with both — currently it does not.

    This is the three-way arbitration: if references disagree with each
    other, one of them is buggy; if they agree and production differs,
    production is buggy. Here production differs.
    """
    R, sig_t = 1.0, 1.0
    n_panels, p_order, dps = 2, 3, 30
    K_prod, x_nodes, pbs, radii, sig_t_arr = _build_sphere_K(
        R, sig_t, n_panels, p_order,
        n_angular=32, n_rho=32, dps=dps,
    )

    # Central entries that avoid ray-crossing-through-origin complications
    entries = [(0, 0), (0, 5), (5, 5), (3, 3)]
    for i, j in entries:
        prod = K_prod[i, j]
        ref_polar = float(curvilinear_K_vol_element(
            SPHERE_1D, i, j, x_nodes, pbs, radii, sig_t_arr, dps=dps,
        ))
        ref_shell = float(_shell_avg_K(i, j, x_nodes, pbs, R, sig_t, dps=dps))

        # References must agree
        ref_consistency = abs(ref_polar - ref_shell) / abs(ref_shell)
        assert ref_consistency < 1e-3, (
            f"Two independent references disagree for K[{i},{j}]: "
            f"polar={ref_polar:.6e} shell={ref_shell:.6e} "
            f"rel_diff={ref_consistency:.3e}. Bug is in one of the "
            f"references themselves."
        )
        # Production must agree with (consistent) references
        prod_err = abs(prod - ref_shell) / abs(ref_shell)
        assert prod_err < 1e-6, (
            f"Production K[{i},{j}] = {prod:.6e} disagrees with "
            f"consistent reference {ref_shell:.6e} by rel_diff={prod_err:.3e}.\n"
            f"Root cause: build_volume_kernel uses fixed GL on (omega, rho) "
            f"with no subdivision at rho values where r'(rho) crosses the "
            f"Lagrange basis panel boundaries (see peierls_geometry.py "
            f"build_volume_kernel, lines ~491-516)."
        )


@pytest.mark.l1
def test_sphere_K_converges_under_quadrature_refinement():
    """Refining n_angular and n_rho should drive K to the reference
    value. If it plateaus, the fixed-GL scheme is missing something
    the reference has — which would be the ray-crossing subdivision.
    """
    R, sig_t = 1.0, 1.0
    n_panels, p_order, dps = 2, 3, 25
    _, x_nodes, pbs, radii, sig_t_arr = _build_sphere_K(
        R, sig_t, n_panels, p_order, n_angular=8, n_rho=8, dps=dps,
    )
    # Reference via shell-average at high precision
    ref = float(_shell_avg_K(3, 3, x_nodes, pbs, R, sig_t, dps=dps))

    errors = {}
    for n_q in (8, 16, 32, 64):
        K, *_ = _build_sphere_K(
            R, sig_t, n_panels, p_order,
            n_angular=n_q, n_rho=n_q, dps=dps,
        )
        err = abs(K[3, 3] - ref) / abs(ref)
        errors[n_q] = err

    # If the bug is present, doubling (n_ang, n_rho) drops error sub-exponentially
    # because the integrand has derivative discontinuities along every ray
    # crossing a panel boundary. The error should still *decrease* with refinement
    # but plateau well above any reasonable tolerance.
    # Typical signature: each doubling cuts the error by 1.5x-3x, not 10x+.
    assert errors[64] > 1e-5, (
        f"Production K[3,3] converged to {errors[64]:.3e} at n_q=64 "
        f"— below the bug signature threshold. Flip this assertion (<) "
        f"and promote to the permanent suite."
    )


@pytest.mark.l1
def test_sphere_K_row_sum_conservation():
    """K @ 1 (pure absorber, uniform unit volumetric source) row-sum test.

    For a homogeneous sphere of radius R with volumetric source S=1 and
    pure absorption, the flux at r_i is

        phi(r_i) = 1/(2 Sigma_t) * [2 - E_2(Sigma_t(R-r_i))
                                        - E_2(Sigma_t(R+r_i))]
                    (see Case & Zweifel, or derive from shell-average)

    Then Sigma_t * phi(r_i) = sum_j K[i, j]  (row-sum identity).

    This is an analytical, independent verification of the operator.
    """
    R, sig_t = 1.0, 1.0

    def phi_exact(r_i):
        """Exact flux for uniform volumetric source S=1 on pure-absorber
        sphere of radius R at observer radius r_i < R.

        Derivation: phi(r_i) = (1/(2 r_i)) int_0^R r' [E_1(s|r_i-r'|)
                                             - E_1(s(r_i+r'))] dr'
        Use d/du E_2(u) = -E_1(u), so int r' E_1(a|r_i-r'|) dr' admits
        explicit anti-derivative.
        """
        r_i_mp = mpmath.mpf(r_i)
        s = mpmath.mpf(sig_t)
        R_mp = mpmath.mpf(R)
        # Use mpmath.quad directly — it's fast for smooth 1-D integrand
        def f(rp):
            rp = mpmath.mpf(rp)
            if rp == 0 or rp == r_i_mp:
                return mpmath.mpf(0)
            term = (mpmath.expint(1, s * abs(r_i_mp - rp))
                    - mpmath.expint(1, s * (r_i_mp + rp)))
            return rp * term
        with mpmath.workdps(30):
            I = mpmath.quad(f, [mpmath.mpf(0), r_i_mp, R_mp])
            return s * I / (2 * r_i_mp)

    n_panels, p_order = 2, 3
    K, x_nodes, _, _, _ = _build_sphere_K(
        R, sig_t, n_panels, p_order,
        n_angular=64, n_rho=64, dps=25,
    )
    N = len(x_nodes)
    max_err = 0.0
    for i in range(N):
        row_sum = float(sum(K[i, j] for j in range(N)))
        ref = float(phi_exact(x_nodes[i])) * sig_t  # Sigma_t * phi
        if abs(ref) > 1e-30:
            rel = abs(row_sum - ref) / abs(ref)
            max_err = max(max_err, rel)

    # SURPRISING DISCOVERY (2026-04-18): the sphere row-sum identity
    # K @ 1 passes to ~1e-15 despite individual K[i,j] being wrong by 0.5-5%.
    # Explanation: sum_j L_j(r') = 1 identically (partition-of-unity), so
    # the summed integrand has NO derivative discontinuity at panel crossings
    # even though each L_j individually does. The ray-crossing bug
    # CANCELS in the row-sum. This is why existing row-sum tests
    # (e.g. test_peierls_sphere_white_bc) did not catch this — and why
    # the bug requires element-wise K[i,j] checks to be visible.
    # K.P: if future work restores row-sum error > 1e-8, that is a
    # DIFFERENT bug (a real conservation failure).
    assert max_err < 1e-10, (
        f"Sphere row-sum error = {max_err:.3e} > 1e-10. This is a new "
        f"conservation-breaking bug (the ray-crossing bug cancels in "
        f"row-sums via partition-of-unity). Investigate immediately."
    )


if __name__ == "__main__":
    R, sig_t = 1.0, 1.0
    n_panels, p_order = 2, 3
    print("Sphere Peierls K-matrix bug isolation")
    print("=" * 70)

    K_prod, x_nodes, pbs, radii, sig_t_arr = _build_sphere_K(
        R, sig_t, n_panels, p_order, n_angular=32, n_rho=32, dps=30,
    )
    print(f"nodes: {x_nodes}")
    print()

    for i, j in [(0, 0), (0, 5), (5, 5), (3, 3), (3, 0), (5, 0)]:
        prod = K_prod[i, j]
        ref_polar = float(curvilinear_K_vol_element(
            SPHERE_1D, i, j, x_nodes, pbs, radii, sig_t_arr, dps=30,
        ))
        ref_shell = float(_shell_avg_K(i, j, x_nodes, pbs, R, sig_t, dps=30))
        pv = abs(ref_polar - ref_shell) / abs(ref_shell) if abs(ref_shell) > 1e-30 else 0
        ps = abs(prod - ref_shell) / abs(ref_shell) if abs(ref_shell) > 1e-30 else 0
        print(
            f"K[{i},{j}]: prod={prod:.5e}  polar={ref_polar:.5e}  "
            f"shell={ref_shell:.5e}  polar_vs_shell={pv:.2e}  "
            f"prod_vs_shell={ps:.2e}"
        )

    # Quadrature scan
    print()
    print("Quadrature refinement scan (K[3,3] vs shell-avg reference):")
    ref = float(_shell_avg_K(3, 3, x_nodes, pbs, R, sig_t, dps=30))
    for n_q in (8, 16, 32, 64, 128):
        K, *_ = _build_sphere_K(
            R, sig_t, n_panels, p_order,
            n_angular=n_q, n_rho=n_q, dps=25,
        )
        err = abs(K[3, 3] - ref) / abs(ref)
        print(f"  n_angular=n_rho={n_q:3d}  rel_err = {err:.3e}")
