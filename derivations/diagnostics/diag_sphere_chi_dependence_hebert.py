"""Test: how does the Hébert closure error depend on the chi spectrum?

Triggered by user observation 2026-04-25: the 2G/2R "<0.05 % near-
exactness" claim looked suspicious given that the geometry and
materials are identical to 1G/2R (which has +10 % overshoot). User's
intuition: there should be parity between 1G/2R and 2G/2R if the
geometry is the same.

This script tests the hypothesis: the difference between 1G/2R and
2G/2R is the source distribution (driven by the chi spectrum), not
the geometry or 2G structure inherently.

For the shipped chi = [1, 0] 2G XS, fission emits into the FAST
group where σ_t ≈ [0.5, 0.6] (nearly uniform across fuel and
moderator). The fast flux is therefore nearly flat, the down-scatter
source for thermal is nearly flat, and the Mark uniformity assumption
holds exactly → Hébert is essentially exact.

For thermal-emission chi = [0, 1], fission emits into the THERMAL
group where σ_t = [1, 2] (heterogeneous). Source localises in fuel
and stays heterogeneous → Mark uniformity assumption breaks → Hébert
overshoots, like 1G/2R.

Result confirms the hypothesis: the 2G/2R near-exactness is coincident
with the chi spectrum, not a structural property of 2G.
"""

from __future__ import annotations

import numpy as np

from orpheus.derivations._xs_library import LAYOUTS, get_xs
from orpheus.derivations import cp_sphere
from orpheus.derivations.peierls_geometry import SPHERE_1D, solve_peierls_mg
from orpheus.derivations.cp_sphere import _sphere_cp_matrix
from orpheus.derivations._eigenvalue import kinf_from_cp


_QUAD = dict(n_panels_per_region=2, p_order=3,
             n_angular=24, n_rho=24, n_surf_quad=24, dps=15)


def _solve(ng_key, n_regions, *, chi_override=None):
    layout = LAYOUTS[n_regions]
    xs_list = [get_xs(r, ng_key) for r in layout]
    sig_t = np.vstack([xs["sig_t"] for xs in xs_list])
    sig_s = np.stack([xs["sig_s"] for xs in xs_list], axis=0)
    nu_sig_f = np.vstack([xs["nu"] * xs["sig_f"] for xs in xs_list])
    chi = np.vstack([xs["chi"] for xs in xs_list])
    if chi_override is not None:
        chi = np.array([chi_override] * n_regions)
    radii = np.array(cp_sphere._RADII[n_regions])
    sol = solve_peierls_mg(
        SPHERE_1D, radii=radii, sig_t=sig_t, sig_s=sig_s,
        nu_sig_f=nu_sig_f, chi=chi,
        boundary="white_hebert", n_bc_modes=1,
        **_QUAD,
    )
    return sol.k_eff


def _kinf(ng_key, n_regions, *, chi_override=None):
    layout = LAYOUTS[n_regions]
    xs_list = [get_xs(r, ng_key) for r in layout]
    sig_t = np.vstack([xs["sig_t"] for xs in xs_list])
    chi_mats = [xs["chi"] for xs in xs_list]
    if chi_override is not None:
        chi_mats = [np.asarray(chi_override) for _ in xs_list]
    radii = np.array(cp_sphere._RADII[n_regions])
    r_inner = np.zeros(n_regions)
    r_inner[1:] = radii[:-1]
    volumes = (4 / 3) * np.pi * (radii**3 - r_inner**3)
    P_inf = _sphere_cp_matrix(sig_t, radii, volumes, radii[-1])
    return kinf_from_cp(
        P_inf_g=P_inf, sig_t_all=sig_t, V_arr=volumes,
        sig_s_mats=[xs["sig_s"] for xs in xs_list],
        nu_sig_f_mats=[xs["nu"] * xs["sig_f"] for xs in xs_list],
        chi_mats=chi_mats,
    )


def main():
    print("=" * 78)
    print("chi-dependence of Hébert closure error on sphere 2R fuel-A/mod-B")
    print("=" * 78)
    print()

    cases = [
        ("1G/2R (single group)", "1g", 2, None),
        ("2G/2R DEFAULT (chi=[1,0])", "2g", 2, None),
        ("2G/2R chi=[0.75, 0.25]", "2g", 2, [0.75, 0.25]),
        ("2G/2R chi=[0.5, 0.5]", "2g", 2, [0.5, 0.5]),
        ("2G/2R chi=[0.25, 0.75]", "2g", 2, [0.25, 0.75]),
        ("2G/2R chi=[0, 1] (thermal)", "2g", 2, [0.0, 1.0]),
    ]

    print(f"  {'Configuration':<35} {'cp k_inf':>10} {'Hébert k_eff':>14} "
          f"{'err':>8}")
    print("  " + "-" * 75)
    for label, ng_key, n_regions, chi in cases:
        k_eff = _solve(ng_key, n_regions, chi_override=chi)
        kinf = _kinf(ng_key, n_regions, chi_override=chi)
        err = (k_eff - kinf) / kinf * 100
        print(f"  {label:<35} {kinf:>10.6f} {k_eff:>14.6f} {err:>+7.2f}%")

    print()
    print("  Pattern: more spatially-localised source → larger Mark uniformity")
    print("  overshoot. The 2G/2R DEFAULT 'good result' is coincident with the")
    print("  chi=[1,0] spectrum that routes emissions into a near-uniform-σ_t")
    print("  group, NOT a structural property of 2G that resolves the Mark limit.")


if __name__ == "__main__":
    main()
