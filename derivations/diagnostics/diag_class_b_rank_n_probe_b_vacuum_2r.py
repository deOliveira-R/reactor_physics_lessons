"""Probe B: drop the closure — check K_vol alone in 2R heterogeneous sphere.

Created by numerics-investigator on 2026-04-24.

Question: is the bug in K_vol (volume kernel) for 2R heterogeneous, or
purely in K_bc (white-BC closure) for 2R heterogeneous?

Test: replace the white-BC closure with a vacuum BC (K_bc = 0) on the
sphere 1G/2R "Z" case (radii=[0.5,1.0], σ_t=[1.0, 2.0], A's XS for both
regions). Run a quadrature convergence sweep on the result.

For vacuum BC the eigenvalue is well-defined and is set by the analytic
sphere transport eigenvalue (numerical reference: vacuum-sphere k_eff).
What we want to verify: does the vacuum-BC k_eff converge cleanly under
refinement (BASE → RICH → ULTRA)? If yes → K_vol works in 2R het. If
no → K_vol itself has an MR het bug, and the closure inherit that.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations._xs_library import get_xs
from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    solve_peierls_mg,
)


def _solve_sphere_vacuum(radii, sig_t_per_region, ng_key="1g", **quad):
    n_regions = len(radii)
    ng = len(sig_t_per_region[0])
    xs_A = get_xs("A", ng_key)
    sig_s = np.stack([xs_A["sig_s"]] * n_regions, axis=0)
    nu_sig_f = np.vstack([xs_A["nu"] * xs_A["sig_f"]] * n_regions)
    chi = np.vstack([xs_A["chi"]] * n_regions)
    sig_t = np.array(sig_t_per_region, dtype=float).reshape(n_regions, ng)

    sol = solve_peierls_mg(
        SPHERE_1D, radii=np.array(radii), sig_t=sig_t,
        sig_s=sig_s, nu_sig_f=nu_sig_f, chi=chi,
        boundary="vacuum",
        **quad,
    )
    return sol.k_eff


def test_probe_b_vacuum_homogeneous_routing_invariance():
    """Pure K_vol with vacuum BC: 1R homog vs 2R-routed homog must agree."""
    quad = dict(n_panels_per_region=2, p_order=3, n_angular=24, n_rho=24,
                n_surf_quad=24, dps=15)
    k_X = _solve_sphere_vacuum([1.0], [[1.0]], **quad)
    k_Y = _solve_sphere_vacuum([0.5, 1.0], [[1.0], [1.0]], **quad)
    rel_diff = abs(k_X - k_Y) / max(abs(k_X), 1e-30)
    print(
        f"\n  Vacuum BC: k_X(1R, σ=1) = {k_X:.10f}, "
        f"k_Y(2R-homog, σ=1) = {k_Y:.10f}, rel_diff = {rel_diff:.3e}"
    )
    assert rel_diff < 1e-4, (
        f"K_vol multi-region routing has a bug: vacuum-BC k_eff(homog 2R) "
        f"differs from k_eff(1R) by {rel_diff:.3e} relative."
    )


def test_probe_b_vacuum_heterogeneous_quadrature_convergence():
    """Vacuum BC heterogeneous sphere — convergence under refinement."""
    print()
    print("=" * 76)
    print("Probe B: vacuum-BC sphere, het 2R (radii=[0.5,1.0], σ=[1,2]),")
    print("         convergence under quadrature refinement")
    print("=" * 76)
    presets = [
        ("BASE",  dict(n_panels_per_region=2, p_order=3, n_angular=24, n_rho=24, n_surf_quad=24, dps=15)),
        ("RICH",  dict(n_panels_per_region=4, p_order=5, n_angular=64, n_rho=48, n_surf_quad=64, dps=20)),
    ]
    print(f"\n  {'preset':<8} {'k_eff':>16}")
    ks = {}
    for name, q in presets:
        k = _solve_sphere_vacuum([0.5, 1.0], [[1.0], [2.0]], **q)
        ks[name] = k
        print(f"  {name:<8} {k:>16.10f}")
    rel = abs(ks["BASE"] - ks["RICH"]) / max(abs(ks["RICH"]), 1e-30)
    print(f"  rel_diff (BASE vs RICH) = {rel:.3e}")


if __name__ == "__main__":
    test_probe_b_vacuum_homogeneous_routing_invariance()
    test_probe_b_vacuum_heterogeneous_quadrature_convergence()
