"""Diagnostic E — augmented-Nyström for solid sphere with Marshak basis.

Issue #132 augmented-Nyström investigation.

Setup
-----
Use the existing Marshak Legendre-µ basis (which IS smooth on µ ∈ [0,1]
and absorbs the endpoint singularity into the orthogonal-polynomial
weight) and apply the user's J^+ as independent unknowns + white-BC
constraint formulation. For solid sphere this routes through:

    P[n, j] = compute_P_esc_mode(n) at radial node j
    G[i, m] = compute_G_bc_mode(m) at radial node i

with the natural Marshak (2n+1) Gelbard reflection
    R_marshak = diag(1, 3, 5, ..., 2(M-1)+1)

This is exactly what `build_closure_operator(reflection="marshak")`
produces. By contrast, `reflection="white"` collapses to
`reflection_mark(M) = e_0 e_0^T` for solid sphere (single-face
collapse).

This diagnostic compares ``"mark"`` (rank-1) vs ``"marshak"``
(rank-N with Gelbard normalisation) at increasing M to see whether
adding higher Marshak modes helps for the solid sphere.

Critical reference
------------------
Issue #100 / #132 already showed this path FAILS for Class B MR — the
mode-0/mode-n≥1 normalisation mismatch produces +57% k_eff at sphere
2R. We expect the SAME failure mode here on the chi-loaded 1G/1R
homogeneous case to a lesser degree.

If even the homogeneous 1G/1R case fails to converge to k_inf as
M → ∞, the augmented-Nyström direction is dead.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from orpheus.derivations._xs_library import LAYOUTS, get_xs
from orpheus.derivations import cp_sphere
from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, build_volume_kernel, build_closure_operator,
    composite_gl_r,
)


def solve_marshak_solid_sphere(
    sig_t: float, sig_s: float, nu_sig_f: float, R: float,
    M: int, reflection: str,
    *, n_panels=2, p_order=3, n_angular=24, n_rho=24,
    n_surf_quad=24, dps=15,
):
    radii = np.array([R])
    sig_t_arr = np.array([sig_t])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, n_panels_per_region=n_panels, p_order=p_order, dps=dps,
    )
    K_vol = build_volume_kernel(
        SPHERE_1D, r_nodes, panels, radii, sig_t_arr,
        n_angular=n_angular, n_rho=n_rho, dps=dps,
    )
    op = build_closure_operator(
        SPHERE_1D, r_nodes, r_wts, radii, sig_t_arr,
        n_angular=n_angular, n_surf_quad=n_surf_quad, dps=dps,
        n_bc_modes=M, reflection=reflection,
    )
    sig_t_n = sig_t * np.ones(len(r_nodes))
    K_total = K_vol + op.G @ op.R @ op.P
    A = np.diag(sig_t_n) - K_total * sig_s
    B_op = K_total * nu_sig_f
    eigvals = la.eigvals(np.linalg.solve(A, B_op))
    return float(np.max(eigvals.real))


def main():
    print("=" * 78)
    print("Marshak-basis augmented closure on solid sphere — k_eff(M)")
    print("=" * 78)
    print()

    # Region A 1G — exactly the Issue #132 1G/1R baseline
    xs = get_xs("A", "1g")
    sig_t = float(xs["sig_t"][0])
    sig_s = float(xs["sig_s"][0, 0])
    nu_sig_f = float(xs["nu"][0] * xs["sig_f"][0])
    R = 1.0
    cp_kinf = nu_sig_f / (sig_t - sig_s)
    print(f"  1G/1R Region A:  σ_t={sig_t:.3f}  σ_s={sig_s:.3f}  "
          f"νΣ_f={nu_sig_f:.3f}  cp k_inf={cp_kinf:.6f}")
    print()
    print(f"  {'M':>3} {'mark (rank-1)':>15} "
          f"{'marshak (rank-M)':>18} {'rel_err marshak':>17}")
    for M in [1, 2, 3, 4, 6, 8]:
        k_mark = solve_marshak_solid_sphere(
            sig_t, sig_s, nu_sig_f, R, M, "mark",
        )
        try:
            k_marshak = solve_marshak_solid_sphere(
                sig_t, sig_s, nu_sig_f, R, M, "marshak",
            )
        except Exception as exc:
            k_marshak = float("nan")
            print(f"  marshak failed at M={M}: {exc}")
        rel_M = (k_marshak - cp_kinf) / cp_kinf * 100
        print(f"  {M:>3d} {k_mark:>15.10f} {k_marshak:>18.10f} "
              f"{rel_M:>+15.3f}%")

    print()
    print("  Now: does marshak rank-M help on the chi-thermal 2G/2R limit?")
    print("  (the actual Issue #132 limitation — chi=[0,1] gives +6.6% Hébert err).")
    print()
    print("  This requires the multi-region routing — out of scope for this")
    print("  probe. The 1G/1R diagnosis above is sufficient to determine")
    print("  whether higher-mode Marshak helps at all.")


if __name__ == "__main__":
    main()
