"""Probe G: mode-0 normalization mismatch — the suspected bug.

Created by numerics-investigator on 2026-04-25.

The Probe F / probe_f_mode_isolation results showed that on Variant B
(sphere 1G/2R-Z with B's strong-scatterer outer region), adding mode-1
to mode-0 jumps k_eff by +84 % (0.55 → 1.02), and rank-N converges to
+66 % above k_inf=0.648.

In contrast on the 1R control (σ_t=1), the same step jumps by +35 %
and rank-2 lands at -1.10 % (essentially correct). This is the
explicitly-acknowledged routing in build_white_bc_correction_rank_n
(line 3618-3622 + docstring §"Mode-0 convention"):

   mode 0 → legacy compute_P_esc / compute_G_bc (NO surface Jacobian)
   mode n ≥ 1 → compute_P_esc_mode / compute_G_bc_mode (WITH (ρ_max/R)² Jacobian)

mixing two inconsistent normalisations into the same rank-N K_bc.

THIS PROBE: replaces the mode-0 routing with the canonical
compute_P_esc_mode(n=0) / compute_G_bc_mode(n=0) — Jacobian-weighted —
and re-runs rank-N on the same configurations. If the canonical mode-0
makes rank-N converge cleanly to k_inf in BOTH 1R and 2R-Z, the bug is
identified. If not, there's a second-order bug.

Result interpretation:
  LEGACY rank-2 1R = -1.10 %  (pre-tuned by mode-0 routing for this case)
  CANONICAL rank-2 1R = -29.3 %
  LEGACY rank-2 2R-Z = +56.7 %
  CANONICAL rank-2 2R-Z = -28.0 %

So canonical normalisation makes rank-N CONSISTENT (both 1R and 2R-Z
plateau near -25 % at large N), while legacy normalisation gives a
spurious "good" answer in 1R that breaks in MR.

Conclusion: the mode-0 routing in build_closure_operator (peierls_geometry.py
line 3618-3622) is a tuning hack that masks a deeper convergence problem
in the rank-N Marshak closure. The 2R-Z catastrophe surfaces this hack
because the per-region σ_t step amplifies any normalization mismatch.

The fix is NOT to change the mode-0 routing alone (canonical gives
worse 1R), but to re-derive the rank-N partial-current normalisation
so mode-0 (legacy) and mode-n≥1 (Marshak with Jacobian) live in the
same expansion space. This is the F.4-style derivation the 2026-04-22
research falsified at Class A — but on Class B the rank-N approach was
NEVER fully audited under MR×MG.

Promotion: this probe is a permanent test pinning the published
docstring-table values (1R rank-N table at lines 3934-3961 of
peierls_geometry.py). Move to tests/derivations/test_peierls_rank_n_bc.py
as `test_rank_n_mode0_normalization_table_1r_2rZ`.
"""

from __future__ import annotations

import numpy as np

from orpheus.derivations._xs_library import get_xs
from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    BoundaryClosureOperator,
    build_volume_kernel,
    composite_gl_r,
    compute_G_bc,
    compute_G_bc_mode,
    compute_P_esc,
    compute_P_esc_mode,
    reflection_marshak,
)


_QUAD = dict(n_panels_per_region=2, p_order=3,
             n_angular=24, n_rho=24, n_surf_quad=24, dps=15)


def _custom_K_bc(modes, radii, sig_t, *, mode0_canonical):
    R_cell = float(radii[-1])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, _QUAD["n_panels_per_region"], _QUAD["p_order"],
        dps=_QUAD["dps"], inner_radius=0.0,
    )
    N_r = len(r_nodes)
    sig_t_n = np.array([sig_t[SPHERE_1D.which_annulus(float(rj), radii)]
                        for rj in r_nodes])
    rv = np.array([SPHERE_1D.radial_volume_weight(float(rj)) for rj in r_nodes])
    divisor = SPHERE_1D.rank1_surface_divisor(R_cell)
    N_modes = max(modes) + 1
    P = np.zeros((N_modes, N_r))
    G = np.zeros((N_r, N_modes))
    R_mat = np.zeros((N_modes, N_modes))
    R_full = reflection_marshak(N_modes)
    for n in modes:
        if n == 0 and not mode0_canonical:
            P_esc = compute_P_esc(SPHERE_1D, r_nodes, radii, sig_t,
                                  n_angular=_QUAD["n_angular"], dps=_QUAD["dps"])
            G_bc = compute_G_bc(SPHERE_1D, r_nodes, radii, sig_t,
                                n_surf_quad=_QUAD["n_surf_quad"], dps=_QUAD["dps"])
        else:
            P_esc = compute_P_esc_mode(SPHERE_1D, r_nodes, radii, sig_t, n,
                                       n_angular=_QUAD["n_angular"], dps=_QUAD["dps"])
            G_bc = compute_G_bc_mode(SPHERE_1D, r_nodes, radii, sig_t, n,
                                     n_surf_quad=_QUAD["n_surf_quad"], dps=_QUAD["dps"])
        P[n, :] = rv * r_wts * P_esc
        G[:, n] = sig_t_n * G_bc / divisor
        R_mat[n, n] = R_full[n, n]
    op = BoundaryClosureOperator(P=P, G=G, R=R_mat)
    return op.as_matrix(), r_nodes, r_wts, panels, sig_t_n


def _solve_keff(K_total, sig_t_n, sig_s_n, nu_sig_f_n):
    A = np.diag(sig_t_n) - K_total * sig_s_n[None, :]
    Op = np.linalg.solve(A, K_total * nu_sig_f_n[None, :])
    eigvals = np.linalg.eigvals(Op)
    return float(np.max(eigvals.real))


def _run_case(label, radii, sig_t, sig_s_per_region, nu_sig_f_per_region, kref):
    K_bc_dummy, r_nodes, r_wts, panels, sig_t_n = _custom_K_bc(
        [0], radii, sig_t, mode0_canonical=False,
    )
    sig_s_n = np.array([sig_s_per_region[SPHERE_1D.which_annulus(float(rj), radii)]
                        for rj in r_nodes])
    nu_sig_f_n = np.array([nu_sig_f_per_region[SPHERE_1D.which_annulus(float(rj), radii)]
                           for rj in r_nodes])
    K_vol = build_volume_kernel(SPHERE_1D, r_nodes, panels, radii, sig_t,
                                n_angular=_QUAD["n_angular"], n_rho=_QUAD["n_rho"],
                                dps=_QUAD["dps"])
    print(f"\n--- {label} ---")
    print(f"  k_inf reference = {kref:.10f}")
    for canonical_label, mode0_canonical in [("LEGACY", False), ("CANONICAL", True)]:
        for N in (1, 2, 3, 5):
            K_bc, _, _, _, _ = _custom_K_bc(list(range(N)), radii, sig_t,
                                            mode0_canonical=mode0_canonical)
            k = _solve_keff(K_vol + K_bc, sig_t_n, sig_s_n, nu_sig_f_n)
            err = (k - kref) / kref * 100
            print(f"  {canonical_label:<10} rank-{N:>1}: k = {k:.6f}  err = {err:+.2f}%")


def main():
    print("=" * 76)
    print("Probe G: mode-0 normalisation routing — legacy vs canonical")
    print("=" * 76)
    xs_A = get_xs("A", "1g")
    xs_B = get_xs("B", "1g")
    _run_case(
        "1R control: σ_t=[1.0], A only, k_inf=1.5",
        np.array([1.0]), np.array([1.0]),
        np.array([float(xs_A["sig_s"][0, 0])]),
        np.array([float(xs_A["nu"][0] * xs_A["sig_f"][0])]),
        1.5,
    )
    _run_case(
        "2R-Z VarB: σ_t=[1,2], A inner / B outer, k_inf=0.6479728",
        np.array([0.5, 1.0]), np.array([1.0, 2.0]),
        np.array([float(xs_A["sig_s"][0, 0]), float(xs_B["sig_s"][0, 0])]),
        np.array([float(xs_A["nu"][0] * xs_A["sig_f"][0]),
                  float(xs_B["nu"][0] * xs_B["sig_f"][0])]),
        0.6479728191,
    )
    # Thickness scan (Probe H consolidated)
    print("\n" + "=" * 76)
    print("Probe H: thickness scan on 2R-Z VarB (uniform σ multiplier m)")
    print("  (k_inf is INVARIANT under uniform multiplicative scaling)")
    print("=" * 76)
    sig_s_pr_base = np.array([float(xs_A["sig_s"][0, 0]),
                              float(xs_B["sig_s"][0, 0])])
    nu_sig_f_pr_base = np.array([float(xs_A["nu"][0] * xs_A["sig_f"][0]),
                                 float(xs_B["nu"][0] * xs_B["sig_f"][0])])
    for m in (0.5, 1.0, 2.0, 5.0, 10.0):
        radii = np.array([0.5, 1.0])
        sig_t = m * np.array([1.0, 2.0])
        sig_s_pr = m * sig_s_pr_base
        nu_sig_f_pr = m * nu_sig_f_pr_base
        K_bc_dummy, r_nodes, r_wts, panels, sig_t_n = _custom_K_bc(
            [0], radii, sig_t, mode0_canonical=False,
        )
        sig_s_n = np.array([sig_s_pr[SPHERE_1D.which_annulus(float(rj), radii)]
                            for rj in r_nodes])
        nu_sig_f_n = np.array([nu_sig_f_pr[SPHERE_1D.which_annulus(float(rj), radii)]
                               for rj in r_nodes])
        K_vol = build_volume_kernel(SPHERE_1D, r_nodes, panels, radii, sig_t,
                                    n_angular=_QUAD["n_angular"],
                                    n_rho=_QUAD["n_rho"], dps=_QUAD["dps"])
        print(f"\nMultiplier m = {m}: σ_t = {sig_t}, R_outer = 1.0, "
              f"MFP_outer = {1/sig_t[1]:.3f}")
        for N in (1, 2, 3, 5):
            K_bc, _, _, _, _ = _custom_K_bc(list(range(N)), radii, sig_t,
                                            mode0_canonical=False)
            k = _solve_keff(K_vol + K_bc, sig_t_n, sig_s_n, nu_sig_f_n)
            print(f"  rank-{N:>1}: k = {k:.6f}")


if __name__ == "__main__":
    main()
