"""Diagnostic 02: Compare K·1 row-sum of specular vs Mark vs Hébert
on the thin homogeneous sphere (τ_R = 2.5).

Created by numerics-investigator on 2026-04-27.

Diagnostic 01 showed that specular gives σ_t·K·1 ≈ 0.5 on the thin
sphere at all N. This is a factor of 2 deficit.

Hypothesis: the white_hebert closure (which empirically recovers k_inf
for thin sphere 1G/1R to within 0.2 %) should give σ_t·K·1 ≈ 1.0. If
white_rank1_mark gives ~0.5 (since Mark = specular at rank-1 for sphere)
and Hébert gives ~1.0, then the Hébert geometric-series factor
(1 - P_ss)^{-1} is exactly what specular is missing.

P_ss (single-surface escape probability) for a homogeneous sphere is
the chord-averaged escape probability for an isotropic surface emitter.

If this diagnostic confirms that K_bc^Hébert·1 ≈ 1 - K_vol·1 (so
σ_t·K·1 ≈ 1) but K_bc^specular·1 ≈ (1 - K_vol·1) · (1 - P_ss), then
the specular K_bc is missing a re-emission feedback that Hébert captures
through the (1 - P_ss)⁻¹ multiplier.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _build_full_K_per_group,
    build_volume_kernel,
    composite_gl_r,
    compute_P_ss_sphere,
)


def _build_K(geometry, R, sigt, *, n_bc_modes, closure):
    radii = np.array([R])
    sig_t_g = np.array([sigt])
    r_nodes, r_wts, panels = composite_gl_r(radii, 2, 4, dps=20)
    K_vol = build_volume_kernel(
        geometry, r_nodes, panels, radii, sig_t_g,
        n_angular=24, n_rho=24, dps=20,
    )
    K_full = _build_full_K_per_group(
        geometry, r_nodes, r_wts, panels, radii, sig_t_g, closure,
        n_angular=24, n_rho=24, n_surf_quad=24, dps=20,
        n_bc_modes=n_bc_modes,
    )
    return r_nodes, K_vol, K_full - K_vol


def _print_summary(label, sigt, K_vol, K_bc):
    Kvol1 = sigt * (K_vol @ np.ones(K_vol.shape[1]))
    Kbc1 = sigt * (K_bc @ np.ones(K_bc.shape[1]))
    K1 = Kvol1 + Kbc1
    print(
        f"  {label:34s} "
        f"avg σt·Kvol·1={Kvol1.mean():.6f} "
        f"avg σt·Kbc·1={Kbc1.mean():.6f} "
        f"avg σt·K·1={K1.mean():.6f} "
        f"err_avg={K1.mean()-1.0:+.4e}"
    )
    return K1


@pytest.mark.parametrize(
    "tag,R,sigt", [
        ("thin τR=2.5", 5.0, 0.5),
        ("thick τR=5.0", 5.0, 1.0),
    ],
)
def test_compare_closures_row_sum(tag, R, sigt, capsys):
    with capsys.disabled():
        radii = np.array([R])
        sig_t_g = np.array([sigt])
        P_ss = compute_P_ss_sphere(
            radii, sig_t_g, n_quad=24, dps=20,
        )
        print(f"\n=== {tag}: P_ss = {P_ss:.6f}, 1/(1-P_ss) = "
              f"{1.0/(1.0-P_ss):.6f} ===")

        # Mark rank-1
        rn, Kv, Kb = _build_K(
            SPHERE_1D, R, sigt, n_bc_modes=1, closure="white_rank1_mark",
        )
        K1_mark = _print_summary("white_rank1_mark, N=1", sigt, Kv, Kb)

        # Hébert rank-1
        rn, Kv, Kb = _build_K(
            SPHERE_1D, R, sigt, n_bc_modes=1, closure="white_hebert",
        )
        K1_heb = _print_summary("white_hebert,      N=1", sigt, Kv, Kb)

        # Specular rank-1, 2, 4
        for N in (1, 2, 4):
            rn, Kv, Kb = _build_K(
                SPHERE_1D, R, sigt, n_bc_modes=N, closure="specular",
            )
            _print_summary(f"specular,          N={N}", sigt, Kv, Kb)

        # Acid test: does Hébert satisfy the contract σt·K·1 ≈ 1?
        if tag.startswith("thin"):
            assert abs(K1_heb.mean() - 1.0) < 5e-3, (
                f"thin sphere Hébert violates row-sum contract: "
                f"avg σt·K·1 = {K1_heb.mean():.6f}, expected 1.0"
            )
            # Confirm the factor-of-2 deficit (rough): Mark gives ~0.5 because
            # at rank-1 P_ss ≈ 0.5 makes (1-P_ss)·1 ≈ 0.5
            print(
                f"\n  [DIAGNOSIS] At rank-1 thin: Mark σt·K·1 = "
                f"{K1_mark.mean():.6f}, P_ss = {P_ss:.6f}; "
                f"(1-P_ss) ≈ {1-P_ss:.6f}"
            )


if __name__ == "__main__":
    import sys
    for tag, R, sigt in [
        ("thin τR=2.5", 5.0, 0.5),
        ("thick τR=5.0", 5.0, 1.0),
    ]:
        radii = np.array([R])
        sig_t_g = np.array([sigt])
        P_ss = compute_P_ss_sphere(
            radii, sig_t_g, n_quad=24, dps=20,
        )
        print(f"\n=== {tag}: P_ss = {P_ss:.6f}, 1/(1-P_ss) = "
              f"{1.0/(1.0-P_ss):.6f} ===")

        rn, Kv, Kb = _build_K(
            SPHERE_1D, R, sigt, n_bc_modes=1, closure="white_rank1_mark",
        )
        K1_mark = _print_summary("white_rank1_mark, N=1", sigt, Kv, Kb)

        rn, Kv, Kb = _build_K(
            SPHERE_1D, R, sigt, n_bc_modes=1, closure="white_hebert",
        )
        K1_heb = _print_summary("white_hebert,      N=1", sigt, Kv, Kb)

        for N in (1, 2, 4, 6):
            rn, Kv, Kb = _build_K(
                SPHERE_1D, R, sigt, n_bc_modes=N, closure="specular",
            )
            _print_summary(f"specular,          N={N}", sigt, Kv, Kb)
    sys.exit(0)
