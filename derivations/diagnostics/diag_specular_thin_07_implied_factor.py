"""Diagnostic 07: implied uniform K_bc multiplier vs N for thin sphere.

Created by numerics-investigator on 2026-04-27.

Find α such that K = K_vol + α·K_bc^specular gives k_eff = k_inf.
This α represents the "missing factor" that perfect specular needs.

If α is N-independent → it's a global geometric factor (Hébert-like).
If α is N-dependent → rank-N specular K_bc is wrong by an N-dependent
amount (basis mismatch / over- or under-counting).
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _build_full_K_per_group,
    build_volume_kernel,
    composite_gl_r,
)
from scipy.optimize import brentq


def _build_K_components(geom, R, sigt, *, n_bc_modes):
    radii = np.array([R])
    sig_t_g = np.array([sigt])
    r_nodes, r_wts, panels = composite_gl_r(radii, 2, 4, dps=20)
    K_vol = build_volume_kernel(
        geom, r_nodes, panels, radii, sig_t_g,
        n_angular=24, n_rho=24, dps=20,
    )
    K_full = _build_full_K_per_group(
        geom, r_nodes, r_wts, panels, radii, sig_t_g, "specular",
        n_angular=24, n_rho=24, n_surf_quad=24, dps=20,
        n_bc_modes=n_bc_modes,
    )
    return r_nodes, K_vol, K_full - K_vol


def _solve(K, sigt, sigs, nuf):
    N = K.shape[0]
    A = sigt * np.eye(N) - sigs * K
    B = nuf * K
    M = np.linalg.solve(A, B)
    eigvals = np.linalg.eigvals(M)
    real_mask = np.abs(eigvals.imag) < 1e-10
    return float(eigvals[real_mask].real.max())


@pytest.mark.parametrize(
    "tag,R,sigt,sigs,nuf",
    [
        ("very thin τR=1.0", 5.0, 0.2, 0.16, 0.01),
        ("thin τR=2.5", 5.0, 0.5, 0.38, 0.025),
        ("medium τR=3.5", 5.0, 0.7, 0.5, 0.5),
        ("thick τR=5.0 fuelA", 5.0, 1.0, 0.5, 0.75),
    ],
)
def test_implied_factor(tag, R, sigt, sigs, nuf, capsys):
    with capsys.disabled():
        k_inf = nuf / (sigt - sigs)
        print(f"\n=== {tag}: σ_t={sigt}, k_inf={k_inf:.6f} ===")
        for N in (1, 2, 3, 4, 6, 8):
            rn, Kv, Kb = _build_K_components(
                SPHERE_1D, R, sigt, n_bc_modes=N,
            )

            def f(alpha):
                K = Kv + alpha * Kb
                return _solve(K, sigt, sigs, nuf) - k_inf

            try:
                alpha = brentq(f, 0.0, 100.0, xtol=1e-8)
            except Exception as e:
                alpha = float("nan")
                print(f"  N={N}: brentq failed: {e}")
                continue
            k_baseline = _solve(Kv + Kb, sigt, sigs, nuf)
            err_b = (k_baseline - k_inf) / k_inf
            print(f"  N={N}: implied α = {alpha:.6f}, "
                  f"baseline α=1 gives {err_b*100:+.4f}%")


if __name__ == "__main__":
    import sys
    for tag, R, sigt, sigs, nuf in [
        ("very thin τR=1.0", 5.0, 0.2, 0.16, 0.01),
        ("thin τR=2.5", 5.0, 0.5, 0.38, 0.025),
        ("medium τR=3.5", 5.0, 0.7, 0.5, 0.5),
        ("thick τR=5.0 fuelA", 5.0, 1.0, 0.5, 0.75),
    ]:
        k_inf = nuf / (sigt - sigs)
        print(f"\n=== {tag}: σ_t={sigt}, k_inf={k_inf:.6f} ===")
        for N in (1, 2, 3, 4, 6, 8):
            rn, Kv, Kb = _build_K_components(
                SPHERE_1D, R, sigt, n_bc_modes=N,
            )
            def f(alpha, _Kv=Kv, _Kb=Kb, _sigt=sigt, _sigs=sigs, _nuf=nuf,
                  _kinf=k_inf):
                K = _Kv + alpha * _Kb
                return _solve(K, _sigt, _sigs, _nuf) - _kinf
            try:
                alpha = brentq(f, 0.0, 100.0, xtol=1e-8)
            except Exception as e:
                alpha = float("nan")
                print(f"  N={N}: brentq failed: {e}")
                continue
            k_baseline = _solve(Kv + Kb, sigt, sigs, nuf)
            err_b = (k_baseline - k_inf) / k_inf
            print(f"  N={N}: implied α = {alpha:.6f}, "
                  f"baseline α=1 gives {err_b*100:+.4f}%")
    sys.exit(0)
