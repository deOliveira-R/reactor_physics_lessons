r"""L1 verification: each production-tier K assembly in
:mod:`peierls_geometry` matches the unified verification primitive
:func:`K_vol_element_adaptive` at machine precision.

The unified verification primitive is a polymorphic wrapper around
adaptive :func:`mpmath.quad` over the polar Peierls integrand, with
geometry-specific primitives drawn from :class:`CurvilinearGeometry`.
It is the single ground-truth K-element computation for slab,
cylinder, and sphere (and, by extension, future 2-D MoC verification
once a ``MoCGeometry`` is added).

Routing (post-cleanup):

* ``SLAB_POLAR_1D`` — :func:`build_volume_kernel` calls the unified
  adaptive primitive directly; agreement is machine precision by
  construction. Verified at small N (one panel × p=2) to keep the
  adaptive cost bounded.
* ``CYLINDER_1D`` — production-tier general-GL fall-through with
  closed-form :math:`\mathrm{Ki}_1` at GL nodes. Spectral-but-finite-rate
  convergence; gated at 5e-4 with n=128/64.
* ``SPHERE_1D`` — production-tier general-GL fall-through with
  closed-form :math:`e^{-\tau}` at GL nodes. Same regime; gated at 2e-5.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SLAB_POLAR_1D,
    SPHERE_1D,
    CYLINDER_1D,
    K_vol_element_adaptive,
    build_volume_kernel,
    composite_gl_r,
)


_DEFAULT_TOL = 1e-10


def _spot_check(
    geometry,
    radii: np.ndarray,
    sig_t_arr: np.ndarray,
    n_panels: int,
    p_order: int,
    n_angular: int = 32,
    n_rho: int = 16,
    dps_ref: int = 20,
    dps_prod: int = 25,
    tol: float = _DEFAULT_TOL,
) -> None:
    r_nodes, _, panel_bounds = composite_gl_r(
        radii, n_panels_per_region=n_panels, p_order=p_order, dps=dps_prod,
    )
    N = len(r_nodes)

    K_prod = build_volume_kernel(
        geometry, r_nodes, panel_bounds, radii, sig_t_arr,
        n_angular=n_angular, n_rho=n_rho, dps=dps_prod,
    )

    spots = [
        (0, 0),
        (0, N - 1),
        (N - 1, 0),
        (N // 2, N // 2),
        (N // 2, N // 2 + 1)
        if N // 2 + 1 < N else (N // 2, N // 2 - 1),
    ]
    for i, j in spots:
        K_ref_mp = K_vol_element_adaptive(
            geometry, i, j, r_nodes, panel_bounds, radii, sig_t_arr,
            dps=dps_ref,
        )
        K_ref = float(K_ref_mp)
        K_p = float(K_prod[i, j])
        if abs(K_ref) < 1e-15:
            assert abs(K_p - K_ref) < tol, (
                f"{geometry.kind}[{i},{j}]: prod={K_p:.6e}, ref={K_ref:.6e}"
            )
        else:
            rel = abs(K_p - K_ref) / abs(K_ref)
            assert rel < tol, (
                f"{geometry.kind}[{i},{j}]: prod={K_p:.6e}, ref={K_ref:.6e}, "
                f"rel={rel:.3e}"
            )


@pytest.mark.l1
@pytest.mark.verifies("peierls-equation")
def test_slab_K_matches_adaptive_verification() -> None:
    """Slab K matrix (now routed through the adaptive primitive itself)
    matches at machine precision. Small N keeps the cost bounded."""
    L = 1.0
    radii = np.array([L])
    sig_t_arr = np.array([1.0])
    _spot_check(SLAB_POLAR_1D, radii, sig_t_arr, n_panels=1, p_order=2)


@pytest.mark.l1
@pytest.mark.verifies("peierls-unified")
def test_sphere_GL_K_matches_adaptive_verification() -> None:
    """Sphere general-GL K matrix matches adaptive ``mpmath.quad``
    polar-form verification at 2e-5 with n=128/64."""
    R = 1.0
    radii = np.array([R])
    sig_t_arr = np.array([1.0])
    _spot_check(
        SPHERE_1D, radii, sig_t_arr,
        n_panels=2, p_order=3, n_angular=128, n_rho=64, tol=2e-5,
    )


@pytest.mark.l1
@pytest.mark.verifies("peierls-unified")
def test_cylinder_K_matches_adaptive_verification() -> None:
    """Cylinder-1d (Ki_n at GL nodes) K matches adaptive verification
    at 5e-4 with n=128/64. Far-corner entries are the slowest to
    converge due to the sqrt cusp at tangent angles."""
    R = 1.0
    radii = np.array([R])
    sig_t_arr = np.array([1.0])
    _spot_check(
        CYLINDER_1D, radii, sig_t_arr,
        n_panels=2, p_order=3, n_angular=128, n_rho=64, tol=5e-4,
    )


@pytest.mark.l1
@pytest.mark.verifies("peierls-equation")
def test_slab_heterogeneous_matches_adaptive_verification() -> None:
    """Heterogeneous slab K matches adaptive verification at machine
    precision."""
    radii = np.array([1.0, 2.0])
    sig_t_arr = np.array([1.0, 0.5])
    _spot_check(SLAB_POLAR_1D, radii, sig_t_arr, n_panels=1, p_order=2)
