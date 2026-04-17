r"""Document and catch ERR-026: curvilinear sweep WDD angular closure
converges to wrong fixed-source solution.

The spherical and cylindrical sweeps use a one-directional WDD angular
face-flux closure that, combined with the zero-area face at r=0,
converges to a non-flat solution for constant-source problems.  The
BiCGSTAB transport operator uses a symmetric closure and gives the
correct answer.

These tests serve as both **regression guards** (catch if the sweep
behavior changes) and **evidence** (document the inconsistency for
the V&V capstone report).

Promoted from ``derivations/diagnostics/diag_{11,13,15}_*.py``.

See:
- GitHub Issue #98 (sweep-operator inconsistency)
- GitHub Issue #99 (Phase 3.3–3.4 MMS blocker)
- ``tests/l0_error_catalog.md`` ERR-026
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.geometry import Mesh1D
from orpheus.geometry.coord import CoordSystem
from orpheus.geometry.mesh import BC
from orpheus.sn.geometry import SNMesh
from orpheus.sn.quadrature import GaussLegendre1D
from orpheus.sn.sweep import transport_sweep
from orpheus.sn.operator import (
    build_equation_map_spherical,
    build_transport_linear_operator_spherical,
    angular_flux_to_scalar,
    solution_to_angular_flux_spherical,
)
from scipy.sparse.linalg import bicgstab


def _make_spherical_problem(nx: int = 10, R: float = 10.0, N_ord: int = 8):
    """Build a constant-source, reflective-BC spherical problem."""
    mesh = Mesh1D(
        edges=np.linspace(0.0, R, nx + 1),
        mat_ids=np.ones(nx, dtype=int),
        coord=CoordSystem.SPHERICAL,
        bc_left=BC("reflective"),
        bc_right=BC("reflective"),
    )
    quad = GaussLegendre1D.create(N_ord)
    sn_mesh = SNMesh(mesh, quad)
    sig_t = np.full((nx, 1, 1), 1.0)
    Q_iso = np.full((nx, 1, 1), 1.0)
    return sn_mesh, quad, sig_t, Q_iso


def _solve_sweep(sn_mesh, sig_t, Q_iso, max_iter=200):
    """Source iteration via the spherical sweep."""
    psi_bc = {}
    phi_old = None
    for it in range(max_iter):
        _, phi = transport_sweep(Q_iso, sig_t, sn_mesh, psi_bc)
        if phi_old is not None:
            res = np.linalg.norm(phi - phi_old) / max(np.linalg.norm(phi), 1e-30)
            if res < 1e-14:
                break
        phi_old = phi.copy()
    return phi[:, 0, 0]


def _solve_bicgstab(sn_mesh, quad, sig_t, Q_iso):
    """Direct BiCGSTAB solve of the transport operator."""
    nx = sn_mesh.nx
    N = quad.N
    W = quad.weights.sum()

    eq_map = build_equation_map_spherical(nx, quad, 1)
    T_op = build_transport_linear_operator_spherical(
        eq_map, quad, sig_t, nx, 1,
        sn_mesh.face_areas, sn_mesh.volumes,
        sn_mesh.alpha_half, sn_mesh.redist_dAw, sn_mesh.tau_mm,
    )
    rhs = np.zeros((1, eq_map.n_eq))
    eq_idx = 0
    for ix in range(nx):
        for n in range(N):
            if ix == nx - 1 and quad.mu_x[n] < -1e-15:
                continue
            rhs[:, eq_idx] = 1.0 / W
            eq_idx += 1
    rhs = rhs.ravel(order="F")

    solution, info = bicgstab(T_op, rhs, rtol=1e-14, maxiter=1000)
    assert info == 0, f"BiCGSTAB failed: info={info}"

    fi = solution_to_angular_flux_spherical(solution, eq_map, quad, nx, 1)
    phi_bicg = angular_flux_to_scalar(fi, quad, nx, 1, 1)
    return phi_bicg[:, 0, 0]


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════

pytestmark = [pytest.mark.l1, pytest.mark.catches("ERR-026")]


def test_spherical_sweep_vs_bicgstab_flat_flux():
    r"""BiCGSTAB gives exact flat flux; sweep deviates significantly.

    Constant isotropic source :math:`Q = \Sigma_t = 1`, reflective BCs.
    Expected :math:`\phi = 1` everywhere. BiCGSTAB gets it; the sweep
    converges to a stable but wrong profile with ~35% error at r=0.

    This test **documents** ERR-026 — it does NOT assert the sweep is
    correct, because it isn't. It asserts BiCGSTAB IS correct and that
    the sweep's deviation is at least as large as observed.
    """
    sn_mesh, quad, sig_t, Q_iso = _make_spherical_problem(nx=20)
    phi_bicg = _solve_bicgstab(sn_mesh, quad, sig_t, Q_iso)
    phi_sweep = _solve_sweep(sn_mesh, sig_t, Q_iso)

    # BiCGSTAB must be exact
    np.testing.assert_allclose(phi_bicg, 1.0, atol=1e-10,
                               err_msg="BiCGSTAB should give exact flat flux")

    # Sweep must deviate significantly (ERR-026 evidence)
    sweep_err = np.max(np.abs(phi_sweep - 1.0))
    assert sweep_err > 0.2, (
        f"Sweep error {sweep_err:.4e} is suspiciously small — "
        f"has the sweep bug been fixed? If so, remove ERR-026."
    )


def test_spherical_sweep_error_does_not_converge():
    r"""The sweep's error at cell 0 does NOT converge with refinement.

    This is the defining characteristic of ERR-026: the error is structural,
    not truncation error.  If this test ever fails (orders > 1.5), the
    sweep has been fixed and the MMS blocker (Issue #99) can be resolved.
    """
    errors = []
    for nx in [10, 20, 40]:
        sn_mesh, quad, sig_t, Q_iso = _make_spherical_problem(nx=nx)
        phi_sweep = _solve_sweep(sn_mesh, sig_t, Q_iso)
        errors.append(np.max(np.abs(phi_sweep - 1.0)))

    errors = np.asarray(errors)
    ratios = errors[:-1] / errors[1:]
    orders = np.log2(ratios)

    # The error should NOT converge (orders < 0.5 means diverging or stagnant)
    assert np.all(orders < 0.5), (
        f"Sweep error is converging (orders={orders}) — "
        f"has ERR-026 been fixed? Update Issue #98."
    )


def test_spherical_sweep_conserves_globally():
    r"""Despite the wrong spatial profile, global conservation holds.

    Volume-weighted average :math:`\phi` equals :math:`Q/\Sigma_t = 1`
    to machine precision, because the balance equation is satisfied per
    cell — the error is in the spatial DISTRIBUTION, not the total.
    """
    sn_mesh, quad, sig_t, Q_iso = _make_spherical_problem(nx=20)
    phi_sweep = _solve_sweep(sn_mesh, sig_t, Q_iso)
    V = sn_mesh.volumes[:, 0]
    phi_vol_avg = np.sum(phi_sweep * V) / V.sum()

    np.testing.assert_allclose(phi_vol_avg, 1.0, atol=1e-10,
                               err_msg="Global conservation should hold even with ERR-026")


def test_cartesian_sweep_gives_exact_flat_flux():
    r"""Cartesian sweep has no angular redistribution — flat flux is exact.

    Control test: confirms the issue is specific to curvilinear geometry.
    """
    nx, R = 20, 10.0
    mesh = Mesh1D(
        edges=np.linspace(0.0, R, nx + 1),
        mat_ids=np.ones(nx, dtype=int),
        coord=CoordSystem.CARTESIAN,
    )
    quad = GaussLegendre1D.create(8)
    sn_mesh = SNMesh(mesh, quad)
    sig_t = np.full((nx, 1, 1), 1.0)
    Q_iso = np.full((nx, 1, 1), 1.0)

    phi = _solve_sweep(sn_mesh, sig_t, Q_iso)
    np.testing.assert_allclose(phi, 1.0, atol=1e-10,
                               err_msg="Cartesian sweep should give exact flat flux")
