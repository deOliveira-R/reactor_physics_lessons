r"""L1 MMS tests for SN — 1D spherical and cylindrical geometries.

Phase 3.3 (spherical) and Phase 3.4 (cylindrical) of the verification
campaign.  Both use isotropic-in-angle ansatz :math:`\psi_n(r) = A(r)/W`
with :math:`A(r) = \sin(\pi r/R)`, so the angular redistribution terms
vanish and the spatial DD convergence rate is isolated.

See :doc:`/theory/discrete_ordinates` (curvilinear MMS sections)
for the full derivation.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.sn_mms import (
    build_spherical_mms_case,
    build_cylindrical_mms_case,
)
from orpheus.sn import solve_sn_fixed_source


def _l2_1d(phi_num: np.ndarray, phi_ref: np.ndarray, volumes: np.ndarray) -> float:
    """Volume-weighted L2 norm for 1D curvilinear meshes."""
    diff = phi_num - phi_ref
    return float(np.sqrt(np.sum(volumes * diff * diff)))


# ═══════════════════════════════════════════════════════════════════════
# Phase 3.3 — 1D Spherical MMS
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
@pytest.mark.verifies(
    "transport-spherical",
    "sn-mms-spherical-psi", "sn-mms-spherical-qext",
)
def test_sn_spherical_mms_converges_second_order():
    r"""Spherical SN with isotropic ansatz shows :math:`\mathcal{O}(h^2)`.

    The ansatz :math:`A(r) = \sin(\pi r/R)` vanishes at r=0 (symmetry)
    and r=R (vacuum).  Angular redistribution vanishes for isotropic
    flux, so only the radial DD closure drives the convergence rate.
    """
    case = build_spherical_mms_case()

    n_cells = [20, 40, 80, 160]
    errors = []
    for nc in n_cells:
        mesh = case.build_mesh(nc)
        Q = case.external_source(mesh)
        result = solve_sn_fixed_source(
            case.materials, mesh, case.quadrature, Q,
            max_inner=500, inner_tol=1e-13,
        )
        phi_num = result.scalar_flux[:, 0, 0]
        phi_ref = case.phi_exact(mesh.centers)
        errors.append(_l2_1d(phi_num, phi_ref, mesh.volumes))

    errors = np.asarray(errors)
    orders = np.log2(errors[:-1] / errors[1:])

    assert np.all(orders > 1.9), (
        f"Expected O(h^2), got orders={orders}, errors={errors}"
    )
    assert 1e-8 < errors[-1] < 1e-3


# ═══════════════════════════════════════════════════════════════════════
# Phase 3.4 — 1D Cylindrical MMS
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
@pytest.mark.verifies(
    "transport-cylindrical",
    "sn-mms-cylindrical-psi", "sn-mms-cylindrical-qext",
)
def test_sn_cylindrical_mms_converges_second_order():
    r"""Cylindrical SN with isotropic ansatz shows :math:`\mathcal{O}(h^2)`.

    Same structure as the spherical test but on a cylindrical mesh
    with Product quadrature (polar × azimuthal).  Azimuthal
    redistribution vanishes for isotropic flux.
    """
    case = build_cylindrical_mms_case()

    n_cells = [20, 40, 80, 160]
    errors = []
    for nc in n_cells:
        mesh = case.build_mesh(nc)
        Q = case.external_source(mesh)
        result = solve_sn_fixed_source(
            case.materials, mesh, case.quadrature, Q,
            max_inner=500, inner_tol=1e-13,
        )
        phi_num = result.scalar_flux[:, 0, 0]
        phi_ref = case.phi_exact(mesh.centers)
        errors.append(_l2_1d(phi_num, phi_ref, mesh.volumes))

    errors = np.asarray(errors)
    orders = np.log2(errors[:-1] / errors[1:])

    assert np.all(orders > 1.9), (
        f"Expected O(h^2), got orders={orders}, errors={errors}"
    )
    assert 1e-8 < errors[-1] < 1e-3
