r"""L1 MMS test for SN — P1 anisotropic scattering.

Phase 3.5 of the verification campaign.  Verifies that the P1
anisotropic scattering source assembly
(the :class:`~orpheus.transport.operators.scattering.ScatteringOperator`'s
ℓ ≥ 1 redistribution, driven as the SI gain) preserves
:math:`\mathcal{O}(h^{2})` diamond-difference convergence on a 1D
Cartesian slab with a weakly anisotropic angular flux ansatz.

The ansatz :math:`\psi_n(x) = (1/W)(A(x) + \alpha\,\mu_n\,B(x))`
with :math:`A = B = \sin(\pi x/L)` and small :math:`\alpha` feeds
the P1 scattering moment through the current
:math:`J(x) = \alpha\,B(x)/3`, exercising the
:math:`\Sigma_s^{(1)}` slot that isotropic (P0) MMS tests skip.

See :doc:`/theory/verification/sn` (P1 scattering MMS section)
for the derivation.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.continuous.mms.sn import build_p1_aniso_mms_case
from orpheus.sn import solve_sn_fixed_source

from tests.sn._test_helpers import volume_weighted_l2


@pytest.mark.l1
@pytest.mark.verifies(
    "transport-cartesian", "dd-cartesian-1d", "dd-slab",
    "pn-scatter", "sn-mms-p1-psi", "sn-mms-p1-qext",
)
def test_sn_p1_aniso_mms_converges_second_order():
    r"""P1 anisotropic scattering MMS on a 1D slab shows :math:`\mathcal{O}(h^2)`.

    The ansatz has weak :math:`\mu`-dependence that feeds the P1
    scattering moment.  A bug that drops the P1 contribution, uses
    the wrong :math:`(2\ell+1)` factor, or transposes the scattering
    matrix index would break the convergence rate.
    """
    case = build_p1_aniso_mms_case()

    n_cells = [20, 40, 80, 160]
    errors = []
    for nc in n_cells:
        mesh = case.build_mesh(nc)
        Q = case.external_source(mesh)
        result = solve_sn_fixed_source(
            case.materials, mesh, case.quadrature, Q,
            boundary_condition="vacuum",
            scattering_order=1,
            max_inner=500,
            inner_tol=1e-13,
        )
        phi_num = result.scalar_flux.values[0, :]  # PR-INDEX-5: g=0 radial slice
        phi_ref = case.phi_exact(mesh.centers)
        errors.append(volume_weighted_l2(phi_num, phi_ref, mesh.widths))

    errors = np.asarray(errors)
    orders = np.log2(errors[:-1] / errors[1:])

    assert np.all(orders > 1.9), (
        f"Expected O(h^2) convergence, got orders={orders} "
        f"from errors={errors}"
    )
    assert 1e-8 < errors[-1] < 1e-4


@pytest.mark.sentinel
@pytest.mark.l1
@pytest.mark.verifies("pn-scatter", "sn-mms-p1-qext")
def test_sn_p1_aniso_mms_source_degrades_to_p0():
    r"""With :math:`\alpha = 0`, the P1 MMS source reduces to the P0 case.

    Regression guard: the manufactured source formula must be
    consistent with the isotropic case when the μ-dependent terms
    vanish.
    """
    from orpheus.derivations.continuous.mms.sn import build_1d_slab_mms_case

    case_p0 = build_1d_slab_mms_case(sigma_t=1.0, sigma_s=0.5)
    case_p1 = build_p1_aniso_mms_case(
        sigma_t=1.0, sigma_s0=0.5, sigma_s1=0.2, alpha=0.0,
    )

    mesh = case_p0.build_mesh(32)
    Q_p0 = case_p0.external_source(mesh)  # (N, 32, 1, 1)
    Q_p1 = case_p1.external_source(mesh)  # (N, 32, 1, 1)

    # With alpha=0, the P1 source should equal the P0 source
    np.testing.assert_allclose(Q_p1, Q_p0, atol=1e-14,
                               err_msg="P1 source with α=0 must match P0 source")
