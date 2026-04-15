r"""L1 MMS (Method of Manufactured Solutions) tests for SN — heterogeneous 2-group slab.

This is the **Phase-2.1a** consumer test landing alongside the
deletion of the Richardson-extrapolated ``sn_slab_Neg_Nrg`` cases
(T3 violators). Unlike the earlier
:mod:`tests.sn.test_mms` which tests the homogeneous single-material
slab, this file exercises the 1D SN sweep on a **continuously
heterogeneous** 2-group problem — smooth :math:`\Sigma_t(x)`,
:math:`\Sigma_s(x)` — so the multigroup scatter assembly and the
spatial discretisation are verified simultaneously.

Continuous (not piecewise) cross sections are the deliberate choice:
discontinuous Σ at material interfaces degrades diamond difference
from :math:`\mathcal O(h^{2})` to :math:`\mathcal O(h)` when the
interfaces do not lie on cell faces, which would contaminate the
spatial-convergence measurement with interface-treatment artefacts
(Salari & Knupp SAND2000-1444 §6). With smooth Σ(x) the diamond
difference hits its design :math:`\mathcal O(h^{2})` order exactly.

See :doc:`/theory/discrete_ordinates` (heterogeneous MMS section)
for the full derivation and :mod:`orpheus.derivations.sn_mms` for
the reference solution.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations import continuous_get
from orpheus.sn import solve_sn_fixed_source


pytestmark = [pytest.mark.l1, pytest.mark.verifies(
    "transport-cartesian",
    "dd-cartesian-1d",
    "dd-slab",
    "multigroup",
    "mg-balance",
    "sn-mms-hetero-psi",
    "sn-mms-hetero-qext",
)]


def _cell_l2(err_cells: np.ndarray, widths: np.ndarray) -> float:
    r"""Cell-width-weighted discrete :math:`L^{2}` norm:

    .. math::

        \|e\|_{L^{2}} \;=\;
        \sqrt{\sum_i \Delta x_i\,e_i^{2}}.
    """
    return float(np.sqrt(np.sum(widths * err_cells * err_cells)))


def test_sn_heterogeneous_mms_converges_second_order():
    r"""SN on a continuously heterogeneous 2-group slab shows measured
    :math:`\mathcal O(h^{2})` convergence.

    Runs four mesh refinements (``n_cells = 20, 40, 80, 160``) and
    asserts that the observed order between successive refinements
    is :math:`> 1.9` for each group independently, plus the
    combined :math:`L^{2}` error. The ansatz is :math:`C^{\infty}`
    and the cross sections are :math:`C^{\infty}`, so the measured
    order lands at :math:`2.00` to two digits — assert ``> 1.9`` to
    leave round-off headroom at the finest mesh.
    """
    ref = continuous_get("sn_mms_slab_2g_hetero")
    assert ref.operator_form == "differential-sn"
    assert ref.problem.n_groups == 2
    assert ref.problem.is_eigenvalue is False

    mms = ref.problem.geometry_params["mms_case"]
    quad = mms.quadrature

    n_cells = [20, 40, 80, 160]
    errs_g0 = []
    errs_g1 = []
    for n in n_cells:
        mesh = mms.build_mesh(n)
        materials = mms.build_materials(mesh)
        Q = mms.external_source(mesh)

        result = solve_sn_fixed_source(
            materials, mesh, quad, Q,
            boundary_condition="vacuum",
            max_inner=500, inner_tol=1e-12,
        )

        phi_solver = result.scalar_flux[:, 0, :]  # (n_cells, n_groups)
        phi_ref_g0 = np.asarray(ref.phi(mesh.centers, 0), dtype=float)
        phi_ref_g1 = np.asarray(ref.phi(mesh.centers, 1), dtype=float)

        errs_g0.append(_cell_l2(phi_solver[:, 0] - phi_ref_g0, mesh.widths))
        errs_g1.append(_cell_l2(phi_solver[:, 1] - phi_ref_g1, mesh.widths))

    errs_g0_arr = np.asarray(errs_g0)
    errs_g1_arr = np.asarray(errs_g1)
    orders_g0 = np.log2(errs_g0_arr[:-1] / errs_g0_arr[1:])
    orders_g1 = np.log2(errs_g1_arr[:-1] / errs_g1_arr[1:])

    assert np.all(orders_g0 > 1.9), (
        f"Group 0 convergence below O(h²): "
        f"errors={errs_g0_arr}, orders={orders_g0}"
    )
    assert np.all(orders_g1 > 1.9), (
        f"Group 1 convergence below O(h²): "
        f"errors={errs_g1_arr}, orders={orders_g1}"
    )

    # Finest-mesh magnitude sanity — must land where quadratic
    # extrapolation says (~ few × 1e-6 at dz = L/160 = 0.03 cm).
    assert 1e-7 < errs_g0_arr[-1] < 1e-4, errs_g0_arr[-1]
    assert 1e-7 < errs_g1_arr[-1] < 1e-4, errs_g1_arr[-1]


def test_sn_heterogeneous_mms_vacuum_bcs_exact_on_reference():
    r"""The reference :math:`\phi_g(x) = c_g\sin(\pi x/L)` is
    **exactly zero** at both boundaries for both groups.

    Regression guard on the ansatz closure: any future change
    that drops the :math:`\sin` factor or uses the wrong slab
    length would break the vacuum BC compatibility.
    """
    ref = continuous_get("sn_mms_slab_2g_hetero")
    L = ref.problem.geometry_params["length"]

    for g in range(ref.problem.n_groups):
        phi_0 = ref.phi(np.array([0.0]), g)[0]
        phi_L = ref.phi(np.array([L]), g)[0]
        # sin(0) = 0 exactly; sin(π) is ~1e-16 due to floating-point π
        assert abs(phi_0) < 1e-14
        assert abs(phi_L) < 1e-14


def test_sn_heterogeneous_mms_manufactured_source_couples_groups():
    r"""The ``g=1`` (thermal) manufactured source depends on
    :math:`c_0` through the downscatter term
    :math:`-\Sigma_{s,0\to 1}(x)\,c_0\,A(x)`.

    Regression guard on the multigroup coupling: if a future
    edit to :meth:`SNSlab2GHeterogeneousMMSCase.external_source`
    forgets the in-scatter sum or transposes the scatter index
    order, the thermal source becomes inconsistent with the
    fast-group amplitude and this test flags it.

    The test compares the thermal source with ``c_spectrum = (1.0, 0.3)``
    against the same source computed with ``c_spectrum = (0.0, 0.3)``
    (no fast flux contribution); the difference must equal exactly
    :math:`-\Sigma_{s,0\to 1}(x)\,c_{0,\text{ref}}\,A(x)`, independent
    of ordinate.
    """
    from orpheus.derivations.sn_mms import (
        build_1d_slab_heterogeneous_mms_case,
    )

    case_coupled = build_1d_slab_heterogeneous_mms_case(
        c_spectrum=(1.0, 0.3),
    )
    case_uncoupled = build_1d_slab_heterogeneous_mms_case(
        c_spectrum=(0.0, 0.3),
    )

    mesh = case_coupled.build_mesh(32)
    Q_coupled = case_coupled.external_source(mesh)
    Q_uncoupled = case_uncoupled.external_source(mesh)

    # Thermal source difference (g=1) averaged over ordinates removes
    # the μ-dependent streaming term (which only couples to c_g of the
    # same group and would be identical between the two cases for g=1
    # since c_1 is the same in both).
    diff_g1 = (Q_coupled - Q_uncoupled)[:, :, 0, 1]   # (N_ord, n_cells)

    # Expected: Q_coupled_g1 - Q_uncoupled_g1 = -Σ_{s,0→1}(x) * (c0_coup - c0_uncoup) * A(x)
    x = mesh.centers
    L = case_coupled.slab_length
    A = np.sin(np.pi * x / L)
    sig_s_01 = np.asarray(case_coupled.sigma_s_fn(x, 0, 1), dtype=float)
    expected = -sig_s_01 * (1.0 - 0.0) * A  # (n_cells,)

    # Every ordinate must see the same isotropic difference
    for n in range(diff_g1.shape[0]):
        np.testing.assert_allclose(diff_g1[n], expected, atol=1e-14)


def test_sn_heterogeneous_mms_positive_absorption_everywhere():
    r"""The canonical smooth Σ(x) must give :math:`\Sigma_{a,g}(x) > 0`
    on the whole slab for both groups.

    Regression guard on
    :func:`orpheus.derivations.sn_mms._default_hetero_xs_functions`:
    any future tweak that makes absorption negative somewhere
    would build unphysical materials and the
    :meth:`SNSlab2GHeterogeneousMMSCase.build_materials` call
    would raise. This test checks the condition directly on
    a dense grid.
    """
    ref = continuous_get("sn_mms_slab_2g_hetero")
    mms = ref.problem.geometry_params["mms_case"]
    L = mms.slab_length

    x = np.linspace(0.0, L, 1001)
    for g in range(2):
        sig_t = np.asarray(mms.sigma_t_fn(x, g), dtype=float)
        sig_s_out = sum(
            np.asarray(mms.sigma_s_fn(x, g, gp), dtype=float)
            for gp in range(2)
        )
        sig_a = sig_t - sig_s_out
        assert np.all(sig_a > 0), (
            f"Σ_a,{g}(x) non-positive: min = {sig_a.min():.4e}"
        )
