r"""L1 MMS (Method of Manufactured Solutions) tests for MOC spatial operator.

Phase-2.2a consumer test.  Verifies that the flat-source MOC spatial
operator converges at :math:`\mathcal{O}(h^{2})` by refining the
FSR mesh (annular subdivisions) while holding track spacing and
angular quadrature fixed.

The MMS sweep uses per-segment manufactured sources along each
characteristic — the streaming residual is angle-dependent and
cannot be represented by an isotropic per-FSR external source.
See :mod:`orpheus.derivations.moc_mms` for the full derivation.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.moc_mms import (
    build_moc_mms_case,
    build_moc_mesh,
    mms_sweep,
)


pytestmark = [pytest.mark.l1, pytest.mark.verifies(
    "characteristic-ode",
    "bar-psi",
    "boyd-eq-45",
    "moc-mms-psi-ref",
    "moc-mms-qext",
)]


def _fsr_l2(err: np.ndarray, areas: np.ndarray) -> float:
    r"""Area-weighted :math:`L^{2}` norm over FSRs:

    .. math::

        \|e\|_{L^{2}} = \sqrt{\sum_i A_i\,e_i^{2}}.
    """
    return float(np.sqrt(np.sum(areas * err * err)))


def test_moc_mms_converges_second_order():
    r"""Flat-source MOC spatial operator shows measured
    :math:`\mathcal{O}(h^{2})` convergence.

    Uses N = 4, 16, 64 equal-area annuli (factor of 4 in N = factor
    of 2 in FSR linear dimension h).  The outermost FSR (square-border
    region) is excluded from the convergence measurement because its
    complex geometry (square minus disk) has a fixed track-sampling
    error that does not converge with FSR refinement.

    Asserts that the inner-annuli convergence order is > 1.9.
    """
    case = build_moc_mms_case()
    n_annuli_list = [4, 16, 64]
    errs = []

    for n_ann in n_annuli_list:
        moc_mesh = build_moc_mesh(case, n_ann)
        phi_solver = mms_sweep(case, moc_mesh)
        phi_ref = case.phi_ref_fsr_average(moc_mesh)
        err_vec = phi_solver - phi_ref
        nr = moc_mesh.n_regions
        inner = slice(0, nr - 1)
        errs.append(_fsr_l2(err_vec[inner], moc_mesh.region_areas[inner]))

    errs_arr = np.asarray(errs)
    orders = np.log2(errs_arr[:-1] / errs_arr[1:])

    assert np.all(orders > 1.9), (
        f"MOC MMS convergence below O(h²): "
        f"errors={errs_arr}, orders={orders}"
    )
    assert errs_arr[-1] < 5e-3, (
        f"Finest-mesh error unexpectedly large: {errs_arr[-1]:.2e}"
    )


def test_moc_mms_reference_is_positive():
    r"""The ansatz :math:`\phi_{\text{ref}} = 1 + 0.3\cos\cos` is
    positive everywhere in the square cell :math:`[0, P]^{2}`.
    """
    case = build_moc_mms_case()
    P = case.pitch
    x = np.linspace(0, P, 51)
    y = np.linspace(0, P, 51)
    X, Y = np.meshgrid(x, y)
    phi = case.phi_ref(X.ravel(), Y.ravel())
    assert np.all(phi > 0), f"phi_ref non-positive: min = {phi.min():.4e}"


def test_moc_mms_volume_conservation():
    r"""The track-geometry volume conservation property

    .. math::

        \sum_{\text{fwd}+\text{bwd}} \omega_a\,\omega_p\,t_s\,\ell_{\text{seg}}
        \;\approx\; A_i

    must hold to within a few percent for the MMS reconstruction to be
    valid.  This test verifies that the identity is satisfied for a
    representative mesh.
    """
    case = build_moc_mms_case()
    moc_mesh = build_moc_mesh(case, n_annuli=8)
    quad = moc_mesh.quad
    nr = moc_mesh.n_regions

    vol_sum = np.zeros(nr)
    for a_idx in range(quad.n_azi):
        ts = moc_mesh.effective_spacing(a_idx)
        omega_a = quad.omega_azi[a_idx]

        for t_idx in moc_mesh.tracks_per_azi[a_idx]:
            track = moc_mesh.tracks[t_idx]
            for p_idx in range(quad.n_polar):
                omega_p = quad.omega_polar[p_idx]
                for seg in track.segments:
                    contribution = omega_a * omega_p * ts * seg.length
                    vol_sum[seg.region_id] += 2.0 * contribution

    areas = moc_mesh.region_areas
    rel_errors = np.abs(vol_sum - areas) / areas

    assert np.all(rel_errors < 0.05), (
        f"Volume conservation violated: max relative error = "
        f"{rel_errors.max():.2e}, per-FSR = {rel_errors}"
    )
