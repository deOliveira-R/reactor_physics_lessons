r"""L1 verification of the homogeneous solver against Phase-0 continuous references.

This is the **first** consumer test of the Phase-0
:class:`~orpheus.derivations.ContinuousReferenceSolution` contract.
It pulls the three homogeneous references (``homo_1eg``,
``homo_2eg``, ``homo_4eg``) from the continuous registry and
asserts that :func:`orpheus.homogeneous.solver.solve_homogeneous_infinite`
reproduces both the eigenvalue **and** the multigroup flux spectrum
to machine precision.

Why this test matters beyond the legacy ``test_homogeneous.py``:

- The legacy test checks only :math:`k_{\text{eff}}`. A solver
  bug that happens to give the right :math:`k_{\text{eff}}` but
  the wrong flux spectrum (e.g. from a swapped scatter matrix
  transpose that preserves the trace but flips the eigenvector)
  would pass the legacy test and be missed.
- The new test checks the :math:`\ell^{2}`-normalised spectrum
  against the continuous reference. The two checks together —
  eigenvalue + eigenvector — are independent proofs of the
  multigroup-matrix chain.
- It also exercises the
  :meth:`~orpheus.derivations.ContinuousReferenceSolution.phi_on_mesh`
  convenience path on a degenerate (single-point) "mesh" so the
  Phase-1.1 API surface is covered before the harder Phase-1.2
  spatial retrofits land.

See :doc:`/verification/reference_solutions` for the contract and
the full verification-campaign phasing.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations import continuous_get
from orpheus.homogeneous.solver import solve_homogeneous_infinite


# Every test in this file exercises the multigroup matrix chain
# against the Phase-0 continuous reference. These labels mirror the
# legacy homogeneous test file so Nexus sees both pointing at the
# same equation suite.
pytestmark = [pytest.mark.l1, pytest.mark.verifies(
    "one-group-kinf",
    "inf-hom-balance",
    "matrix-eigenvalue",
    "removal-matrix",
    "fission-matrix",
    "mg-balance",
    "two-group-A",
    "two-group-F",
    "two-group-Ainv",
    "two-group-M",
)]


@pytest.mark.parametrize("case_name, ng", [
    ("homo_1eg", 1),
    ("homo_2eg", 2),
    ("homo_4eg", 4),
])
def test_solver_matches_continuous_reference_eigenvalue(case_name: str, ng: int):
    r"""The homogeneous solver must reproduce the reference :math:`k_\infty`
    to machine precision.

    Matches the legacy ``test_kinf_exact`` tolerance, but pulls the
    eigenvalue from the continuous-reference path to exercise the
    Phase-0 registry and prove the retrofit is consistent with
    the legacy case.
    """
    ref = continuous_get(case_name)
    assert ref.problem.n_groups == ng
    assert ref.operator_form == "homogeneous"
    assert ref.k_eff is not None

    mix = next(iter(ref.problem.materials.values()))
    result = solve_homogeneous_infinite(mix)

    assert abs(result.k_inf - ref.k_eff) < 1e-12, (
        f"k_inf mismatch on {case_name}: "
        f"solver={result.k_inf:.14f} reference={ref.k_eff:.14f}"
    )


@pytest.mark.parametrize("case_name, ng", [
    ("homo_2eg", 2),
    ("homo_4eg", 4),
])
def test_solver_flux_spectrum_matches_reference(case_name: str, ng: int):
    r"""Multigroup flux spectrum must match the reference eigenvector.

    The reference normalises the dominant eigenvector of
    :math:`\mathbf{A}^{-1}\mathbf{F}` to unit :math:`\ell^{2}` norm
    with non-negative components. The solver normalises to a fixed
    production rate. To compare shapes we strip both normalisations
    by rescaling the solver output to unit :math:`\ell^{2}` norm.

    Skipped for ``homo_1eg`` because the 1-group spectrum is
    trivially ``[1.0]`` and the shape assertion degenerates.
    """
    ref = continuous_get(case_name)
    assert ref.problem.n_groups == ng

    mix = next(iter(ref.problem.materials.values()))
    result = solve_homogeneous_infinite(mix)

    phi_solver = np.asarray(result.flux, dtype=float)
    phi_solver /= np.linalg.norm(phi_solver)  # l2-normalise

    phi_ref = np.array([ref.phi(np.array([0.0]), g)[0] for g in range(ng)])
    # Guard: reference must be l2-normalised by construction
    assert abs(np.linalg.norm(phi_ref) - 1.0) < 1e-14

    # Non-negative by construction, but fix any global sign difference
    # (eigenvectors are only defined up to a sign).
    if phi_solver.sum() < 0:
        phi_solver = -phi_solver

    np.testing.assert_allclose(phi_solver, phi_ref, atol=1e-10, rtol=0)


def test_continuous_phi_is_spatially_flat():
    r"""The homogeneous reference :math:`\phi(x, g)` must be spatially
    flat — evaluating it at different ``x`` values must return
    identical numbers.

    Regression guard on the ``phi`` closure: any future retrofit
    that accidentally makes the homogeneous reference depend on
    ``x`` (e.g. by bundling an eigenfunction from a finite-slab
    derivation into the homogeneous module) breaks the operator
    form and this test catches it.
    """
    ref = continuous_get("homo_2eg")

    x1 = np.array([0.0])
    x2 = np.array([1e6])
    x3 = np.linspace(-100.0, 100.0, 17)

    for g in range(ref.problem.n_groups):
        phi1 = ref.phi(x1, g)
        phi2 = ref.phi(x2, g)
        phi3 = ref.phi(x3, g)
        assert phi1.shape == x1.shape
        assert phi2.shape == x2.shape
        assert phi3.shape == x3.shape
        assert np.all(phi1 == phi1[0])
        assert np.all(phi2 == phi1[0])
        assert np.all(phi3 == phi1[0])


class _FakeMesh:
    """Minimal mesh stand-in for the convenience-API test.

    Mirrors the attribute surface ``phi_on_mesh`` and
    ``phi_cell_average`` consume (``centers`` and ``edges``) without
    pulling in the real ``Mesh1D`` constructor. Keeps this test file
    independent of the geometry module.
    """

    def __init__(self, edges: np.ndarray) -> None:
        self.edges = np.asarray(edges, dtype=float)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])


def test_phi_on_mesh_and_cell_average_are_consistent_for_flat_flux():
    r"""For a homogeneous reference, :meth:`phi_on_mesh` and
    :meth:`phi_cell_average` must return the same constant vector.

    Exercises the Phase-0 convenience API on a trivial mesh. This
    is the API surface that later spatial retrofits (Phase 1.2
    SN heterogeneous transfer matrix, Phase 4 Peierls) will lean
    on heavily; locking it in now on a case with a known-flat
    solution means any future bug in the helpers shows up as a
    homogeneous-test regression rather than as a subtle spatial
    bug two phases later.
    """
    ref = continuous_get("homo_2eg")
    mesh = _FakeMesh(edges=np.linspace(0.0, 5.0, 11))

    for g in range(ref.problem.n_groups):
        phi_centers = ref.phi_on_mesh(mesh, g)
        phi_average = ref.phi_cell_average(mesh, g)

        # Both must be length-10 (10 cells in the fake mesh)
        assert phi_centers.shape == (10,)
        assert phi_average.shape == (10,)

        # Both must be constant and equal to phi_ref[g]
        phi_ref_scalar = ref.phi(np.array([0.0]), g)[0]
        np.testing.assert_allclose(phi_centers, phi_ref_scalar, atol=1e-14)
        np.testing.assert_allclose(phi_average, phi_ref_scalar, atol=1e-14)
