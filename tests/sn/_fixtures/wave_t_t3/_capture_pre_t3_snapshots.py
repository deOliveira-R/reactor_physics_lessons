"""Wave T step T.3a — pre-T.3 snapshot capture script.

Captures the numerical output of `ScatteringOperator.apply` (each
dispatch arm), `ScatteringOperator.build_aniso_source` (the verb retired at
#448; the route it wrapped is `_redistribute_ordinates`, which is what the
script calls now so it can still RUN — ⛔ never re-capture: the `.npz` is the
pre-T.3 structurally-independent anchor), and
`TransferMaterialField.moment_source` (né the facade arm) on deterministic
fixtures, BEFORE the T.3 lift rewires `build_aniso_source` to use the
`SumOfTensorProductsOperator` kernel.  The captured arrays are loaded
back by the L1-1..L1-4 and L6-1..L6-2 tests in
`tests/sn/test_scattering_operator.py::TestPreT3RegressionSnapshot`
and `tests/sn/test_material_xs_field.py::TestApplyLegendreScatteringMoments`.

Per the T.3 verification spec
(`.claude/agent-memory/test-architect/wave_t_t3_scattering_verification_spec.md`)
§4 substep T.3a: "Snapshot file exists; tests/sn/_fixtures/wave_t_t3/
schema validated".  Without these snapshots the L1 regression tests
have no pre-T.3 reference to compare against.

Usage
-----

    .venv/bin/python tests/sn/_fixtures/wave_t_t3/_capture_pre_t3_snapshots.py

Writes `pre_t3_snapshots.npz` to the same directory.  Re-run only when
the upstream test fixtures (XS libraries, mesh dimensions, seed)
change AND the change is intentional — otherwise the snapshot is
stale and the L1 regression tests will fail on a moving target.

Determinism
-----------

* All input fluxes use `np.random.default_rng(<fixed_seed>)`.
* All meshes use fixed `nx`, `ny`, `delta`.
* All materials use programmatically-built `make_mixture` with
  hardcoded `sig_*` arrays (no library lookup that could drift).
* All quadratures use fixed Lebedev order.

Output schema
-------------

`pre_t3_snapshots.npz` contains:

    p1_apply_angular_flux        : (N, ng, nx, ny) — apply(AngularFlux).values
    p1_apply_scalar_flux         : (ng, nx, ny)    — apply(ScalarFlux).values
    p1_apply_timed_full_field_bulk     : (N, ng, nx, ny) — .interior.values
    p1_apply_timed_full_field_boundary : (...) flat boundary buffer
    p1_build_aniso_source        : (N, ng, nx, ny) — direct call output
    p1_apply_legendre_scattering_moments : (L+1, 2L+1, ng, nx, ny)
    p3_apply_legendre_scattering_moments : same shape with L=3
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

# Ensure the orpheus package on the path when running standalone.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from orpheus.derivations.common.xs_library import make_mixture
from orpheus.geometry import Mesh2D
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.sn.solver import SNSolver
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.fields.scalar_flux import ScalarFlux
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
from orpheus.transport.timed_full_field import TimedFullField
from orpheus.numerics.basis.spherical_harmonic_basis import SphericalHarmonicBasis
from orpheus.transport.material_field import TransferMaterialField

OUT_FILE = Path(__file__).parent / "pre_t3_snapshots.npz"

# ─────────────────────────────────────────────────────────────────────
# Fixture A — P1 anisotropic with asymmetric SigS + (n,2n)
# (mirrors `solver_2g_p1_n2n` at tests/sn/test_scattering_operator.py:604)
# ─────────────────────────────────────────────────────────────────────


def build_p1_solver() -> SNSolver:
    """2G solver with asymmetric P0 + P1 + non-zero (n,2n).

    Catches ERR-002 (SigS transpose) and other group-coupling drift.
    """
    p0 = np.array([[0.38, 0.10], [0.05, 0.90]])
    p1 = np.array([[0.02, 0.01], [0.00, 0.04]])
    mix = make_mixture(
        sig_t=np.array([0.5, 1.0]),
        sig_c=np.array([0.01, 0.02]),
        sig_f=np.array([0.01, 0.08]),
        nu=np.array([2.5, 2.5]),
        chi=np.array([1.0, 0.0]),
        sig_s=p0,
    )
    mix.SigS = [csr_matrix(p0), csr_matrix(p1)]
    mix.Sig2 = [csr_matrix(np.array([[0.0, 0.03], [0.01, 0.0]]))]

    nx, ny = 3, 2
    mesh = Mesh2D(
        edges_x=np.linspace(0, nx * 0.4, nx + 1),
        edges_y=np.linspace(0, ny * 0.4, ny + 1),
        mat_map=np.zeros((nx, ny), dtype=int),
    )
    quad = Quadrature.lebedev(order=17)
    return SNSolver(SNMesh(mesh, quad, {0: mix}), scattering_order=1)


def build_p3_solver() -> SNSolver:
    """Same materials as P1 fixture, extended with P2 and P3 blocks to
    exercise the higher-order ℓ loop for the L6-2 einsum invariance.
    """
    p0 = np.array([[0.38, 0.10], [0.05, 0.90]])
    p1 = np.array([[0.02, 0.01], [0.00, 0.04]])
    p2 = np.array([[0.005, 0.002], [0.000, 0.010]])
    p3 = np.array([[0.001, 0.0005], [0.000, 0.002]])
    mix = make_mixture(
        sig_t=np.array([0.5, 1.0]),
        sig_c=np.array([0.01, 0.02]),
        sig_f=np.array([0.01, 0.08]),
        nu=np.array([2.5, 2.5]),
        chi=np.array([1.0, 0.0]),
        sig_s=p0,
    )
    mix.SigS = [csr_matrix(p0), csr_matrix(p1), csr_matrix(p2), csr_matrix(p3)]
    mix.Sig2 = [csr_matrix(np.array([[0.0, 0.03], [0.01, 0.0]]))]

    nx, ny = 3, 2
    mesh = Mesh2D(
        edges_x=np.linspace(0, nx * 0.4, nx + 1),
        edges_y=np.linspace(0, ny * 0.4, ny + 1),
        mat_map=np.zeros((nx, ny), dtype=int),
    )
    quad = Quadrature.lebedev(order=17)
    return SNSolver(SNMesh(mesh, quad, {0: mix}), scattering_order=3)


# ─────────────────────────────────────────────────────────────────────
# Capture
# ─────────────────────────────────────────────────────────────────────


def _make_psi(solver: SNSolver, seed: int) -> AngularFlux:
    rng = np.random.default_rng(seed)
    N = solver.quad.N
    ng = solver.ng
    nx, ny = solver.sn_mesh.spatial_shape
    psi_values = rng.uniform(0.05, 1.0, size=(N, ng, nx, ny))
    return AngularFlux(values=psi_values, space=solver.sn_mesh.angular_bulk_space)


def _make_phi(solver: SNSolver, seed: int) -> ScalarFlux:
    rng = np.random.default_rng(seed)
    ng = solver.ng
    nx, ny = solver.sn_mesh.spatial_shape
    phi_values = rng.uniform(0.05, 1.0, size=(ng, nx, ny))
    return ScalarFlux(values=phi_values, space=solver.sn_mesh.bulk_space)


def main() -> None:
    snapshots: dict[str, np.ndarray] = {}

    # ── P1 fixture (the highest-leverage carve target) ────────────────
    p1_solver = build_p1_solver()
    p1_op = p1_solver.scattering_op

    psi = _make_psi(p1_solver, seed=20260530)
    phi = _make_phi(p1_solver, seed=20260530 + 1)
    state = TimedFullField.zeros(interior=AngularFlux, boundary=AngularBoundaryFlux, space=p1_solver.sn_mesh.full_field_space)
    from dataclasses import replace

    bulk_values = psi.values.copy()
    state = replace(state, interior=replace(state.interior, values=bulk_values))

    # L1-1: AngularFlux dispatch arm
    out_af = p1_op.apply(psi)
    snapshots["p1_apply_angular_flux"] = out_af.values.copy()

    # L1-2: ScalarFlux dispatch arm (P0 + n2n only)
    out_sf = p1_op.apply(phi)
    snapshots["p1_apply_scalar_flux"] = out_sf.values.copy()

    # L1-3: TimedFullField dispatch arm (bulk + boundary)
    out_tff = p1_op.apply(state)
    snapshots["p1_apply_timed_full_field_bulk"] = out_tff.interior.values.copy()
    snapshots["p1_apply_timed_full_field_boundary"] = (
        out_tff.boundary.values.copy()
    )

    # L1-4: build_aniso_source direct (the inner R · Λ · M pipeline)
    aniso = p1_op._redistribute_ordinates(psi)
    snapshots["p1_build_aniso_source"] = aniso.values.copy()

    # L6-1: per-material per-ℓ einsum invariance at P1
    moments_p1 = _capture_legendre_moments(p1_solver, psi, L=1)
    snapshots["p1_apply_legendre_scattering_moments"] = moments_p1

    # ── P3 fixture (higher-order ℓ loop coverage) ─────────────────────
    p3_solver = build_p3_solver()
    psi_p3 = _make_psi(p3_solver, seed=20260530 + 2)
    moments_p3 = _capture_legendre_moments(p3_solver, psi_p3, L=3)
    snapshots["p3_apply_legendre_scattering_moments"] = moments_p3

    # ── Persist ────────────────────────────────────────────────────────
    np.savez(OUT_FILE, **snapshots)
    print(f"wrote {OUT_FILE}")
    for k, v in snapshots.items():
        print(f"  {k:50s} shape={v.shape} dtype={v.dtype}")


def _capture_legendre_moments(
    solver: SNSolver, psi: AngularFlux, L: int,
) -> np.ndarray:
    """Apply the per-ℓ moment verb at order L (CS4c 3c: the arm moved to
    TransferMaterialField.moment_source — same einsum leaf; the FROZEN
    snapshot is the anchor, this script only documents provenance)."""
    from orpheus.transport.fields.harmonic_moment_flux import (
        HarmonicMomentFlux,
    )
    from orpheus.transport.operators.transfer import LegendreMomentTransfer

    quad = solver.quad
    moments_values = quad.angular_frame(L).analysis.apply(psi.values)
    moments = HarmonicMomentFlux.from_mesh_and_L(
        moments_values, solver.sn_mesh, L,
    )
    Lam = LegendreMomentTransfer.on_basis(
            TransferMaterialField.scattering(solver.mat_xs), SphericalHarmonicBasis(L=L), skip_l0=False,
        )
    scattered = Lam.apply(moments)
    return scattered.values.copy()


if __name__ == "__main__":
    main()
