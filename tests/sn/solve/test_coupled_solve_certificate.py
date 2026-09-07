r"""Step 5 (R-5.2) — the end-of-solve convergence CERTIFICATE (C2/C3).

The SI stop is the FREE-IDENTITY residual (``r = rhs_{n−1} − rhs_n``,
:class:`~orpheus.numerics.iteration.SourceIteration`) — exact when the
step operator honestly inverts the splitting's ``M``, and STRUCTURALLY
BLIND to an in-``M`` inconsistency (the #282 lag-death class: a
stale/lagged block leaves the iteration's fixed point off the equation
while every increment-family test reports "converged"). The driver-level
certificate (:func:`orpheus.sn.solver._certify_within_group_exit`) is
the ONE honest :func:`~orpheus.sn.solver.evaluate_residual` per solve
that closes exactly that hole.

* **C2** — the exact-M bridge, positive control: a REAL carrying-sphere
  production solve passes the LIVE certificate silently, both drivers
  (the raise-free return IS the pass — the certificate sits inside the
  driver's exit path, pinned by C3's mutated leg reddening through the
  SAME call).
* **C3 (THE HEADLINE)** — the lag-death classifier's teeth: an injected
  in-``M`` lag (the ray march returns a STALE ZERO ψ_B — the #282
  surrogate) leaves the running stop convergent (the identity sees only
  ``N·Δψ``, which contracts) while the certificate raises
  :class:`~orpheus.sn.solver.ConvergenceCertificateError` LOUDLY. The
  asymmetry (stop green / certificate red) IS the classifier's proof —
  without it the ρ-honest stop is a Mode-11 vacuous claim (TA step-5
  memo C3; the in-M defect is off the free identity's call graph).
* Plus the no-claim no-op law (best-effort ``max_iter`` exits stay
  legal).

vv Mode-8: ``np.testing`` / ``pytest.raises`` / ``pytest.fail`` only
(the suite runs under ``python -O``).
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.common.xs_library import get_mixture
from orpheus.geometry import BC, CoordSystem, Mesh1D
from orpheus.numerics.convergence import IterationRecord, StoppingCriterion
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn import solve_sn_fixed_source
from orpheus.sn.coupled_system import build_within_group_system
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.sn.operators.radial_characteristic import (
    RadialCharacteristicOperator,
)
from orpheus.sn.solver import ConvergenceCertificateError, SNSolver
from orpheus.transport.radial_characteristic_field import (
    RadialCharacteristicField,
)

pytestmark = pytest.mark.foundation

_MATERIALS = {0: get_mixture("A", "2g"), 1: get_mixture("B", "2g")}


def _mesh1d() -> Mesh1D:
    return Mesh1D(edges=np.array([0.0, 0.3, 0.8, 1.0]),
                  mat_ids=np.array([0, 1, 0]),
                  bc_left=BC("reflective"), bc_right=BC("vacuum"),
                  coord=CoordSystem.SPHERICAL)


def _sphere() -> SNMesh:
    return SNMesh(
        _mesh1d(), Quadrature.gauss_legendre(n_ordinates=4),
        dict(_MATERIALS),
    )


@pytest.mark.parametrize("inner", ["source_iteration", "krylov"])
def test_c2_production_carrying_solve_passes_the_certificate(inner):
    """C2 — positive control on the LIVE certificate: the carrying-sphere
    production solve exits raise-free through the certified path (C3's
    mutated leg proves the same call site raises when the equation is
    violated), and the converged iterate is finite and typed."""
    sn = _sphere()
    solver = SNSolver(sn, inner_solver=inner)
    solver.solve_fixed_source(
        np.ones((sn.ng, sn.nx)), np.ones((sn.ng, sn.nx)),
    )
    assert solver._inner is not None
    psi_pair = solver._inner.iterate
    if psi_pair is None:
        pytest.fail(f"[{inner}] no converged iterate stored")
    if not np.isfinite(np.asarray(psi_pair.to_flat())).all():
        pytest.fail(f"[{inner}] non-finite converged iterate")


def test_c3_in_m_lag_trips_the_certificate_while_the_stop_stays_green(
        monkeypatch):
    """C3 — THE lag-death classifier proof, both legs (L18 discipline):
    the CONTROL solve converges with the certificate silent; the MUTATED
    solve (a stale zero ψ_B inside M) still satisfies its own running
    stop but the certificate raises the loud lag-death error."""
    sn = _sphere()
    q = np.ones((sn.quad.N, sn.ng, sn.nx))  # per-ordinate (the module entry's contract)

    # ── CONTROL: unmutated — the certificate stays silent.
    solution = solve_sn_fixed_source(
        dict(_MATERIALS), _mesh1d(), sn.quad, q,
        inner_solver="source_iteration",
    )
    if not bool(solution.history.converged):
        pytest.fail("control leg did not converge — fixture drift")

    # ── MUTATED: the march returns a stale ZERO ψ_B (the #282 surrogate:
    # the TRUE q½ is ignored, the bulk consumes a wrong seed, the SI
    # increment still contracts — only the certificate can see it).
    def _stale_ray(self, source):
        del source
        return RadialCharacteristicField.flux_zeros(self._field_space)

    with monkeypatch.context() as m:
        m.setattr(RadialCharacteristicOperator, "solve", _stale_ray)
        with pytest.raises(ConvergenceCertificateError, match="lag-death"):
            solve_sn_fixed_source(
                dict(_MATERIALS), _mesh1d(), sn.quad, q,
                inner_solver="source_iteration",
            )

    # ── And the mutation reverted cleanly (the monkeypatch context):
    again = solve_sn_fixed_source(
        dict(_MATERIALS), _mesh1d(), sn.quad, q,
        inner_solver="source_iteration",
    )
    if not bool(again.history.converged):
        pytest.fail("post-revert leg did not converge — the mutation leaked")


def test_certificate_is_a_noop_without_a_convergence_claim():
    """A ``max_iter``-hit best-effort exit makes NO claim — the
    certificate must not raise (legal non-converged returns stay legal).
    The ψ/q sentinels are ``None``: a no-op that touched them would
    explode, so the silent pass proves the claim-gate short-circuits.

    ⚠ Read this as HALF the contract, not the project's whole position
    on truncation.  The certificate audits a *claim*, so no claim means
    nothing to audit — but since #340 a best-effort exit is legal **and
    audible**: the public entry emits
    :class:`~orpheus.numerics.convergence.ConvergenceWarning`, and the
    caller can read ``solution.history.converged``.  The silence proven
    here belongs to the certificate alone; the entry point is loud.
    That other half is gated in
    ``tests/sn/solve/test_convergence_contract.py`` (deliberately NOT
    duplicated here — one contract, one home).
    """
    from orpheus.sn.solver import _certify_within_group_exit

    sn = _sphere()
    solver = SNSolver(sn)
    system = build_within_group_system(
        sn, solver.mat_xs, scattering_op=solver.scattering_op,
    )
    # Both no-op arms, now stated as RECORDS (#340 N2a — the certificate
    # reads the driver's own record rather than re-deriving the claim from
    # a bare list, so `_claims_convergence` could retire).
    def _record(trajectory: tuple[float, ...], **kw) -> IterationRecord:
        return IterationRecord(
            label="inner(probe)",
            criteria=(
                StoppingCriterion(
                    name="residual", trajectory=trajectory, tolerance=1e-8,
                ),
            ),
            **kw,
        )

    # (a) the level never entered its loop — nothing measured, nothing to
    #     certify.  ⚠ This arm's MEANING changed with the record: the old
    #     predicate read an empty history as "did not converge", which is
    #     wrong for a driver that returned on its initial guess.  It is a
    #     no-op either way here, but for opposite reasons — worth stating,
    #     because the two readings differ wherever a claim IS made.
    _certify_within_group_exit(
        system.loss, None, None,  # type: ignore[arg-type]
        sn_mesh=sn, record=_record(()), where="noop-test",
    )
    # (b) a genuine truncation: measured 1.0 against tol 1e-8, no claim.
    _certify_within_group_exit(
        system.loss, None, None,  # type: ignore[arg-type]
        sn_mesh=sn, record=_record((1.0,), iterations_run=1), where="noop-test",
    )
