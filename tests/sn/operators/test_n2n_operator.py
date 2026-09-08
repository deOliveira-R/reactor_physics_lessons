r"""Gates for :class:`~orpheus.transport.operators.n2n.N2NOperator` — the
:math:`(n,2n)` role of the transfer binding (CS4c step 3 §14.1 extracted
the term; #426 step 2, 2026-09-04, made it ANISOTROPIC — the same binding
as :math:`S`, at the solve's order, over the channel's own Legendre stack).

The claims, each with its own witness:

* the action IS the frame conjugation at the SOLVE's order —
  ``N2N.apply(ψ) ≡ (1/W)·frame.conjugate(Λ₂ₙ).apply(ψ)`` with ``Λ₂ₙ`` the
  full (``skip_l0=False``) moment transfer over the (n,2n) field at ``L``
  (the P0 half on the reaction-rate fast path, the ℓ ≥ 1 half through
  the frame — the shape ``S`` has always had);
* the ℓ = 1 moment REACHES the action (the activation leg: a P0 twin of
  this operator would leave the difference at exactly 0.0 — the §6c
  red-before, measured on this fixture);
* the F2 ruling as arithmetic — ``N2N.apply(ψ) == 2·S'.apply(ψ)`` with
  ``S'`` the SCATTERING binding over the SAME stack: the two terms differ
  by the yield alone, and the law is bit-exact because scaling by 2 is
  exact in binary floating point;
* the Euclidean transpose closes reciprocity
  ``⟨N2N ψ, χ⟩ = ⟨ψ, N2Nᵀ χ⟩`` on the ℓ ≥ 1 fixture (G2.10 — the
  transpose is the product chain's reversal, now with a per-ℓ middle
  factor);
* the carriers are the core's, PER END (CS4c step 5): this binding
  admits its composite ``FullField`` and nothing else; the trace never
  reaches the bulk emission and the emitted trace is exactly zero; the
  moment iterate is the SIBLING's operand (``on_moment_domain()``, whose
  action is bit-identical to this one fed the corresponding angular
  field); the scalar magnitude is the ENERGY binding's
  (``isotropic_energy``, bare arrays) — the ``ScalarFlux`` arm on the
  angular binding is RETIRED (R-3) and its refusal is pinned;
* admission: a face minted on another quadrature's space is refused; a
  bare space is refused; the two terms share ONE interned frame.

Fixture: asymmetric ``Σ_s0`` + P1 elastic (so the solve runs at L = 1)
and a two-block ``Σ_2n`` stack whose ℓ = 1 block is non-zero and
cross-group (L27 — a group flip, a dropped channel, or a dropped ORDER is
detectable). Entrywise ``|Σ_1| ≤ Σ_0`` (the tape's physics).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from orpheus.geometry import BC, CoordSystem, Mesh1D
from orpheus.numerics.basis.base import TruncatedBasis
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.sn.solver import SNSolver
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.fields.harmonic_moment_flux import HarmonicMomentFlux
from orpheus.transport.fields.scalar_flux import ScalarFlux
from orpheus.transport.frames.harmonic_frame import HarmonicFrame
from orpheus.transport.full_field import FullField
from orpheus.transport.material_field import TransferMaterialField
from orpheus.transport.operators.n2n import N2NOperator
from orpheus.transport.operators.scattering import ScatteringOperator
from orpheus.transport.operators.transfer import LegendreMomentTransfer

from tests.sn._test_helpers import material_xs_from_raw
from tests.sn.operators._composite_operand import (
    bulk_apply,
    transpose_values,
    zero_trace_composite,
)

pytestmark = pytest.mark.foundation

_SIGS = [
    np.array([[0.38, 0.10], [0.05, 0.90]]),   # ℓ=0
    np.array([[0.20, -0.04], [0.02, 0.30]]),  # ℓ=1 — the solve runs at L = 1
]
#: The (n,2n) stack: a non-zero cross-group ℓ = 1 block that is NOT a
#: scaled copy of ℓ = 0 (a moments-off-by-one reads differently).
_SIG2 = [
    np.array([[0.00, 0.03], [0.01, 0.00]]),
    np.array([[0.00, 0.012], [-0.004, 0.00]]),
]
_SIG2_P0_ONLY = [_SIG2[0], np.zeros((2, 2))]
_NX, _NG, _L = 4, 2, 1


def _mesh():
    return Mesh1D(
        edges=np.linspace(0.0, 1.0, _NX + 1),
        mat_ids=np.zeros(_NX, dtype=int),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("vacuum"),
        bc_right=BC("vacuum"),
    )


def _mat_xs(sig_s=_SIGS, sig2=_SIG2):
    return material_xs_from_raw(
        sig_s={0: sig_s}, sig2=None if sig2 is None else {0: sig2},
        cells_by_mat={0: (np.arange(_NX), np.zeros(_NX, dtype=int))},
        ng=_NG, nx=_NX, ny=1,
    )


def _solver(sig_s=_SIGS, sig2=_SIG2, order=_L):
    mat_xs = _mat_xs(sig_s=sig_s, sig2=sig2)
    sn = SNMesh(_mesh(), Quadrature.gauss_legendre(n_ordinates=4), mat_xs.materials)
    # SNMesh re-derives its own mat_xs field; use the SOLVER path so the
    # operator pair is the production mint (injection-consistent).
    return SNSolver(sn, scattering_order=order), sn


def _psi(sn, seed=3):
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.05, 1.0, size=sn.angular_bulk_space.shape)
    return AngularFlux(values=values, space=sn.angular_bulk_space)


class TestTheBindingAtTheSolveOrder:
    def test_apply_equals_the_frame_conjugation_at_the_solve_order(self):
        r"""``N2N.apply(ψ)`` ≡ ``(1/W)·conjugate(Λ₂ₙ).apply(ψ)`` at ``L`` —
        the fast path (P0 lift + frame-conjugated ℓ ≥ 1) IS the full frame
        conjugation of the yield-carrying moment transfer (algebra eager,
        performance lazy). Until 2026-09-04 this row held at ``L = 0``
        only, with the moment operator's ℓ = 0 block alone.

        ⛔ RE-KEYED 2026-09-02 (#429). The moment operator is minted on
        the basis read OFF THE FRAME (the flat Legendre family on this 1-D
        rule): the conjugation ``R ∘ X ∘ M`` is well-posed exactly when
        ``X``'s ends are the frame's own coefficient space.
        """
        solver, sn = _solver()
        n2n = solver.n2n_op
        S = solver.scattering_op
        psi = _psi(sn)
        # CS4c step 5: the gain is composite-bound; the bulk action rides a
        # zero-trace composite (the trace the lift itself emits back).
        got = bulk_apply(n2n, psi).values

        basis = S.frame.basis
        assert isinstance(basis, TruncatedBasis)
        assert basis.L == S.legendre_order == n2n.legendre_order == _L
        moment = LegendreMomentTransfer.on_frame(
            TransferMaterialField.n2n(solver.mat_xs), S.frame, skip_l0=False,
        )
        conjugated = S.frame.conjugate(moment).apply(psi.values)
        np.testing.assert_allclose(
            got, np.asarray(conjugated) / n2n.total_weight,
            rtol=1e-13, atol=1e-16,
            err_msg="the (n,2n) binding drifted from its own frame conjugation",
        )

    def test_the_first_moment_reaches_the_action(self):
        r"""ACTIVATION (vv #19; the §6c red-before): the ℓ = 1 block of the
        (n,2n) stack moves the action. Vary ONE thing — the same stack with
        its ℓ = 1 block zeroed, the same length, the same solve order — so a
        P0 twin of this operator (every arm with ``aniso = None``, the
        shipped model until 2026-09-04) reads exactly 0.0 here."""
        solver, sn = _solver()
        control, control_sn = _solver(sig2=_SIG2_P0_ONLY)
        psi = _psi(sn, 7)
        full = bulk_apply(solver.n2n_op, psi).values
        p0 = bulk_apply(control.n2n_op, _psi(control_sn, 7)).values
        moved = float(np.max(np.abs(full - p0)))
        if moved == 0.0:
            pytest.fail(
                "the ℓ = 1 (n,2n) moment did not reach the action — the "
                "operator is the P0 model (#426)"
            )
        # and the P0 halves agree: the ℓ = 0 lift is untouched by the ℓ = 1 block
        phi = psi.integrate_angular().values
        np.testing.assert_array_equal(
            np.asarray(solver.n2n_op.isotropic_energy.apply(phi)),
            np.asarray(control.n2n_op.isotropic_energy.apply(phi)),
        )

    def test_the_two_terms_differ_by_the_yield_alone(self):
        r"""The F2 ruling, as arithmetic at the operator tier: the (n,2n)
        binding equals TWICE the scattering binding built over the SAME
        stack — forward AND transpose. Scaling by 2 is exact in binary
        floating point (every product and partial sum doubles exactly), so
        the identity is ``array_equal``, not a tolerance; a red here at ULP
        level would mean the two roles no longer share one arithmetic."""
        solver, sn = _solver()
        # the SAME stack as the SCATTERING channel of an otherwise-equal solve
        twin, _ = _solver(sig_s=_SIG2, sig2=None)
        S_prime = twin.scattering_op
        assert isinstance(S_prime, ScatteringOperator)
        n2n = solver.n2n_op
        psi, chi = _psi(sn, 21), _psi(sn, 22)
        np.testing.assert_array_equal(
            bulk_apply(n2n, psi).values,
            2 * bulk_apply(S_prime, psi).values,
        )
        np.testing.assert_array_equal(
            transpose_values(n2n, chi.values),
            2 * transpose_values(S_prime, chi.values),
        )
        assert n2n.transfer.multiplicity == 2 and S_prime.transfer.multiplicity == 1

    def test_euclidean_reciprocity(self):
        r"""``⟨N2N ψ, χ⟩ = ⟨ψ, N2Nᵀ χ⟩`` (full per-ordinate contraction) on
        the ℓ ≥ 1 fixture — the transpose is the product chain's reversal
        with the per-ℓ middle factor (G2.10)."""
        solver, sn = _solver()
        n2n = solver.n2n_op
        psi, chi = _psi(sn, 5), _psi(sn, 6)
        lhs = float(np.sum(bulk_apply(n2n, psi).values * chi.values))
        rhs = float(np.sum(psi.values * transpose_values(n2n, chi.values)))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12)

    def test_transpose_reds_on_group_flip(self):
        r"""Negative leg (vv #11): a hand-flipped transpose breaks
        reciprocity on the asymmetric fixture — the identity has teeth."""
        solver, sn = _solver()
        n2n = solver.n2n_op
        psi, chi = _psi(sn, 5), _psi(sn, 6)
        lhs = float(np.sum(bulk_apply(n2n, psi).values * chi.values))
        # The WRONG transpose: forward applied to χ (un-transposed K).
        wrong = float(np.sum(psi.values * bulk_apply(n2n, chi).values))
        if abs(lhs - wrong) <= 1e-10 * abs(lhs):
            pytest.fail(
                "group-flip control did not move reciprocity — the "
                "fixture cannot see a transposed kernel",
            )


class TestCarrierArms:
    r"""Which carrier each BINDING admits — the step-5 outcome, per end.

    Until 2026-09-04 one angular-bound instance dispatched on the operand's
    class per call (``AngularFlux`` / ``HarmonicMomentFlux`` / ``ScalarFlux``
    / ``FullField`` arms). Now *each binding acts through the body its ends
    select*: this binding admits its composite ``FullField`` and nothing
    else, the moment iterate is the moment SIBLING's operand
    (``on_moment_domain()``), and the scalar magnitude is the ENERGY
    binding's (``isotropic_energy``, plain-bound, bare arrays).
    """

    def test_the_trace_does_not_reach_the_bulk_emission(self):
        r"""⛔ RE-KEYED (CS4c step 5) — was ``test_composite_arm_is_bulk_only``.

        The claim is unchanged (*a collision gain is volumetric*) but its
        spelling had to move: the old row compared the composite arm against
        ``N2N.apply(bulk)``, and a bare bulk field is now refused. The
        equivalent — and strictly stronger, because the old row fed a ZERO
        trace on both sides — is trace-INDEPENDENCE on a NON-ZERO trace:

        * the emitted trace is exactly zero (extension-by-zero), and
        * the emitted interior is bit-identical to the zero-trace composite's,
          i.e. no part of the trace reaches the bulk body.

        A body that read the trace reddens the second leg; the old row could
        not have seen it.
        """
        from orpheus.transport.fields.angular_boundary_flux import (
            AngularBoundaryFlux,
        )

        solver, sn = _solver()
        rng = np.random.default_rng(9)
        trace = AngularBoundaryFlux.zeros(sn.angular_trace)
        loud = type(trace)(
            values=rng.uniform(0.5, 1.5, size=np.asarray(trace.values).shape),
            space=trace.space,
        )
        if not float(np.max(np.abs(np.asarray(loud.values)))) > 0.0:
            pytest.fail("the trace probe is zero — the independence leg is vacuous")

        psi = _psi(sn, 9)
        out = solver.n2n_op.apply(FullField(interior=psi, boundary=loud))
        assert isinstance(out, FullField)
        np.testing.assert_array_equal(out.boundary.values, 0.0)
        np.testing.assert_array_equal(
            out.interior.values, bulk_apply(solver.n2n_op, psi).values,
            err_msg="the (n,2n) bulk emission moved with the TRACE — a "
                    "volumetric gain must not read the boundary block.",
        )

    def test_moment_arm_equals_angular_arm_on_projected_flux(self):
        r"""**G5.3b (N2N)** — ``N2N_w.apply(Mψ ⊕ 0) == N2N.apply(ψ ⊕ 0)``, 0 ULP.

        ⛔ RE-KEYED (CS4c step 5). The claim is the one this row always made
        — *the moments ARE* :math:`M\psi`, *so windowing loses nothing* — but
        the moment iterate is no longer handed to the ANGULAR binding (the
        shipped non-endomorphism, `[M]` 143 feeds per windowed solve). It goes
        to the SIBLING bound on the moment end, whose body is selected at
        construction: :math:`\Lambda` then the source-reconstruction FACE,
        with :math:`M` skipped because the operand already carries it.

        The tolerance TIGHTENS: `[M]` 2026-09-04, this fixture, 200 seeds —
        **200/200 ``array_equal``, max |Δ| = 0.0**. Both ends share
        :math:`\Lambda` and the frame's :math:`R`, and their ℓ = 0 halves are
        the same scalar flux, so bit-identity is a property of the FIXTURE,
        not of one draw (`vv` anti-#31).
        """
        solver, sn = _solver()
        n2n = solver.n2n_op
        psi = _psi(sn, 11)
        moments = n2n.flux_analysis.apply(psi)          # M·ψ, TYPED
        assert isinstance(moments, HarmonicMomentFlux)
        np.testing.assert_array_equal(
            n2n.on_moment_domain().apply(
                zero_trace_composite(moments, n2n.domain.trace_space),
            ).interior.values,
            bulk_apply(n2n, psi).values,
            err_msg="the moment-domain sibling drifted from the angular "
                    "binding fed the corresponding angular field.",
        )

    def test_the_scalar_magnitude_is_the_energy_bindings_and_it_is_refused_here(self):
        r"""⛔ **THE ARM IS RETIRED** (CS4c step 5, R-3) — was
        ``test_scalar_flux_arm_is_the_p0_emission_in_iso_magnitude``.

        The old row pinned a ``ScalarFlux`` arm on the ANGULAR binding that
        returned the P0 emission in iso scalar magnitude. That arm is gone:
        the scalar magnitude is the ENERGY binding's action
        (``N2N.isotropic_energy``, plain-bound, bare arrays in and out), and
        the angular binding admits its composite alone. Re-pointing the old
        row to ``isotropic_energy.apply(phi.values)`` on both sides would
        make it a tautology, so it is split into the two claims that survive:

        * the angular binding REFUSES the scalar carrier, naming itself;
        * the energy binding's emission IS the lift's ℓ = 0 half — checked on
          an ISOTROPIC ψ, where the (n,2n) stack's ℓ ≥ 1 block contributes
          exactly nothing, so the whole composite emission must be the
          broadcast ``E φ / W``. `[M]` bit-exact on this fixture.
        """
        solver, sn = _solver()
        rng = np.random.default_rng(31)
        phi = ScalarFlux(
            values=rng.uniform(0.1, 1.0, size=sn.bulk_space.shape),
            space=sn.bulk_space,
        )
        with pytest.raises(TypeError, match="N2NOperator: this binding acts"):
            solver.n2n_op.apply(phi)

        # The ℓ=0 half, isolated: an isotropic ψ has no ℓ ≥ 1 moments, so the
        # aniso route emits exactly zero and the lift is E φ / W broadcast.
        flat = np.empty(sn.angular_bulk_space.shape)
        flat[:] = rng.uniform(0.1, 1.0, size=flat.shape[1:])[None]
        psi_iso = AngularFlux(values=flat, space=sn.angular_bulk_space)
        emitted = bulk_apply(solver.n2n_op, psi_iso).values
        iso = (
            np.asarray(
                solver.n2n_op.isotropic_energy.apply(
                    psi_iso.integrate_angular().values,
                ),
            )
            / solver.n2n_op.total_weight
        )
        np.testing.assert_array_equal(
            emitted, np.broadcast_to(iso[None], np.shape(emitted)),
            err_msg="on an isotropic flux the (n,2n) lift must be exactly the "
                    "energy binding's emission per ordinate (E φ / W).",
        )

    def test_unknown_carrier_refused_with_named_operator(self):
        """A carrier that is neither the composite nor anything else the
        binding knows: refused, naming THIS operator and the carrier it
        wants (the one-body admission's message, ``lift.admit_composite``)."""
        solver, _ = _solver()
        with pytest.raises(TypeError, match="N2NOperator: this binding acts"):
            solver.n2n_op.apply(cast(Any, object()))


class TestAdmission:
    def test_face_from_another_quadrature_refused(self):
        """The core's face-agreement guard (the F guard's shape): a face
        minted on a DIFFERENT posed space than this binding's interior is
        refused at construction — the CS4c step-4 harmonization kept the
        frame's ordinates as OPERATIVE state, and this is what enforces it."""
        solver, _ = _solver()
        other = SNMesh(_mesh(), Quadrature.gauss_legendre(n_ordinates=2), solver.mat_xs.materials)
        other_interior = other.full_field_space.interior_space
        assert other_interior is not None
        wrong_face = HarmonicFrame.for_space(other_interior, _L).flux_analysis_on(other_interior)
        with pytest.raises(TypeError, match="mint the faces"):
            replace(solver.n2n_op, flux_analysis=wrong_face)

    def test_reconstruction_face_from_another_quadrature_refused(self):
        """The other half of the face-binding admission (the elegance review's
        S1, measured red-before): a source-reconstruction face minted on an
        8-ordinate space over a 4-ordinate interior was ACCEPTED and the
        windowed arm returned an ``(8, 2, 4)`` source — no raise, because the
        scalar P0 half broadcasts against the wrong ordinate count inside the
        shared combine. Both faces are admitted now."""
        solver, _ = _solver()
        other = SNMesh(_mesh(), Quadrature.gauss_legendre(n_ordinates=8), solver.mat_xs.materials)
        other_interior = other.full_field_space.interior_space
        assert other_interior is not None
        wrong_face = HarmonicFrame.for_space(other_interior, _L).source_reconstruction_on(other_interior)
        with pytest.raises(TypeError, match="mint the faces"):
            replace(solver.n2n_op, source_reconstruction=wrong_face)

    def test_from_solver_data_refuses_a_bare_space(self):
        solver, _ = _solver()
        from orpheus.numerics.space import FunctionSpace

        with pytest.raises(TypeError, match="composite FullFieldSpace"):
            N2NOperator.from_solver_data(
                mat_xs=solver.mat_xs,
                scattering_order=_L,
                space=FunctionSpace("bare", (2, 4, 1)),
            )

    def test_the_two_terms_share_one_interned_frame(self):
        r"""Both gains mint at the same ``(rule, L)`` through the hub, so
        they carry ONE frame object — one metric for ``− S − N₂ₙ``."""
        solver, _ = _solver()
        if solver.n2n_op.frame is not solver.scattering_op.frame:
            pytest.fail("S and N2N must share the interned frame at the solve's order")

    def test_total_weight_is_the_measure_mass(self):
        solver, sn = _solver()
        np.testing.assert_allclose(
            solver.n2n_op.total_weight, float(sn.quad.weights.sum()),
        )
