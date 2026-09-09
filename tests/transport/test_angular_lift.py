r"""**G5.8** — ``AngularLift``'s own laws (the intrinsic-properties standard; CS4c step 5).

The lift of an ENERGY binding onto the angular composite is a mathematical
object with defining laws, gated here on its own terms — not through the
solvers that consume it:

1. **Linearity**, with an ACTIVATION leg (`lessons L40c`: a zero morphism
   satisfies every linearity row with both sides structurally zero).
2. **The ℓ = 0 conjugation identity** ``lift(E)(ψ) = R₀ E M₀ ψ / W``: the
   reaction-rate fast path the base runs (``∫ψ dΩ``, then ``E``, then the
   producer-side ``/W``) against the frame form the transpose is spelled
   with (``full_*_kernel.apply(ψ.values) / W``) — two DIFFERENT reduction
   trees over the same factors. `[M]` 2026-09-04 (the verification plan,
   200-seed sweep, GL8 slab / mixture A 2g / 20 cells): the pure ℓ = 0
   lifts — ``F``, and ``S`` at ``L = 0`` — agree **bit-for-bit, 200/200**;
   ``S`` at ``L = 1`` does NOT (0/200, max |Δ| 2.2e-16), because the ℓ ≥ 1
   half is summed in a different order on the two routes. So the identity
   is pinned ``array_equal`` on the BASE's law and at the draw-stable
   absolute band ``max |Δ| ≤ 2.3e-16`` on the subclass sum (never a nulp
   band — `plan-authoring` 2026-08-28: a nulp band on near-zero outputs
   pins a seed). The partition falls exactly along the base/subclass line
   — the measurement that R-1's split is the right one.
3. **The transpose** is the conjugated product's reversal ``/W``:
   Euclidean reciprocity ``⟨T ψ, χ⟩ = ⟨ψ, Tᵀ χ⟩`` on raw arrays.
4. **The datum → energy derivation**: the energy binding is DERIVED from
   the datum on the CODOMAIN's scalar sub-space (F-1), and the role's own
   ``isotropic_binding`` is what the transfer lift derives.
5. The base is ABSTRACT, and its selection refuses a third interior
   (the admission legs G5.3d exercises on the moment sibling live in
   ``tests/sn/operators/test_moment_domain_binding.py``).
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.common.xs_library import get_mixture
from orpheus.geometry import BC, CoordSystem, Mesh1D
from orpheus.numerics.quadrature import Quadrature
from orpheus.numerics.space import FunctionSpace
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.frames.harmonic_frame import HarmonicFrame
from orpheus.transport.full_field import FullField
from orpheus.transport.operators.angular_lift import AngularEnd, AngularLift, MomentEnd
from orpheus.transport.operators.fission import FissionOperator
from orpheus.transport.operators.isotropic_transfer import (
    IsotropicFission,
    IsotropicScattering,
)
from orpheus.transport.operators.n2n import N2NOperator
from orpheus.transport.operators.scattering import ScatteringOperator
from orpheus.transport.source_sinks import AngularSourceSink

pytestmark = pytest.mark.foundation

#: The draw-stable band for the ℓ ≥ 1 sum (`[M]` 200 seeds: max |Δ| 2.2e-16).
_L1_ABS_BAND = 2.3e-16


def _sn() -> SNMesh:
    materials = {0: get_mixture("A", "2g")}
    mesh = Mesh1D(
        edges=np.linspace(0.0, 1.0, 21), mat_ids=np.zeros(20, dtype=int),
        coord=CoordSystem.CARTESIAN, bc_left=BC("vacuum"), bc_right=BC("vacuum"),
    )
    return SNMesh(mesh, Quadrature.gauss_legendre(n_ordinates=8), materials)


def _mat_xs(sn: SNMesh):
    return sn.material_xs_field()


def _state(sn: SNMesh, seed: int) -> FullField:
    rng = np.random.default_rng(seed)
    interior = sn.full_field_space.interior_space
    assert interior is not None
    return FullField(
        interior=AngularFlux(values=rng.standard_normal(interior.shape), space=interior),
        boundary=AngularBoundaryFlux.zeros(sn.angular_trace),
    )


def _cotangent(sn: SNMesh, seed: int) -> FullField:
    rng = np.random.default_rng(seed)
    interior = sn.full_field_space.interior_space
    assert interior is not None
    return FullField(
        interior=AngularSourceSink(values=rng.standard_normal(interior.shape), space=interior),
        boundary=AngularBoundaryFlux.zeros(sn.angular_trace).into_role(
            __import__("orpheus.transport.fields._bases", fromlist=["FieldRole"]).FieldRole.SOURCE_SINK,
            np.zeros(sn.angular_trace.shape),
        ),
    )


def _lifts(sn: SNMesh) -> dict[str, AngularLift]:
    mat_xs = _mat_xs(sn)
    space = sn.full_field_space
    return {
        "F": FissionOperator.from_solver_data(mat_xs=mat_xs, space=space),
        "S_L0": ScatteringOperator.from_solver_data(mat_xs=mat_xs, scattering_order=0, space=space),
        "S_L1": ScatteringOperator.from_solver_data(mat_xs=mat_xs, scattering_order=1, space=space),
        "N2N_L1": N2NOperator.from_solver_data(mat_xs=mat_xs, scattering_order=1, space=space),
    }


def _frame_form(lift: AngularLift):
    return lift.full_fission_kernel if isinstance(lift, FissionOperator) else lift.full_transfer_kernel  # type: ignore[attr-defined]


@pytest.mark.parametrize("key", ["F", "S_L0", "S_L1", "N2N_L1"])
def test_activation_and_linearity(key):
    sn = _sn()
    lift = _lifts(sn)[key]
    a, b = _state(sn, 1), _state(sn, 2)
    out_a = lift.apply(a).interior.values
    if key == "N2N_L1":
        # mixture A has no (n,2n) data: the lift is honestly the zero
        # morphism here, so the activation leg is the fact recorded, and
        # the linearity row below is structurally vacuous for this key.
        assert not out_a.any()
        return
    assert np.abs(out_a).max() > 0.0, "activation: the lift must move a random flux"
    lhs = lift.apply(2.0 * a - 3.0 * b).interior.values
    rhs = 2.0 * out_a - 3.0 * lift.apply(b).interior.values
    np.testing.assert_allclose(lhs, rhs, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("key", ["F", "S_L0"])
def test_the_l0_conjugation_identity_is_bit_exact_on_the_base(key):
    r"""``lift(ψ) == R₀ E M₀ ψ / W`` — the fast path and the frame form agree
    bit-for-bit on the PURE ℓ = 0 lifts (`[M]` 200/200)."""
    sn = _sn()
    lift = _lifts(sn)[key]
    for seed in range(8):
        psi = _state(sn, seed)
        fast = lift.apply(psi).interior.values
        frame = np.asarray(_frame_form(lift).apply(psi.interior.values)) / lift.total_weight
        np.testing.assert_array_equal(fast, frame)


def test_the_l1_sum_agrees_at_the_draw_stable_absolute_band():
    r"""On the ANISOTROPIC subclass the two routes sum ℓ ≥ 1 in different
    orders: not bit-exact (`[M]` 0/200), pinned at the draw-stable absolute
    band. Both facts are the claim — a bit-exact green here would mean the
    ℓ ≥ 1 body is not running."""
    sn = _sn()
    lift = _lifts(sn)["S_L1"]
    assert not lift.is_isotropic
    worst = 0.0
    any_differs = False
    for seed in range(8):
        psi = _state(sn, seed)
        fast = lift.apply(psi).interior.values
        frame = np.asarray(_frame_form(lift).apply(psi.interior.values)) / lift.total_weight
        delta = np.abs(fast - frame).max()
        worst = max(worst, float(delta))
        any_differs |= not np.array_equal(fast, frame)
    assert worst <= _L1_ABS_BAND, f"max |Δ| = {worst:.3e} exceeds the band"
    assert any_differs, "the ℓ ≥ 1 sum came out bit-exact — is the anisotropic body running?"


@pytest.mark.parametrize("key", ["F", "S_L0", "S_L1"])
def test_transpose_reciprocity_on_raw_arrays(key):
    sn = _sn()
    lift = _lifts(sn)[key]
    psi, chi = _state(sn, 5), _cotangent(sn, 6)
    lhs = float(np.vdot(lift.apply(psi).interior.values, chi.interior.values))
    rhs = float(np.vdot(psi.interior.values, lift.apply_transpose(chi).interior.values))
    assert lhs != 0.0
    assert abs(lhs - rhs) <= 1e-13 * abs(lhs), (lhs, rhs)
    assert not lift.apply_transpose(chi).boundary.values.any()


def test_the_energy_binding_is_derived_from_the_datum_on_the_codomain_scalar_subspace():
    sn = _sn()
    lifts = _lifts(sn)
    interior = sn.full_field_space.interior_space
    assert interior is not None and interior.axes is not None
    scalar = FunctionSpace.of_axes(*interior.axes[1:])
    F = lifts["F"]
    assert isinstance(F.isotropic_energy, IsotropicFission)
    assert F.isotropic_energy.fission is F.fission
    assert F.isotropic_energy.domain == scalar and F.isotropic_energy.codomain == scalar
    assert F.isotropic_energy is F.isotropic_energy  # cached once
    S = lifts["S_L1"]
    assert type(S).isotropic_binding is IsotropicScattering
    assert type(S.isotropic_energy) is IsotropicScattering
    assert S.isotropic_energy.transfer.order == 0  # the P0 head, nothing richer
    assert S.isotropic_energy.domain == scalar
    assert F.frame is S.frame.__class__.for_space(interior, 0)  # the interned hub frame
    assert F.total_weight == S.total_weight == 2.0


def test_the_base_is_abstract_and_refuses_a_third_interior():
    sn = _sn()
    interior = sn.full_field_space.interior_space
    assert interior is not None
    frame = HarmonicFrame.for_space(interior, 0)
    space = sn.full_field_space
    with pytest.raises(TypeError, match="abstract"):
        AngularLift(  # type: ignore[abstract]
            flux_analysis=frame.flux_analysis_on(interior),
            source_reconstruction=frame.source_reconstruction_on(interior),
            domain=space, codomain=space,
        )
    # a THIRD interior: a composite whose bulk is neither face end
    from orpheus.numerics.spaces.full_field_space import FullFieldSpace
    from orpheus.transport.material_field import FissionMaterialField

    other = FullFieldSpace.from_blocks(
        FunctionSpace(name="third", shape=tuple(interior.shape)), sn.angular_trace,
    )
    with pytest.raises(TypeError, match="neither end of the analysis face"):
        FissionOperator(
            FissionMaterialField.from_material_xs(_mat_xs(sn)),
            flux_analysis=frame.flux_analysis_on(interior),
            source_reconstruction=frame.source_reconstruction_on(interior),
            domain=other, codomain=space,
        )


def test_the_lift_population_is_the_two_cores_and_their_roles():
    """G5.4b's companion at the base: the lift's subclass population."""
    def walk(c):
        direct = set(c.__subclasses__())
        return direct.union(*(walk(s) for s in direct))

    names = {c.__name__ for c in walk(AngularLift)}
    assert names == {"TransferOperator", "FissionOperator", "ScatteringOperator", "N2NOperator"}, names


# ═══════════════════════════════════════════════════════════════════════
# The elegance review round (2026-09-05) — witnesses for S1/S2/S3/S4/S6
# ═══════════════════════════════════════════════════════════════════════


def test_the_route_map_is_keyed_on_the_exported_end_classes():
    r"""**S1** — the ℓ ≥ 1 route is selected by the END CLASS (a dict keyed
    on ``AngularEnd`` / ``MomentEnd``), never by a string: on the angular
    end the cached-kernel route, on the moment sibling the typed route, and
    the two ends are the package's public vocabulary. (`[M]` the first
    spelling compared ``_end.__name__`` to a literal — a rename would have
    routed the moment sibling through the angular body, bit-identically,
    and silently zeroed the R-5 typed route's traffic.)"""
    from orpheus.transport.operators import AngularEnd as A2, MomentEnd as M2

    assert A2 is AngularEnd and M2 is MomentEnd
    sn = _sn()
    S = _lifts(sn)["S_L1"]
    assert S._end is AngularEnd
    assert S._redistribution == S._redistribute_ordinates
    S_w = S.on_moment_domain()
    assert S_w._end is MomentEnd
    assert S_w._redistribution == S_w._redistribute_moments
    assert _lifts(sn)["S_L0"]._redistribution is None


def test_the_operand_role_is_admitted_not_cast():
    r"""**S4** — space does not determine role: a source/sink on the SAME
    interior space as the flux is refused by the verb, naming the operator
    and the leaf the body reads (not an ``AttributeError`` from inside the
    body)."""
    sn = _sn()
    S = _lifts(sn)["S_L1"]
    interior = sn.full_field_space.interior_space
    assert interior is not None
    wrong_role = FullField(
        interior=AngularSourceSink(values=np.ones(interior.shape), space=interior),
        boundary=AngularBoundaryFlux.zeros(sn.angular_trace),
    )
    with pytest.raises(TypeError, match=r"ScatteringOperator.*AngularFlux interior"):
        S.apply(wrong_role)
    # the transpose reads VALUES: any angular-family cotangent is admitted
    assert not S.apply_transpose(wrong_role).boundary.values.any()


def test_a_composite_with_a_foreign_trace_is_refused():
    r"""**S2** — the admission reads BOTH blocks: an operator whose bound
    end carries a different trace refuses the mesh's own composite, naming
    the trace, instead of echoing the operand's trace back as its own."""
    from dataclasses import replace

    from orpheus.numerics.spaces.full_field_space import FullFieldSpace

    sn = _sn()
    S = _lifts(sn)["S_L0"]
    interior = sn.full_field_space.interior_space
    assert interior is not None
    foreign = FullFieldSpace.from_blocks(
        interior, FunctionSpace(name="foreign-trace", shape=tuple(sn.angular_trace.shape)),
    )
    S_foreign = replace(S, domain=foreign, codomain=foreign)
    with pytest.raises(TypeError, match="trace"):
        S_foreign.apply(_state(sn, 21))


def test_mixed_frame_faces_are_refused_at_construction():
    r"""**S3** — the two faces must MEET in the middle: an analysis face at
    ``L = 1`` with a reconstruction face at ``L = 0`` (two mints of one
    recipe at two orders — the defect #426 step 2 repaired) is a
    construction refusal, not a binding whose moment route dies later."""
    from orpheus.transport.material_field import FissionMaterialField

    sn = _sn()
    interior = sn.full_field_space.interior_space
    assert interior is not None
    space = sn.full_field_space
    with pytest.raises(TypeError, match="do not meet"):
        FissionOperator(
            FissionMaterialField.from_material_xs(_mat_xs(sn)),
            flux_analysis=HarmonicFrame.for_space(interior, 1).flux_analysis_on(interior),
            source_reconstruction=HarmonicFrame.for_space(interior, 0).source_reconstruction_on(interior),
            domain=space, codomain=space,
        )


def test_the_scalar_subspace_is_the_codomains_memoised_angular_marginal():
    r"""**S6** — the energy binding lives on the SAME object the angular
    integral rides: ``retraction("angular").codomain`` (memoised on the
    space), so the ℓ = 0 emission's space and the energy binding's domain
    agree by identity, not by content."""
    sn = _sn()
    S = _lifts(sn)["S_L0"]
    interior = sn.full_field_space.interior_space
    assert interior is not None
    marginal = interior.retraction("angular").codomain
    assert S._scalar_interior_space is marginal
    assert S.isotropic_energy.domain is marginal
    phi = _state(sn, 3).interior.integrate_angular()
    assert phi.space is marginal
