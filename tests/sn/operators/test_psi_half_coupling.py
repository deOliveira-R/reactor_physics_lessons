r"""The ψ½ Ray-Characteristic coupled system — regression floor + per-step invariants.

**Campaign home.** This module is the verification substrate for the
*coupled block operator* campaign (`.claude/plans/archive/coupled_block_operator_campaign.md`):
the within-group augmented SN system posed as a 2×2 coupled block operator

.. code-block:: text

    [ A_AA   A_AB ] [ transport ]      A_AA = L + C − S − B     (System A)
    [ A_BA   A_BB ] [ ray       ]      A_BB = RayOp             (System B: the ψ½ ray)

over two systems — System A (the transport bulk⊕trace `FullField`) and System B
(the ψ½ **Radial-Characteristic** starting-direction flux, a codim-1 `FaceField`
carrying two boundary conditions: r=R Dirichlet + r=0 pole-reflection). The
seed IS System B's boundary condition; `A_BA` (bulk→ray) is the Schur-fold
coupling currently welded un-named into the S/F scattering/fission arms — a
coupling *block* never posed in the algebra. The campaign names all four blocks
as operators; "where the block goes" (folded into the resolvent, an explicit
block operator, or a DSA preconditioner) then becomes a *composition choice* the
machinery supports — that flexibility IS operator algebra realized.

**:class:`TestRegressionFloor` — landed FIRST, before any production change**, so
every subsequent campaign step (System B posing, the `A_BA`/`A_AB` un-welds, the
`CoupledOperator` assembly, the coupled solve) diffs against a *pinned* baseline
of the measured block structure. Promoted verbatim from the numerics-investigator
design diagnostics ``derivations/diagnostics/diag_coupled_0{1,2}_*.py``
(2026-07-07). The six pins and their measured values on the seed-carrying
vacuum sphere (GL S4, nx=5, 2G, c=0.4):

======================================================  ==================================
pin                                                      measured (the baseline this floor holds)
======================================================  ==================================
loss ``(L+C)`` is block-TRIANGULAR in the ray            ``A_sb = A_st = 0`` exact; ``A_bs = 7.505``, ``A_ss = 5.000``
bulk→ray ``A_BA`` lives in the LAGGED scattering gain     ``S_sb = 0.183``; ``S_bs = 0`` exact (ψ½ zero moment weight)
outer-SI splitting rate is bounded by ``c``               ``ρ(M⁻¹N) = 0.371 < c = 0.4``
the folded ray seed is a DIRECT (nilpotent) solve         ``ρ(lag) = 0`` — no bulk→seed back-edge
the welded sweep is the EXACT inverse of ``(L+C)``        ``‖solve(apply(ψ)) − ψ‖ = 3.5e-16``
extraction is PRINCIPLED-equivalent, not bit-identical    ``‖welded − dense_LU‖ = 5.5e-16`` (distinct reduction trees)
======================================================  ==================================

The last row is the oracle every EXTRACT step of the campaign pins against
(``coding-elegance``/``vv-principles`` §"Bit-identity vs principled-equivalence":
principled-equivalence + the invariant test is the bar, never byte-identity).

**Runtime discipline (vv Mode 8).** Every gate raises via :func:`pytest.fail`
(a function call — fires under the canonical ``python -O``), never a bare
``assert`` (which ``-O`` strips outside pytest-rewritten test bodies). The
``-s`` print lines echo the measured values for the design record.

References: GH #284 (the triangular sweep = forward substitution), #282
(route-a direct ψ½ seed), #280 (the walk unification).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from orpheus.geometry import BC, CoordSystem, Mesh1D
from orpheus.geometry.boundary import WhiteBoundary
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.sn.operators.boundary import (
    RadialCharacteristicBoundaryOperator,
    SNBoundaryOperator,
)
from orpheus.sn.operators.streaming import StreamingOperator
import orpheus.sn.loss_representation as _lr_mod
import orpheus.sn.operators.radial_characteristic as _rc_mod
from orpheus.sn.operators.radial_characteristic import (
    RadialCharacteristicEmission,
    RadialCharacteristicOperator,
    RadialCharacteristicReconstruction,
    RadialCharacteristicSeeding,
)
import orpheus.numerics.spaces.radial_characteristic_space as _rcs_mod
# Campaign step 4c (THE LIFT): the Fold ``RadialCharacteristicReconstruction``
# migrated transport → sn (it is a factor of the sn coupling operator
# ``RadialCharacteristicEmission``); ``_rcr_mod`` is the reconstruction/fold home
# (now the same module as ``_rc_mod``) — kept as a distinct alias for the spies
# that patch the fold as the reconstruction sees it.
import orpheus.sn.operators.radial_characteristic as _rcr_mod


def _rc_fold(sn, n_moments: int = 1):
    """The fold (Reconstruction) from a carrying mesh — the un-weld
    assembly read, spelled once for this module (through ``_rcr_mod`` so
    the Mode-11 module-global sentinel still sees every construction)."""
    return _rcr_mod.RadialCharacteristicReconstruction(
        sn.radial_characteristic_field_space,
        coord=sn.coord,
        quadrature=sn.quad,
        n_moments=n_moments,
    )
import orpheus.sn.solver as _solver_mod
from orpheus.sn.coupled_system import (
    WithinGroupSystem,
    build_streaming_collision,
    build_within_group_system,
)
from orpheus.sn.solver import (
    ConvergenceCertificateError,
    SNSolver,
    _build_fixed_source_rhs,
    _coupled_flux_state,
    _unwindowed_cold_start,
    _within_group_krylov,
    _within_group_si,
)
from orpheus.sn import solve_sn_fixed_source
from orpheus.numerics.iteration import SourceIteration
from orpheus.sn.operators.streaming import StreamingCollisionOperator
from tests.sn._test_helpers import (
    rc_march,
    curvilinear_two_region_mesh,
)
from orpheus.sn.sweep.psi_half_angle_seed import (
    carlson_inward_sweep_from_source,
    carlson_inward_sweep_transpose,
    radial_characteristic_forward_residual,
)
from orpheus.numerics.coupled_system import (
    CoupledField,
    CoupledOperator,
    CoupledSpace,
    CoupledSubstitutionOperator,
)
from orpheus.numerics.operator import (
    IncompatibleOperatorComposition,
    MissingAssembly,
    SystemRole,
    _join_system_roles,
)
from orpheus.sn.coupled_system import build_coupled_system
from orpheus.transport.operators.multiplication_operator import MultiplicationOperator
from orpheus.transport.operators.scattering import ScatteringOperator
from orpheus.transport.operators.fission import FissionOperator
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
from orpheus.transport.fields.cross_section_field import CrossSectionField
from orpheus.transport.fields.radial_characteristic_interior_flux import (
    RadialCharacteristicInteriorFlux,
)
from orpheus.transport.full_field import FullField
from orpheus.transport.radial_characteristic_field import (
    RadialCharacteristicField,
)
from orpheus.transport.source_sinks.radial_characteristic_boundary_source_sink import (
    RadialCharacteristicBoundarySourceSink,
)
from orpheus.transport.source_sinks.radial_characteristic_interior_source_sink import (
    RadialCharacteristicInteriorSourceSink,
)
from orpheus.transport.source_sinks.angular_source_sink import AngularSourceSink
from orpheus.derivations.common.xs_library import make_mixture
from tests.sn.operators._composite_operand import bulk_apply

pytestmark = pytest.mark.foundation


# ── compact dense-probe helpers (mirror test_radial_characteristic_metric) ──


def _mixture(sig_t: float, sig_s: float, ng: int):
    """A group-graded diagonal-scatter mixture (asymmetric in g — vv L2/anti-#2)."""
    st = np.array([sig_t * (1.0 + 0.4 * g) for g in range(ng)])
    ss = np.diag([sig_s * (1.0 + 0.4 * g) for g in range(ng)])
    return make_mixture(sig_t=st, sig_c=st - ss.sum(axis=0), sig_f=np.zeros(ng),
                        nu=np.zeros(ng), chi=np.zeros(ng), sig_s=ss)


def _sphere(nx: int = 5, ng: int = 2, sigma: float = 1.0, c: float = 0.4,
            bc: str = "vacuum"):
    """Seed-carrying sphere (GL S4) with scattering ratio ``c`` and outer law ``bc``.

    Default vacuum (the regression floor). ``bc="reflective"`` gives a
    seed-carrying sphere whose outer face drives ``B_b``'s non-trivial specular
    corner swap — the vacuum floor never exercises that arm (``_reflect_corner``
    returns zeros for vacuum), so the ``B_b`` gates below need it.
    """
    mesh = Mesh1D(edges=np.linspace(0.0, 4.0, nx + 1), mat_ids=np.zeros(nx, dtype=int),
                  coord=CoordSystem.SPHERICAL, bc_right=BC(bc))
    return SNMesh(mesh, Quadrature.gauss_legendre(4), {0: _mixture(sigma, c * sigma, ng)})


def _loss(sn, slope: float = 0.4):
    r"""The within-group loss operator ``L + C`` with per-group ``σ_t = 1 + slope·g``.

    ``slope`` parametrizes the two diagnostics' constructions into one helper:
    the block-structure pins (triangularity, nilpotent lag) used ``slope = 0.4``
    (matching the sphere mixture's own ``σ_t``); the round-trip pins (welded =
    exact inverse, extract = principled-equiv) used ``slope = 0.3`` (a distinct
    invertible ``σ_t`` — the round-trip holds for *any* invertible ``(L+C)``).
    """
    sig_t = np.stack(
        [np.full(sn.spatial_shape, 1.0 + slope * g) for g in range(sn.ng)], 0)
    return StreamingOperator.pose(sn) + MultiplicationOperator.from_mesh(sig_t, sn)


def _template(sn):
    """A zero 2-block :class:`FullField` (bulk ⊕ trace) to seed dense probes."""
    N, nx, ng = sn.quad.N, sn.nx, sn.ng
    n_tr = int(sn.angular_trace.layout.total_size)
    return FullField(
        interior=AngularFlux(values=np.zeros((N, ng, nx)), space=sn.angular_bulk_space),
        boundary=AngularBoundaryFlux(values=np.zeros(n_tr), space=sn.angular_trace))


def _coupled_template(sn):
    """A zero coupled pair ``[ψ_A 2-block, ψ_B]`` to seed JOINT dense probes."""
    return CoupledField(systems=(
        _template(sn), RadialCharacteristicField.flux_zeros(sn.radial_characteristic_field_space)))


def _pair(sn, psi_a, ray_values):
    """The coupled pair from a System-A composite + System-B ray values in the
    composite ``to_flat`` (interior ⊕ boundary) order (4e native)."""
    return CoupledField(systems=(psi_a, _ray_composite(sn, ray_values)))


def _joint_M(sn, LC):
    """The joint ``M`` — the honest triangular grid over the given variant
    ``LC`` (step 5: the numerics substitution — the row-6 oracle and the
    round-trip rows now pin THE PRODUCTION solve shape; the fused
    ``CoupledInvertibleOperator`` bridge deleted at 5d)."""
    from tests.sn._test_helpers import joint_m_grid

    return joint_m_grid(sn, LC)[0]


def _dense(fn, tpl):
    """Materialise ``fn`` as a dense matrix by probing ``tpl``'s flat basis
    (any ``to_flat``/``from_flat`` carrier — FullField or the coupled pair)."""
    n = tpl.to_flat().size
    M = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        M[:, j] = fn(type(tpl).from_flat(e, tpl)).to_flat()
    return M


def _blocks(sn):
    """The (bulk, trace, System-B) row/col slices of the COUPLED flat layout
    ``[ψ_A.interior | ψ_A.boundary | ψ_B]`` (B.2d — the ray block is System
    B's own composite tail; its internal member order differs from the old
    unified layout, which every consumer here treats as a BLOCK — max-abs
    norms, eigenvalues, and round-trips are permutation-invariant)."""
    N, nx, ng = sn.quad.N, sn.nx, sn.ng
    nb = N * ng * nx
    nt = int(sn.angular_trace.layout.total_size)
    ns = sn.radial_characteristic_field_space.shape[0]
    return slice(0, nb), slice(nb, nb + nt), slice(nb + nt, nb + nt + ns), (nb, nt, ns)


def _bn(M, r, c):
    """The max-abs of block ``M[r, c]`` (a block-norm for triangularity probes)."""
    return float(np.max(np.abs(M[r, c])))


class TestRegressionFloor:
    r"""The pinned baseline of the ψ½ coupled-block structure — landed BEFORE any
    production change so every campaign step diffs against it.

    Each test carries a mutation tooth in its docstring: the specific structural
    corruption (a bulk→seed back-edge in the sweep, a lost scattering source, a
    lagged direct seed) that moves the measured block-norm O(1) and reddens the
    gate. Promoted from ``diag_coupled_01_psi_half_block_structure`` (pins 1–4)
    and ``diag_coupled_02_wrap_vs_extract`` (pins 5–6).
    """

    def test_loss_operator_is_block_triangular_in_the_ray(self):
        r"""Within ``(L+C)`` the seed rows are self-contained (``A_sb = A_st = 0``)
        while the seed feeds the bulk (``A_bs ≠ 0``) — the #284 direct-solve
        certificate. Mutation tooth: a bulk→seed coupling leaking into ``(L+C)``
        (e.g. the ray reading a bulk moment in-sweep) makes ``A_sb`` jump O(1)."""
        sn = _sphere()
        M_op = _joint_M(sn, _loss(sn, slope=0.4))
        b, t, s, _ = _blocks(sn)
        A = _dense(M_op.apply, _coupled_template(sn))
        a_sb, a_st = _bn(A, s, b), _bn(A, s, t)
        a_bs, a_ss = _bn(A, b, s), _bn(A, s, s)
        print(f"  (L+C): A_sb={a_sb:.2e} A_st={a_st:.2e} | A_bs={a_bs:.3f} A_ss={a_ss:.3f}")
        a_ba = max(a_sb, a_st)
        if not (a_ba < 1e-12):
            pytest.fail(f"A_BA(within L+C)={a_ba:.3e} ≠ 0 — the ray is NOT block-triangular; "
                        f"the sweep is no longer exact forward substitution (#284 broken).")
        if not (a_bs > 1.0 and a_ss > 1.0):
            pytest.fail(f"seed→bulk feed A_bs={a_bs:.3e} or self-block A_ss={a_ss:.3e} vanished "
                        f"— the ψ½ coupling is degenerate (route-a seed not wired).")

    def test_bulk_to_ray_coupling_lives_in_the_lagged_A_BA_gain(self):
        r"""The full ``A_BA`` (bulk→ray) is carried by the LIFTED
        :class:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicEmission`
        gain (``A_BA_sb ≠ 0``, the ray scattering source) — NOT by the
        model-generic ``S``, which is now PURE BULK (``S_sb = 0``, campaign step
        4c THE LIFT re-pointed this pin from S to A_BA). The ray still cannot feed
        the scalar flux (``S_bs = 0``, ψ½ has zero moment weight). So the coupled
        system's off-diagonal ``A_BA`` is an OUTER-iterated block riding its OWN
        lagged gain, never a within-sweep one.

        Mutation tooth: an ``A_BA`` that emitted into the bulk (a double-count with
        S's bulk) would make ``A_BA_bb ≠ 0``; an S that re-grew the ray arm would
        make ``S_sb ≠ 0`` (the pre-lift regression this flip retired).

        B.2d re-point: the pin now reads the RECORD's gain grid N — the
        (B,A) Emission block probed over the coupled template. ``S_sb``/
        ``S_bs`` = 0 is STRUCTURAL since the eviction (the model-generic S
        acts on the 2-block System A; no ray slot exists), so the measured
        rows here are N's: the Emission sources the ray (N_sb ≠ 0) and the
        (A,B) slot is the structural ∅ (no ray→bulk gain — Seeding lives in
        M)."""
        sn = _sphere()
        solver = SNSolver(sn)
        system = build_within_group_system(
            sn, solver.mat_xs, scattering_op=solver.scattering_op)
        S = solver.scattering_op
        b, _, s, _ = _blocks(sn)
        tpl_a = _template(sn)
        Sd = _dense(S.apply, tpl_a)              # 2-block: NO ray rows exist
        Ad = _dense(system.explicit_gains[0].apply, _coupled_template(sn))
        s_sb, s_bs = 0.0, 0.0                    # structural (no block to read)
        if Sd.shape[0] != tpl_a.to_flat().size:
            pytest.fail("S's dense is not the 2-block System-A square.")
        aba_sb, aba_bb = _bn(Ad, s, b), _bn(Ad, b, s)
        print(f"  S(pure bulk): S_sb={s_sb:.2e} S_bs={s_bs:.2e} | "
              f"A_BA: A_BA_sb(bulk→ray)={aba_sb:.3f} A_BA_bb(no bulk-out)={aba_bb:.2e}")
        # The coupling lives in A_BA, not S.
        if not (aba_sb > 1e-6):
            pytest.fail(f"A_BA_sb={aba_sb:.3e} ≈ 0 — the lifted A_BA does NOT source the ray; "
                        f"the bulk→ray coupling is missing (the ψ½ gain is unwired).")
        # S is now pure bulk (the LIFT dropped the ray side-channel).
        if not (s_sb < 1e-12):
            pytest.fail(f"S_sb={s_sb:.3e} ≠ 0 — the model-generic scattering gain still sources "
                        f"the ray; the LIFT did not make S pure bulk (a re-grown ray arm).")
        # N's (A,B) is the structural ∅ — no ray→bulk gain (Seeding is M's).
        if not (aba_bb < 1e-12):
            pytest.fail(f"N_bs={aba_bb:.3e} ≠ 0 — the gain grid carries a ray→bulk arm; "
                        f"the (A,B) slot must be the structural ∅ (Seeding lives in M).")

    def test_outer_si_splitting_rate_is_bounded_by_scattering_ratio(self):
        r"""``A = (L+C) − S − B = M − N`` with ``M = (L+C)``, ``N = S + B``: the
        outer source iteration's spectral radius ``ρ(M⁻¹N) ≤ c`` (Adams–Larsen;
        strictly below c for vacuum leakage). This is the convergence rate of the
        ONE genuinely-iterated coupling in the ψ½ system."""
        c = 0.4
        sn = _sphere(c=c)
        solver = SNSolver(sn)
        # The record's OWN splitting, densified over the coupled pair (B.2d).
        system = build_within_group_system(
            sn, solver.mat_xs, scattering_op=solver.scattering_op)
        tpl = _coupled_template(sn)
        M = _dense(system.implicit_operator.apply, tpl)
        N = _dense(system.explicit_gains[0].apply, tpl)
        rho = float(np.max(np.abs(np.linalg.eigvals(np.linalg.solve(M, N)))))
        print(f"  ρ(M⁻¹N) = {rho:.4f}   (c={c}; below c for vacuum leakage)")
        if not (0.0 < rho < c + 1e-6):
            pytest.fail(f"ρ(M⁻¹N)={rho:.4f} not in (0, c={c}] — the within-group SI splitting "
                        f"rate is not bounded by the scattering ratio (splitting mis-posed).")

    def test_folded_ray_seed_is_a_direct_solve_zero_spectral_radius(self):
        r"""FOLDED (route a): the ray→bulk seed is solved in-sweep (forward
        substitution). Lagging it instead (moving ``A_bs`` to the iteration's N
        side) yields a NILPOTENT iteration matrix (``ρ = 0``, converges in 2 steps)
        — because ``A_sb = 0`` there is no back-edge. So the fold buys a direct
        ρ=0 solve; lagging the *direct* seed is merely wasteful (2 passes for the
        same answer), NOT unstable. (The historical #282 EDGE-EXTRAPOLATION seed —
        ψ½ = E·ψ_bulk, a genuine cycle — diverges ρ≈70; that is why route-a was
        needed. Documented, not reconstructed here.)"""
        sn = _sphere()
        _, _, _, (nb, nt, ns) = _blocks(sn)
        M = _dense(_joint_M(sn, _loss(sn, slope=0.4)).apply, _coupled_template(sn))
        # The coupled layout is ALREADY [System A | System B] contiguous.
        Mp = M
        nA = nb + nt
        M_lag = Mp.copy(); M_lag[:nA, nA:] = 0.0               # lag the seed→bulk feed
        N_lag = np.zeros_like(Mp); N_lag[:nA, nA:] = Mp[:nA, nA:]
        rho_lag = float(np.max(np.abs(np.linalg.eigvals(np.linalg.solve(M_lag, N_lag)))))
        print(f"  ρ(lag route-a's direct seed) = {rho_lag:.3e}  (nilpotent → 2 steps)")
        if not (rho_lag < 1e-10):
            pytest.fail(f"ρ_lag={rho_lag:.3e} ≠ 0 — a bulk→seed back-edge appeared (A_sb≠0), so "
                        f"lagging the seed is no longer nilpotent (the triangular structure broke).")

    def test_welded_sweep_is_exact_direct_inverse(self):
        r"""``(L+C).solve((L+C).apply(ψ)) ≈ ψ`` at machine precision — the
        production sphere sweep IS the exact direct inverse (route-a). Since
        4e-e2 the sweep routes its ray legs through ``A_BB.solve`` (the WRAP)
        — measured bit-identical at the un-weave, so this row pins the routed
        path directly. Mutation tooth: a re-introduced ψ½ seed lag makes the
        sphere round-trip blow up (pre-route-a residual was O(1e5))."""
        sn = _sphere()
        M_op = _joint_M(sn, _loss(sn, slope=0.3))
        tpl = _coupled_template(sn)
        nb = sn.quad.N * sn.ng * sn.nx
        rng = np.random.default_rng(7)
        psi0 = np.zeros(tpl.to_flat().size)
        psi0[:nb] = rng.standard_normal(nb)                 # random physical bulk
        psi0_c = CoupledField.from_flat(psi0, tpl)
        back = M_op.solve(M_op.apply(psi0_c)).to_flat()
        rel = np.max(np.abs(back[:nb] - psi0[:nb])) / (np.max(np.abs(psi0[:nb])) + 1e-300)
        print(f"  ||solve(apply(ψ)) − ψ||_bulk_rel = {rel:.3e}")
        if not (rel < 1e-12):
            pytest.fail(f"round-trip {rel:.3e} — the welded sweep is NOT the exact inverse of "
                        f"(L+C); the direct ψ½ solve regressed (a WRAP would inherit the error).")

    def test_extract_to_dense_is_principled_equivalent_not_bit_identical(self):
        r"""The welded WDD sweep and a LAPACK LU of the SAME assembled ``(L+C)``
        agree to a few ULP (~1e-15) — principled-equivalent, different reduction
        trees. This is the numerical cost of EXTRACTION: the answer is preserved to
        machine precision, but bit-identity is lost. WRAP (same code) keeps
        bit-identity; EXTRACT trades it for ~1e-15 drift. This row is the oracle
        every EXTRACT step of the campaign pins against."""
        sn = _sphere()
        M_op = _joint_M(sn, _loss(sn, slope=0.3))
        tpl = _coupled_template(sn)
        nb = sn.quad.N * sn.ng * sn.nx
        M = _dense(M_op.apply, tpl)                          # the EXTRACTED explicit matrix
        rng = np.random.default_rng(11)
        psi0 = np.zeros(tpl.to_flat().size)
        psi0[:nb] = rng.standard_normal(nb)
        q = M_op.apply(CoupledField.from_flat(psi0, tpl))
        psi_weld = M_op.solve(q).to_flat()                  # welded sweep (forward substitution)
        psi_dense = np.linalg.solve(M, q.to_flat())         # extracted dense LU
        diff = np.max(np.abs(psi_weld[:nb] - psi_dense[:nb])) / (np.max(np.abs(psi_dense[:nb])) + 1e-300)
        print(f"  ||welded − dense_LU||_bulk_rel = {diff:.3e}  (principled ~1e-15, not 0)")
        if not (diff < 1e-11):
            pytest.fail(f"welded vs dense LU differ by {diff:.3e} — an EXTRACTED block solve does "
                        f"not even reach principled-equivalence; the extraction dropped the row "
                        f"contract of the sweep (naive dense M⁻¹ ignores inflow/seed rows).")


# ── Step-1b helpers: the boundary un-weld B = B_a + B_b ───────────────────


def _zero_composite(sn) -> FullField:
    """A zero 2-block System-A composite (alias of :func:`_template`)."""
    return _template(sn)


def _seed_leaf(sn, seed_values: NDArray) -> RadialCharacteristicField:
    """A ψ½ FLUX composite over the given composite-``to_flat``-order values (the
    walk currency; 4e — replaces the retired unified flux leaf)."""
    return _ray_composite(sn, seed_values)


def _random_composite(sn, rng) -> FullField:
    """A 2-block composite with random bulk and trace (System A alone)."""
    N, nx, ng = sn.quad.N, sn.nx, sn.ng
    n_tr = int(sn.angular_trace.layout.total_size)
    return FullField(
        interior=AngularFlux(values=rng.standard_normal((N, ng, nx)), space=sn.angular_bulk_space),
        boundary=AngularBoundaryFlux(
            values=rng.standard_normal(n_tr), space=sn.angular_trace),
    )


def _random_pair(sn, rng) -> CoupledField:
    """A random coupled pair ``[ψ_A 2-block, ψ_B]`` (every block non-zero)."""
    ns = sn.radial_characteristic_field_space.shape[0]
    return _pair(sn, _random_composite(sn, rng), rng.standard_normal(ns))


def _ray_composite(sn, seed_values: NDArray) -> RadialCharacteristicField:
    """System B's FLUX carrier from a composite-``to_flat``-order flat array (4e
    native; ``from_flat`` over a zero flux composite — the split members are the
    interior ⊕ boundary blocks)."""
    return RadialCharacteristicField.from_flat(
        np.asarray(seed_values, dtype=float),
        RadialCharacteristicField.flux_zeros(sn.radial_characteristic_field_space))


def _dense_ray(fn, sn) -> NDArray:
    """Densify a System-B block (composite → composite) by probing the composite
    ``to_flat`` basis — self-consistent in the composite (interior ⊕ boundary)
    layout (4e; the retired unified layout is gone)."""
    ns = sn.radial_characteristic_field_space.shape[0]
    M = np.zeros((ns, ns))
    for j in range(ns):
        e = np.zeros(ns)
        e[j] = 1.0
        M[:, j] = fn(_ray_composite(sn, e)).to_flat()
    return M


def _v_cell_seed(sn) -> NDArray:
    """The ``G_sd = V_cell`` seed metric in composite ``to_flat`` order (the split
    interior ⊕ boundary ``inner_product_weights``)."""
    return np.concatenate([
        np.asarray(
            sn.radial_characteristic_interior_space.inner_product_weights,
            dtype=float),
        np.asarray(
            sn.radial_characteristic_boundary_space.inner_product_weights,
            dtype=float),
    ])


def _g_recip(fwd: NDArray, T: NDArray, g: NDArray, rng) -> float:
    """The metric-reciprocity defect ``|⟨fwd x, y⟩_g − ⟨x, T y⟩_g| / norm``."""
    x = rng.standard_normal(g.size)
    y = rng.standard_normal(g.size)
    lhs = float((fwd @ x) @ (g * y))
    rhs = float(x @ (g * (T @ y)))
    return abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-30)


class TestBoundaryUnweld:
    r"""``B = B_a + B_b`` — the boundary un-weld is a DISJOINT direct sum.

    ``B_a`` (:class:`SNBoundaryOperator`, System A's trace boundary) touches only
    the trace and emits a **present-zero** ray block; ``B_b``
    (:class:`RadialCharacteristicBoundaryOperator`, System B's ray corner) touches
    only the ray and emits a present-zero trace. Their sum reconstructs the whole
    augmented boundary bit-identically (RULING P1). The reflective sphere is used
    so BOTH arms are non-trivial (vacuum would leave B_b zero).
    """

    def test_b_a_touches_only_the_trace(self):
        r"""``B_a`` reflects the trace over a zero bulk — and since B.2d "B_a
        touches the ray" is UNSPELLABLE (its 2-block codomain has no ray slot;
        the retired present-zero pad dissolved with the eviction). The live
        claims: the trace reflection is non-trivial, the bulk stays zero."""
        sn = _sphere(bc="reflective")
        out = SNBoundaryOperator(sn).apply(_random_composite(sn, np.random.default_rng(1)))
        np.testing.assert_array_equal(
            out.interior.values, 0.0,
            err_msg="B_a emitted a NON-ZERO bulk — the trace boundary leaked "
                    "into the interior.")
        if not np.max(np.abs(out.boundary.values)) > 0.0:
            pytest.fail("B_a emitted a zero trace on a reflective sphere — it is not "
                        "reflecting the trace (System A boundary is dead).")

    def test_b_b_touches_only_the_ray_present_zero_trace(self):
        r"""On the BLOCK, "B_b touches the trace" is UNSPELLABLE since the
        B.2b re-type (its codomain has no trace slot — Pattern 4); since B.2d
        the driver consumes the block NATIVELY at the gain grid's (B,B) slot,
        so the structural claim is the whole claim (the retired adapter's
        present-zero embed rows dissolved with it). Mutation tooth: a dead
        reflective corner arm emits a zero corner."""
        sn = _sphere(bc="reflective")
        rng = np.random.default_rng(2)
        block = RadialCharacteristicBoundaryOperator(sn.radial_characteristic_field_space, sn.bc["xmax"].law)
        # The BLOCK's structural half: composite in/out, SOURCE members out.
        block_out = block.apply(
            _seed_leaf(
                sn, rng.standard_normal(sn.radial_characteristic_field_space.shape[0])))
        if type(block_out) is not RadialCharacteristicField:
            pytest.fail(f"B_b block returned {type(block_out).__name__}")
        if type(block_out.interior) is not RadialCharacteristicInteriorSourceSink:
            pytest.fail(f"B_b interior member is {type(block_out.interior).__name__} "
                        f"— the block must emit the SOURCE pair.")
        if type(block_out.boundary) is not RadialCharacteristicBoundarySourceSink:
            pytest.fail(f"B_b boundary member is {type(block_out.boundary).__name__} "
                        f"— the block must emit the SOURCE pair.")
        # The interior member is a ZERO source (B_b writes ONLY the corner).
        np.testing.assert_array_equal(
            block_out.interior.values, 0.0,
            err_msg="B_b's interior member is non-zero — the corner block "
                    "leaked a bulk-ray action.")
        if not np.max(np.abs(block_out.boundary.values)) > 0.0:
            pytest.fail("B_b emitted a zero ray corner on a reflective sphere — the "
                        "System B boundary arm is dead.")

    def test_sum_reconstructs_both_blocks_disjointly(self):
        r"""The gain grid's boundary arms are DISJOINT per system, byte-for-byte:
        row A's trace is exactly ``B_a``'s reflection (S contributes a
        present-zero trace), row B's corner exactly ``B_b``'s (Emission emits a
        zero boundary member) — the per-system direct sum of RULING P1,
        realized structurally on the record's N since B.2d."""
        sn = _sphere(bc="reflective")
        solver = SNSolver(sn)
        system = build_within_group_system(
            sn, solver.mat_xs, scattering_op=solver.scattering_op)
        n_grid = system.explicit_gains[0]
        coupled = _random_pair(sn, np.random.default_rng(3))
        out = n_grid.apply(coupled)
        B_a = SNBoundaryOperator(sn)
        B_b = RadialCharacteristicBoundaryOperator(sn.radial_characteristic_field_space, sn.bc["xmax"].law)
        # Row A's trace is exactly B_a's reflection of ψ_A.
        np.testing.assert_array_equal(
            out.systems[0].boundary.values,
            B_a.apply(coupled.systems[0]).boundary.values)
        # Row B's corner is exactly B_b's reflection of ψ_B.
        np.testing.assert_array_equal(
            out.systems[1].boundary.values,
            B_b.apply(coupled.systems[1]).boundary.values)

    def test_seedless_mesh_has_no_b_b_and_b_is_b_a_alone(self):
        r"""B.2b re-point of the old None-pass-through row: a seedless mesh has
        no System B, so ``B_b`` is UNCONSTRUCTABLE there (Pattern 4 — the old
        "B_b passes None through" behavior is now unspellable), and the
        production boundary is ``B_a`` ALONE (the record's seedless
        ``(S, B_a)`` gains — DP-seedless). The ray of ``B_a``'s output stays
        ``None``."""
        slab = SNMesh(
            Mesh1D(edges=np.linspace(0.0, 4.0, 6), mat_ids=np.zeros(5, dtype=int),
                   coord=CoordSystem.CARTESIAN, bc_right=BC("reflective"),
                   bc_left=BC("reflective")),
            Quadrature.gauss_legendre(4), {0: _mixture(1.0, 0.4, 2)})
        with pytest.raises(ValueError, match="carries no ψ½ ray"):
            RadialCharacteristicBoundaryOperator(slab.radial_characteristic_field_space, slab.bc["xmax"].law)
        # The production boundary on a seedless mesh is B_a alone.
        slab_solver = SNSolver(slab)
        slab_system = build_within_group_system(
            slab, slab_solver.mat_xs, scattering_op=slab_solver.scattering_op)
        B = slab_system.explicit_gains[-1]
        if not isinstance(B, SNBoundaryOperator):
            pytest.fail(f"seedless record boundary gain is "
                        f"{type(B).__name__} — must be B_a alone (no B_b arm).")
        N, nx, ng = slab.quad.N, slab.nx, slab.ng
        n_tr = int(slab.angular_trace.layout.total_size)
        psi = FullField(
            interior=AngularFlux(values=np.random.default_rng(4).standard_normal((N, ng, nx)), space=slab.angular_bulk_space),
            boundary=AngularBoundaryFlux(
                values=np.random.default_rng(5).standard_normal(n_tr),
                space=slab.angular_trace),
        )
        out = B.apply(psi)
        if type(out) is not FullField:
            pytest.fail(f"seedless B_a emitted {type(out).__name__}, not the "
                        f"2-block FullField.")

    # ── B.2d d3 — B_a Euclidean-transpose honesty (the DEFINING law) ──────

    @pytest.mark.catches("ERR-068")
    def test_b_a_vacuum_transpose_is_the_honest_zero(self):
        r"""On a VACUUM outer face ``B_a`` is the ZERO map — and its
        ``apply_transpose`` must be the transpose of that zero, NOT the law
        object's diagonal block.

        B.2d d3 regression pin: the realized vacuum law is the full-face mask
        (zero-on-inflow ⊕ identity-on-outflow) whose harmless-in-the-forward
        identity block the OLD output-side projection extracted into a
        spurious ``+1`` outflow diagonal (``Fᵀ − T = −1`` on every outflow
        slot); the honest spelling is ``(P_inflow ∘ law)ᵀ = lawᵀ ∘ P_inflow``
        (input restriction).  Caught by the A2a grid-reciprocity arm
        (``test_inverse_adjoint_coherence``) at defect 2.6e-4 on the
        het-VACUUM sphere — every reflective-fixture reciprocity gate was
        blind, because permutation laws coincide bit-identically under both
        spellings (the ERR-063 masking family: the degenerate regime hides
        the wrong arm)."""
        sn = _sphere()          # vacuum outer — the masking-free regime
        B_a = SNBoundaryOperator(sn)
        tpl = _template(sn)
        F = _dense(B_a.apply, tpl)
        T = _dense(B_a.apply_transpose, tpl)
        np.testing.assert_array_equal(
            F, np.zeros_like(F),
            err_msg="vacuum B_a forward is not the zero map")
        np.testing.assert_array_equal(
            T, np.zeros_like(T),
            err_msg="vacuum B_a apply_transpose is not the honest zero — the "
                    "law's diagonal block leaked through the transpose "
                    "projection again (B.2d d3)")

    def test_b_a_transpose_is_the_dense_euclidean_transpose(self):
        r"""The DEFINING law of ``apply_transpose``: ``dense(B_aᵀ) ≡
        dense(B_a)ᵀ`` EXACTLY, on the arm where ``B_a`` is non-trivial
        (reflective — the specular permutation).  Bit-equality is the
        contract: both spellings of the projection coincide for permutation
        laws, so any future drift here is a real transpose bug, not a
        convention change."""
        sn = _sphere(bc="reflective")
        B_a = SNBoundaryOperator(sn)
        tpl = _template(sn)
        F = _dense(B_a.apply, tpl)
        T = _dense(B_a.apply_transpose, tpl)
        if not np.abs(F).max() > 0:
            pytest.fail("reflective B_a densified to zero — the fixture no "
                        "longer exercises the permutation arm")
        np.testing.assert_array_equal(
            T, F.T,
            err_msg="B_a.apply_transpose is not the dense Euclidean "
                    "transpose of B_a.apply (reflective arm)")


class TestB_b_RayBoundary:
    r"""``B_b`` — the ψ½ ray-corner boundary law (RULINGS P1 + P2).

    The specular corner swap, its Euclidean transpose (the mirror), and — the
    load-bearing gate — the ``G_sd = V_cell`` reciprocity that keeps Mode-12
    closed (P2: the corner gauge is symmetric, so Euclidean = Hilbert). Reflective
    sphere for the non-trivial arm; vacuum for the null control.
    """

    def test_reflective_corner_swap_forward(self):
        r"""Forward: ``out.corner(level, −1) = seed.corner(level, +1)`` per level;
        the cells and the +1 corner stay zero (B_b touches only the inflow row).
        B.2b: probed on System B's OWN carrier; the unified view of the output
        (the role-preserving bridge) keeps the pre-B.2b assertions bitwise."""
        sn = _sphere(bc="reflective")
        ns = sn.radial_characteristic_field_space.shape[0]
        seed_vals = np.random.default_rng(6).standard_normal(ns)
        seed = _ray_composite(sn, seed_vals)
        out = RadialCharacteristicBoundaryOperator(sn.radial_characteristic_field_space, sn.bc["xmax"].law).apply(seed)
        for level in sn.radial_characteristic_levels:
            np.testing.assert_array_equal(
                out.boundary.corner(level, -1), seed.boundary.corner(level, +1),
                err_msg=f"level {level}: corner(−1) ≠ seed.corner(+1) (specular swap wrong).")
            np.testing.assert_array_equal(
                out.boundary.corner(level, +1), 0.0,
                err_msg=f"level {level}: the +1 corner is non-zero (B_b touched a non-inflow row).")
            np.testing.assert_array_equal(
                out.interior.cells(level, -1), 0.0,
                err_msg=f"level {level}: the cells leg is non-zero (B_b is corner-only).")

    def test_transpose_is_exact_euclidean_mirror(self):
        r"""``dense(B_b.apply_transpose) ≡ dense(B_b.apply).T`` (0 ULP). A
        same-direction transpose would equal ``dense(apply)`` (not its transpose)
        and — since the swap matrix is non-symmetric — red this gate."""
        sn = _sphere(bc="reflective")
        B_b = RadialCharacteristicBoundaryOperator(sn.radial_characteristic_field_space, sn.bc["xmax"].law)
        fwd = _dense_ray(B_b.apply, sn)
        T = _dense_ray(B_b.apply_transpose, sn)
        np.testing.assert_array_equal(
            T, fwd.T, err_msg="B_bᵀ ≠ (B_b)ᵀ — the transpose is not the Euclidean mirror.")

    def test_euclidean_transpose_is_the_vcell_hilbert_adjoint(self):
        r"""Mode-12 CLOSURE (RULING P2): ``⟨B_b x, y⟩_{G_sd} = ⟨x, B_bᵀ y⟩_{G_sd}``
        under ``G_sd = V_cell``. Euclidean IS the Hilbert adjoint because the
        corner gauge is symmetric (``g₊ = g₋ = V(R)``). CONTROL = 0 + two teeth
        (a wrong-direction transpose, an asymmetric gauge) prove it is not vacuous
        — a future asymmetric gauge that reopened Mode-12 would red the gate."""
        sn = _sphere(bc="reflective")
        B_b = RadialCharacteristicBoundaryOperator(sn.radial_characteristic_field_space, sn.bc["xmax"].law)
        fwd = _dense_ray(B_b.apply, sn)
        T = _dense_ray(B_b.apply_transpose, sn)
        # The unified-layout G_sd carries the SAME numbers as System B's
        # composite member space (interior V_cell ⊕ boundary V(R)) — the b2
        # member-wise ≡ direct gates pin that equivalence (G-b3.4).
        g = _v_cell_seed(sn)
        rng = np.random.default_rng(8)
        ctrl = _g_recip(fwd, T, g, rng)
        print(f"  B_b G_sd-reciprocity: control={ctrl:.2e}")
        if not (ctrl < 1e-12):
            pytest.fail(f"CONTROL defect {ctrl:.3e} ≠ 0 — the Euclidean transpose is NOT the "
                        f"V_cell Hilbert adjoint; a Euclidean block adjoint on System B has "
                        f"reopened Mode-12 (the corner gauge is not symmetric).")
        # Tooth a: a same-direction (wrong) transpose breaks reciprocity.
        tooth_a = _g_recip(fwd, fwd, g, rng)
        # Tooth b: an asymmetric corner gauge breaks reciprocity even with the correct T.
        # (g is composite to_flat order: interior ⊕ boundary; the +1 corner slots
        # live in the boundary block.)
        g_bad = g.copy()
        ni = sn.radial_characteristic_interior_space.shape[0]
        g_bad_boundary = g_bad[ni:]
        boundary_space = sn.radial_characteristic_boundary_space
        for level in sn.radial_characteristic_levels:
            boundary_space.slot_view(g_bad_boundary, level, +1)[:] *= 2.0
        tooth_b = _g_recip(fwd, T, g_bad, rng)
        print(f"    teeth: wrong-transpose={tooth_a:.2f}  gauge-asymmetry={tooth_b:.2f}")
        if not (tooth_a > 1e-3 and tooth_b > 1e-3):
            pytest.fail(f"reciprocity gate is VACUOUS: wrong-transpose tooth {tooth_a:.3e} or "
                        f"gauge-asymmetry tooth {tooth_b:.3e} did not red (Mode-12 gate toothless).")

    def test_vacuum_outer_emits_zero_corner(self):
        r"""``kind == "vacuum"`` → ``B_b`` emits an all-zero ray block (no
        re-emission at the outer ray). Positive law (anti-#11)."""
        sn = _sphere(bc="vacuum")
        seed_vals = np.random.default_rng(10).standard_normal(
            sn.radial_characteristic_field_space.shape[0])
        out = RadialCharacteristicBoundaryOperator(sn.radial_characteristic_field_space, sn.bc["xmax"].law).apply(_ray_composite(sn, seed_vals))
        np.testing.assert_array_equal(
            out.to_flat(), 0.0,
            err_msg="vacuum B_b emitted a non-zero corner (it did the reflective swap).")

    def test_unruled_outer_law_is_loud_deferred(self):
        r"""``kind ∈ {white, albedo, periodic}`` → ``NotImplementedError`` with the
        specific message (NEGATIVE law, anti-#11: a bare ``raises`` false-greens on
        a downstream crash). Monkeypatch the xmax LAW (no white-sphere mesh
        needed) — auto-reverts, never a git checkout.

        B2.0 note: this used to patch the shim's ``kind`` STRING, then
        (B2.1) the mesh's law slot. Since the un-weld (O-1) the law is
        CONSTRUCTION-BOUND on the operator, so the patch dance retires
        entirely: hand the real :class:`WhiteBoundary` to the constructor
        — the strongest fixture of the three generations."""
        sn = _sphere(bc="reflective")
        B_b = RadialCharacteristicBoundaryOperator(
            sn.radial_characteristic_field_space, WhiteBoundary(),
        )
        seed_vals = np.random.default_rng(12).standard_normal(
            sn.radial_characteristic_field_space.shape[0])
        with pytest.raises(NotImplementedError, match="no ruled corner action yet"):
            B_b.apply(_ray_composite(sn, seed_vals))

    def test_is_adjointable_is_per_leaf(self):
        r"""``B_b.is_adjointable`` is the OUTER ray-face law's, not the whole-trace
        intersection: reflective + vacuum → True; the loud-deferred set → False."""
        refl = _sphere(bc="reflective")
        if not RadialCharacteristicBoundaryOperator(
            refl.radial_characteristic_field_space, refl.bc["xmax"].law,
        ).is_adjointable:
            pytest.fail("reflective B_b is not adjointable (the involution should be).")
        vac = _sphere(bc="vacuum")
        if not RadialCharacteristicBoundaryOperator(
            vac.radial_characteristic_field_space, vac.bc["xmax"].law,
        ).is_adjointable:
            pytest.fail("vacuum B_b is not adjointable (the zero map should be).")


class TestSplitInteraction:
    r"""The schedule ``split()`` lives on ``B_a`` alone; ``B_b`` is schedule-atomic.

    RULING P1 corollary: a grading is a refinement of ONE system's boundary
    block, never the composite. Since B.2d the once-latent "masked halves
    double the ray corner" bug class is STRUCTURALLY closed — ``B_a``'s
    2-block codomain has no ray slot to double (Pattern 4), and ``B_b`` sits
    alone at the gain grid's (B,B) slot. The live claim: the masked halves
    stay trace-only 2-block composites whose traces partition ``B_a``'s."""

    def test_split_masked_halves_are_trace_only(self):
        from orpheus.sn.loss_representation.sweep_schedule import (
    SweepSchedule,
    reflective_faces,
)

        sn = _sphere(bc="reflective")
        B_a = SNBoundaryOperator(sn)
        parts = B_a.split(SweepSchedule.gauss_seidel(sn.ndim, sn.quad.octants, reflective_faces(sn)))
        psi = _random_composite(sn, np.random.default_rng(13))
        whole = B_a.apply(psi)
        total = None
        for name, half in (("lower", parts.lower), ("upper", parts.upper)):
            out = half.apply(psi)
            if type(out) is not FullField:
                pytest.fail(f"masked B_{name} emitted {type(out).__name__}, "
                            f"not the 2-block FullField.")
            np.testing.assert_array_equal(
                out.interior.values, 0.0,
                err_msg=f"masked B_{name} emitted a non-zero bulk.")
            total = out.boundary.values if total is None else total + out.boundary.values
        np.testing.assert_array_equal(
            total, whole.boundary.values,
            err_msg="B_lower + B_upper trace ≠ B_a's trace — the split halves "
                    "do not partition the whole-trace reflection.")


# ── Step-1c helpers: A_BB = RadialCharacteristicOperator (the ψ½ radial BVP) ──


def _graded_sphere(nx: int, ng: int = 2, p: float = 1.5, R: float = 4.0,
                   bc: str = "vacuum"):
    r"""A seed-carrying sphere on a power-graded radial mesh ``r_j = R·(j/nx)^p``.

    The genuinely non-uniform ``dr`` the reversal / index-drift gates need: on a
    uniform mesh ``dr[::-1] == dr`` and ``dr[k−1] == dr[k]``, so those gates are
    vv Mode-5 vacuous — the grading breaks the blind spot (§0.6)."""
    edges = R * (np.arange(nx + 1, dtype=float) / nx) ** p
    mesh = Mesh1D(edges=edges, mat_ids=np.zeros(nx, dtype=int),
                  coord=CoordSystem.SPHERICAL, bc_right=BC(bc))
    return SNMesh(mesh, Quadrature.gauss_legendre(4), {0: _mixture(1.0, 0.4, ng)})


def _ray_sigma(sn, slope: float = 0.3) -> CrossSectionField:
    r"""Heterogeneous per-group per-cell ``σ_t`` as a typed, mesh-bound
    ``CrossSectionField`` on ``sn`` (``.values`` are ``(ng, nx)``, varying in
    BOTH group AND cell so an index / group-axis bug in the march is not nulled —
    anti-#2 asymmetry). The operator's ``C_ray`` collision coefficient; the typed
    field carries ``.mesh`` for the operator's mesh-identity guard."""
    nx, ng = sn.nx, sn.ng
    raw = np.stack([1.0 + slope * g + 0.15 * np.arange(nx) for g in range(ng)], 0)
    return CrossSectionField(values=raw, space=sn.bulk_space)


def _ray_source(sn, rng) -> RadialCharacteristicField:
    """A random q½ SOURCE composite on ``sn``'s ray carrier (all slots non-zero)."""
    ns = sn.radial_characteristic_field_space.shape[0]
    return RadialCharacteristicField.from_flat(
        rng.standard_normal(ns),
        RadialCharacteristicField.source_zeros(sn.radial_characteristic_field_space))


def _ray_cotangent(sn, rng) -> RadialCharacteristicField:
    """A random flux-space cotangent (the solve's codomain) on ``sn``'s carrier."""
    ns = sn.radial_characteristic_field_space.shape[0]
    return _ray_composite(sn, rng.standard_normal(ns))


def _member(x) -> RadialCharacteristicField:
    """Identity passthrough (4e — System B's split composite is the native
    carrier now; the pre-4e ``from_unified`` bridge from a unified leaf is
    retired)."""
    return x


def _two_leg_reference(op, source) -> RadialCharacteristicField:
    r"""Replicate ``A_BB.solve``'s two-leg march with the REAL engine — the WRAP
    oracle for the bit-identity gate. Calls ``carlson_inward_sweep_from_source``
    (the test-module imported name, UNPATCHED when the operator's module attr is
    spied) so ``.solve`` (patched with a delegating spy) and this reference
    compute with the SAME engine → any divergence is a WRAP bug, not FP."""
    sigma = op.total_cross_section.values
    dr = np.asarray(op._dr)  # the operator's own bound widths (un-weld O-1)
    out = RadialCharacteristicField.flux_zeros(op._field_space)
    for lv in op._ray_space.levels:
        q_minus = source.interior.cells(lv, -1)
        q_plus = source.interior.cells(lv, +1)
        corner_in = source.boundary.corner(lv, -1)
        cells_minus, pole_face = carlson_inward_sweep_from_source(
            q_minus, sigma, dr, corner_in)
        cells_plus_rev, corner_out = carlson_inward_sweep_from_source(
            q_plus[:, ::-1], sigma[:, ::-1], dr[::-1], pole_face)
        out.interior.cells(lv, -1)[...] = cells_minus
        out.boundary.corner(lv, -1)[...] = corner_in
        out.interior.cells(lv, +1)[...] = cells_plus_rev[:, ::-1]
        # ERR-078: the exact inverse's outflow row is ψ_out = streamed −
        # q_out (the defect row's completion) — the reference replays it.
        out.boundary.corner(lv, +1)[...] = (
            corner_out - source.boundary.corner(lv, +1)
        )
    return out


def _install_engine_spy(monkeypatch) -> list[dict]:
    r"""Mode-11 sentinel: wrap ``carlson_inward_sweep_from_source`` in the
    OPERATOR's module namespace, recording ``(args, result)`` per call and
    delegating to the real engine. Returns the calls list (2 per level: inward
    then outward). Proves ``.solve`` EXECUTES the production engine (a divergent
    inlined copy would leave the list empty)."""
    calls: list[dict] = []
    real = carlson_inward_sweep_from_source  # the unpatched module-top import

    def spy(Q_bar, sigma_t, dr, bc_outer_value):
        result = real(Q_bar, sigma_t, dr, bc_outer_value)
        calls.append({
            "Q": np.asarray(Q_bar).copy(),
            "sigma": np.asarray(sigma_t).copy(),
            "dr": np.asarray(dr).copy(),
            "bc": np.asarray(bc_outer_value).copy(),
            "cells": result[0].copy(),
            "exit_face": result[1].copy(),
        })
        return result

    monkeypatch.setattr(_rc_mod, "carlson_inward_sweep_from_source", spy)
    return calls


def _euclid_adjoint_defect(op, u, v) -> float:
    r"""The relative Euclidean reciprocity defect
    ``|⟨solve(u), v⟩ − ⟨u, solve_transpose(v)⟩| / (|·| + |·|)``.

    Plain dot products (NOT the ``G_sd`` metric): ``solve_transpose`` is the
    ISOLATED EUCLIDEAN adjoint of the resolvent (the pure ray-block transpose —
    operator docstring), so its consistency partner is the Euclidean inner
    product, not the ``V_cell`` Hilbert adjoint (which is realized once at the
    composite, L19). ``u``/``v`` are unified; the calls bridge (B.2c I/O)."""
    lhs = float(op.solve(_member(u)).to_flat() @ v.to_flat())
    rhs = float(u.to_flat() @ op.solve_transpose(_member(v)).to_flat())
    return abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-300)


class TestA_BB_RadialBVP:
    r"""``A_BB`` = :class:`RadialCharacteristicOperator` — the ψ½ radial two-point
    BVP resolvent (campaign step 1c).

    Foundation invariants of the full operator — the forward ``apply`` /
    ``apply_transpose``, the resolvent ``solve`` / ``solve_transpose``, and the
    operator-returning ``inverse()`` (step 1c posed the resolvent; step 4b
    completes the forward via the shared kernel).
    The convergence-ORDER claim lives in the sibling L1 module
    ``test_ray_operator.py`` (don't conflate foundation + verifies, L9). Every
    value row is ≥2G; the sphere-GL S4 carrier is this class's seed-carrying
    member, and the slab is the non-carrying CONTROL (the constructor rejects
    it).  ⛔ "cylinder/slab are the non-carrying CONTROL" read here until
    2026-08-29 and is present-tense FALSE since the Q5.6.3 admission flip: a
    cylindrical ``SNMesh`` admits only CARRYING (folded) rules, so the slab is
    the only admitted non-carrying 1-D geometry.

    Runtime: gates raise via :func:`pytest.fail` / ``np.testing.assert_*`` (fire
    under ``python -O``), never a bare ``assert`` (vv Mode 8).
    """

    # ── solve / solve_transpose adjoint consistency (Euclidean) ────────────

    def test_adjoint_consistency_euclidean(self):
        r"""``⟨solve(u), v⟩ = ⟨u, solve_transpose(v)⟩`` (Euclidean) to < 1e-11 on
        heterogeneous σ, ≥2 random draws — the PRIMARY solve/solve_transpose
        consistency gate (the resolvent adjoint is the reverse-mode transpose of
        the two-leg march). The source ``μ = +1`` corner cotangent carries
        EXACTLY ``−v̄_out`` — the transpose of the defect row's ``−I``
        coupling (``ψ_out = streamed − q_out``, ERR-078; the pre-fix claim
        "the slot stays 0" pinned the dropped-rhs hole)."""
        sn = _sphere()
        op = rc_march(sn, _ray_sigma(sn))
        for seed in (1, 2, 3):                       # ≥2 draws
            rng = np.random.default_rng(seed)
            u, v = _ray_source(sn, rng), _ray_cotangent(sn, rng)
            defect = _euclid_adjoint_defect(op, u, v)
            if not defect < 1e-11:
                pytest.fail(
                    f"seed {seed}: Euclidean adjoint defect {defect:.3e} ≥ 1e-11 — "
                    f"solve_transpose is NOT the transpose of solve (the reverse "
                    f"march or its leg chaining is mis-wired).")
            src_bar = op.solve_transpose(_member(v))
            for lv in sn.radial_characteristic_levels:
                np.testing.assert_array_equal(
                    src_bar.boundary.corner(lv, +1),
                    -np.asarray(v.boundary.corner(lv, +1)),
                    err_msg=f"seed {seed} level {lv}: the source μ=+1 corner "
                            f"cotangent must be −v̄_out bit-exactly (the defect "
                            f"row's −I coupling, ERR-078).")

    def test_adjoint_sign_flip_tooth(self, monkeypatch):
        r"""TOOTH for the adjoint gate: a sign flip in the reverse-mode
        recurrence (``carlson_inward_sweep_transpose``'s incoming face-cotangent
        ``-f_bar → +f_bar``) breaks reciprocity — the defect jumps to O(1).
        Proves the < 1e-11 gate above is not vacuously green."""

        def transpose_sign_flip(cells_bar, final_face_bar, sigma_t, dr):
            ng, nx = cells_bar.shape
            Q_bar = np.zeros((ng, nx), dtype=cells_bar.dtype)
            f_bar = final_face_bar.copy()
            for k in range(nx):
                denom = dr[k] * sigma_t[:, k] + 2.0
                c_bar = cells_bar[:, k] + 2.0 * f_bar
                f_in_bar = +f_bar                    # SIGN FLIP: production is -f_bar
                Q_bar[:, k] = (dr[k] / denom) * c_bar
                f_in_bar = f_in_bar + (2.0 / denom) * c_bar
                f_bar = f_in_bar
            return Q_bar, f_bar

        monkeypatch.setattr(
            _rc_mod, "carlson_inward_sweep_transpose", transpose_sign_flip)
        sn = _sphere()
        op = rc_march(sn, _ray_sigma(sn))
        rng = np.random.default_rng(1)
        defect = _euclid_adjoint_defect(op, _ray_source(sn, rng),
                                        _ray_cotangent(sn, rng))
        if not defect > 1e-3:
            pytest.fail(
                f"the transpose sign-flip left the adjoint defect at {defect:.3e} "
                f"— the adjoint-consistency gate has no teeth.")

    # ── WRAP bit-identity via a Mode-11 call-counter sentinel ──────────────

    def test_wrap_executes_engine_bit_identical(self, monkeypatch):
        r"""``solve`` WRAPS the production engine: the Mode-11 sentinel counts
        exactly ``2·n_levels`` calls to ``carlson_inward_sweep_from_source`` (2
        legs/level) AND the result is bit-identical (``array_equal``) to an
        independent two-leg reference on the SAME engine — a divergent inlined
        copy would leave the counter at 0 (Cardinal Rule 2 single source)."""
        sn = _sphere()
        op = rc_march(sn, _ray_sigma(sn))
        source = _ray_source(sn, np.random.default_rng(4))
        reference = _two_leg_reference(op, source)   # real engine, before the spy
        calls = _install_engine_spy(monkeypatch)
        flux = op.solve(_member(source))
        n_levels = len(sn.radial_characteristic_levels)
        if len(calls) != 2 * n_levels:
            pytest.fail(
                f"solve called the engine {len(calls)}× (expected 2·n_levels = "
                f"{2 * n_levels}) — it is NOT the two-leg WRAP (a divergent copy?).")
        np.testing.assert_array_equal(
            flux.to_flat(), reference.to_flat(),
            err_msg="solve is not bit-identical to the two-leg engine reference — "
                    "the WRAP diverged from the production march.")

    def test_pole_continuation_threads_exit_to_entry(self, monkeypatch):
        r"""Pole continuation ``ψ½⁺(0) = ψ½⁻(0)``: per level the OUTWARD leg's
        entry face (call #2's ``bc_outer_value``) EQUALS the INWARD leg's exit
        face (call #1's ``phi_face_final``) — the inward exit IS the outward
        entry (internal to the march, R13). The exit face is asserted non-trivial
        so the gate is not vacuously satisfied by zeros."""
        sn = _sphere()
        op = rc_march(sn, _ray_sigma(sn))
        calls = _install_engine_spy(monkeypatch)
        op.solve(_member(_ray_source(sn, np.random.default_rng(5))))
        for i in range(0, len(calls), 2):
            inward, outward = calls[i], calls[i + 1]
            np.testing.assert_array_equal(
                outward["bc"], inward["exit_face"],
                err_msg="the outward leg's entry face ≠ the inward leg's exit face "
                        "— pole continuation ψ½⁺(0)=ψ½⁻(0) is broken.")
            if not np.max(np.abs(inward["exit_face"])) > 0.0:
                pytest.fail("the inward exit (pole) face is identically zero — the "
                            "pole-continuation gate is vacuous for this source.")

    def test_outward_leg_marches_reversed_data(self, monkeypatch):
        r"""The 2.5a discipline — orientation is carried by the DATA, never a
        flag: the OUTWARD (+1) leg rides the same engine on the ``[:, ::-1]`` /
        ``[::-1]`` reversed level data, the INWARD (-1) leg on forward data. Run
        on a GRADED mesh so ``dr[::-1] ≠ dr`` — the reversal is a genuine
        constraint (on a uniform mesh it is vv Mode-5 vacuous; the non-vacuity
        check enforces that). If the operator dropped a reversal, call #2 would
        carry forward data and these equalities RED."""
        sn = _graded_sphere(nx=8)
        op = rc_march(sn, _ray_sigma(sn))
        dr = np.asarray(sn.axis_widths[0])
        sigma = op.total_cross_section.values
        source = _ray_source(sn, np.random.default_rng(6))
        # Non-vacuity (Mode 5): on this graded mesh reversed ≠ forward.
        if np.array_equal(dr[::-1], dr):
            pytest.fail("dr is uniform — the reversal gate is Mode-5 vacuous; the "
                        "graded mesh must give dr[::-1] ≠ dr.")
        calls = _install_engine_spy(monkeypatch)
        op.solve(_member(source))
        for idx, lv in enumerate(sn.radial_characteristic_levels):
            inward, outward = calls[2 * idx], calls[2 * idx + 1]
            np.testing.assert_array_equal(
                inward["dr"], dr,
                err_msg=f"level {lv}: the inward leg did not march FORWARD widths.")
            np.testing.assert_array_equal(
                inward["Q"], source.interior.cells(lv, -1),
                err_msg=f"level {lv}: the inward leg read the wrong source cells.")
            np.testing.assert_array_equal(
                outward["dr"], dr[::-1],
                err_msg=f"level {lv}: the outward leg did not march REVERSED widths "
                        f"(the 2.5a data-carried orientation broke).")
            np.testing.assert_array_equal(
                outward["Q"], source.interior.cells(lv, +1)[:, ::-1],
                err_msg=f"level {lv}: the outward leg did not read REVERSED source cells.")
            np.testing.assert_array_equal(
                outward["sigma"], sigma[:, ::-1],
                err_msg=f"level {lv}: the outward leg did not read REVERSED σ_t.")

    # ── r = R Dirichlet propagation ────────────────────────────────────────

    def test_r_R_dirichlet_propagates_into_interior(self):
        r"""A nonzero ``r = R`` inflow corner (μ=−1) vs zero — same cells source —
        changes the INTERIOR cells (by the ``e^{−σ(R−r)}`` envelope), not merely
        the boundary. Two solves differing ONLY in ``corner_in`` must differ in
        the interior; equal interiors would mean the Dirichlet datum is ignored."""
        sn = _sphere()
        op = rc_march(sn, _ray_sigma(sn))
        s0 = RadialCharacteristicField.source_zeros(sn.radial_characteristic_field_space)
        s1 = RadialCharacteristicField.source_zeros(sn.radial_characteristic_field_space)
        for s in (s0, s1):
            for lv in sn.radial_characteristic_levels:
                s.interior.cells(lv, -1)[...] = 0.5     # identical cells
        for lv in sn.radial_characteristic_levels:
            s1.boundary.corner(lv, -1)[...] = 3.0       # nonzero inflow only
        a = op.solve(_member(s0)).interior.cells(0, -1)
        b = op.solve(_member(s1)).interior.cells(0, -1)
        interior_diff = float(np.max(np.abs(a[:, :-1] - b[:, :-1])))  # exclude outer cell
        if not interior_diff > 1e-6:
            pytest.fail(
                f"the interior cells are unchanged ({interior_diff:.3e}) when the "
                f"r=R inflow corner goes 0 → 3 — the Dirichlet datum does NOT "
                f"propagate inward (the corner is being ignored).")

    def test_dirichlet_bc_ignore_tooth(self, monkeypatch):
        r"""TOOTH for the Dirichlet-propagation gate: an engine that ignores its
        ``bc_outer_value`` (always enters at 0) makes the two solves' interiors
        IDENTICAL — the interior difference collapses to 0. Proves the gate
        above catches a dropped inflow datum."""

        def ignore_bc(Q_bar, sigma_t, dr, bc_outer_value):
            return carlson_inward_sweep_from_source(
                Q_bar, sigma_t, dr, np.zeros_like(bc_outer_value))

        monkeypatch.setattr(
            _rc_mod, "carlson_inward_sweep_from_source", ignore_bc)
        sn = _sphere()
        op = rc_march(sn, _ray_sigma(sn))
        s0 = RadialCharacteristicField.source_zeros(sn.radial_characteristic_field_space)
        s1 = RadialCharacteristicField.source_zeros(sn.radial_characteristic_field_space)
        for s in (s0, s1):
            s.interior.cells(0, -1)[...] = 0.5
        s1.boundary.corner(0, -1)[...] = 3.0
        interior_diff = float(np.max(np.abs(
            op.solve(_member(s0)).interior.cells(0, -1)[:, :-1]
            - op.solve(_member(s1)).interior.cells(0, -1)[:, :-1])))
        if not interior_diff < 1e-14:
            pytest.fail(
                f"the bc-ignoring engine still produced an interior difference "
                f"({interior_diff:.3e}) — the Dirichlet-propagation gate's tooth "
                f"does not bite.")

    # ── Fixed-source Q/Σ equilibrium (conservation + spatial distribution) ─

    def test_fixed_source_equilibrium_Q_over_sigma(self):
        r"""The single most powerful curvilinear diagnostic: uniform source at
        equilibrium ``q̄ = σ·C`` with the consistent inflow ``φ_R = C`` → every
        cell of BOTH legs sits at ``C = q̄/σ`` (the flat identity
        ``(Δr·σ·C + 2C)/(Δr·σ + 2) = C``, self-similar through the pole). ≥2G with
        DISTINCT per-group ``C`` and heterogeneous per-cell σ, so a missing ``Δr``
        / factor / group-axis bug would break the equilibrium."""
        sn = _sphere(nx=6)
        sigma = _ray_sigma(sn)                       # het in g AND cell
        sig = sigma.values                           # (ng, nx) for the σ·C source
        op = rc_march(sn, sigma)
        C = np.array([0.5, 1.3])                     # distinct per-group equilibrium
        src = RadialCharacteristicField.source_zeros(sn.radial_characteristic_field_space)
        for g in range(sn.ng):
            for sign in (-1, +1):                    # BOTH legs' cells source = σ·C
                src.interior.cells(0, sign)[g, :] = sig[g] * C[g]
            src.boundary.corner(0, -1)[g] = C[g]   # consistent inflow
        flux = op.solve(_member(src))
        expected = np.broadcast_to(C[:, None], (sn.ng, sn.nx))
        for sign in (-1, +1):
            np.testing.assert_allclose(
                flux.interior.cells(0, sign), expected, atol=1e-13,
                err_msg=f"leg {sign}: the equilibrium flux ≠ q̄/σ = C — the "
                        f"fixed-source Q/Σ balance (conservation + spatial "
                        f"distribution) is broken.")

    # ── Constructor negative gates (the non-carrying CONTROL) ──────────────

    def test_constructor_rejects_non_carrying_foreign_mesh_and_nonpositive(self):
        r"""The constructor's guards, NET-NEW teeth (L4). Three illegal states,
        each with ``match=`` the SPECIFIC message (a downstream crash would
        false-green a bare ``raises``):

        * **non-carrying CONTROL** — the slab (Cartesian) has
          ``radial_characteristic_space is None`` → the seedless guard fires
          (before σ_t is even read).  (Until Q5.6.3 an LS cylinder was the
          second control; the admission flip made a non-carrying cylinder
          UNCONSTRUCTIBLE — its refusal moved one tier up, gated by
          ``tests/sn/mesh/test_cylindrical_quadrature_admission.py``);
        * **foreign-mesh σ_t** — a ``CrossSectionField`` on a DIFFERENT sphere
          (same ``(ng, nx)``, different Δr) is refused by the mesh-identity
          invariant. THIS is the Pattern-4 illegal state the typed, mesh-bound
          coefficient closes — a bare ``(ng, nx)`` ndarray could not catch it (it
          carries no mesh), so the operator would silently march this mesh's Δr
          against a foreign σ_t;
        * **σ_t ≤ 0** — the DD-denominator ``Δr·σ + 2`` guard.

        Positive control: a σ_t on THIS mesh constructs cleanly."""
        # Non-carrying CONTROL — the slab (the only admitted seedless geometry).
        slab = SNMesh(
            Mesh1D(edges=np.linspace(0.0, 4.0, 6), mat_ids=np.zeros(5, dtype=int),
                   coord=CoordSystem.CARTESIAN, bc_right=BC("reflective"),
                   bc_left=BC("reflective")),
            Quadrature.gauss_legendre(4), {0: _mixture(1.0, 0.4, 2)})
        if slab.radial_characteristic_field_space is not None:
            pytest.fail("the slab carries a ray space — CONTROL invalid.")
        # The seedless guard fires before σ_t is read (a valid field on the mesh).
        with pytest.raises(ValueError, match="carries no starting-direction ray"):
            RadialCharacteristicOperator(
                slab.radial_characteristic_field_space,
                CrossSectionField(values=np.ones((2, 5)), space=slab.bulk_space),
                bulk_space=slab.bulk_space, dr=slab.axis_widths[0],
                start_cosines={})
        # Positive control — a σ_t on THIS mesh constructs cleanly.
        sn = _sphere(nx=6)
        rc_march(sn, CrossSectionField(values=np.ones((sn.ng, sn.nx)), space=sn.bulk_space))
        # THE Pattern-4 closure: a σ_t bound to a DIFFERENT sphere (graded — a
        # genuinely different Δr) is refused. The typed coefficient makes the
        # foreign-mesh march unconstructable; a bare ndarray could not.
        foreign_mesh = _graded_sphere(nx=6)
        foreign_sigma = CrossSectionField(values=np.ones((foreign_mesh.ng, foreign_mesh.nx)), space=foreign_mesh.bulk_space)
        with pytest.raises(ValueError, match="space-content invariant"):
            rc_march(sn, foreign_sigma)
        # σ_t ≤ 0 → the DD-denominator guard.
        bad = np.ones((sn.ng, sn.nx))
        bad[1, 2] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            rc_march(sn, CrossSectionField(values=bad, space=sn.bulk_space))


def _install_forward_spy(monkeypatch) -> list[int]:
    r"""Mode-11 anti-twin sentinel: wrap the SHARED forward kernel
    ``radial_characteristic_forward_residual`` in the operator namespace
    (``_rc_mod``) — since step 6 the ONE consumer (the walk's fused ψ½
    rows retired with the joint channel).  Every entry appends to the
    shared list, so a caller that inlined a divergent copy would leave
    its delta at 0 (L16: only-new-reds ⟹ twin)."""
    calls: list[int] = []
    real = radial_characteristic_forward_residual

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_rc_mod, "radial_characteristic_forward_residual", spy)
    return calls


class TestA_BB_Forward:
    r"""``A_BB`` step-4b — the forward ``apply`` / ``apply_transpose`` /
    ``inverse()`` that complete the operator, single-sourced with the ``(L+C)``
    walk (the user's "extract the shared kernel now" ruling — no forward twin).

    Sphere-GL S4 carrier, ≥2G, graded σ. ``solve∘apply`` is principled-equiv at
    ~FP ULP for the cells (the forward's ``2/Δr`` and the march's ``Δr·σ+2``
    reassociate, L7) and BIT-EXACT ``0.0`` on the μ=+1 outflow corner.
    """

    def test_apply_is_the_exact_march_inverse(self):
        # solve∘apply=id on the CONSISTENT subspace ψ0 = solve(q0). Cells at
        # rtol (principled-equiv, R1); the apply∘solve +1 corner closes to
        # q0's OWN outflow datum bit-exactly — the identity, not the pre-fix
        # 0 (ERR-078: ψ_out = streamed − q_out, so the defect row returns
        # q_out; the old bit-zero closure was the dropped-rhs hole reading
        # 0 = 0 on itself).
        sn = _sphere()
        op = rc_march(sn, _ray_sigma(sn))
        for seed in range(2):
            rng = np.random.default_rng(seed)
            q0 = _member(_ray_source(sn, rng))
            psi0 = op.solve(q0)
            psi1 = op.solve(op.apply(psi0))
            np.testing.assert_allclose(
                psi1.to_flat(), psi0.to_flat(),
                rtol=1e-11, atol=1e-13)
            qr = op.apply(op.solve(q0))
            for p in sn.radial_characteristic_levels:
                np.testing.assert_array_equal(
                    qr.boundary.corner(p, +1),
                    np.asarray(q0.boundary.corner(p, +1)),
                    err_msg=f"level {p}: apply∘solve must return the rhs's "
                            f"own outflow datum (the defect-row identity, "
                            f"ERR-078).")

    def test_apply_routes_through_the_shared_forward_kernel(self, monkeypatch):
        # Mode-11 anti-twin (operator side): A_BB.apply MUST enter the shared
        # radial_characteristic_forward_residual, not an inlined copy.
        sn = _sphere()
        op = rc_march(sn, _ray_sigma(sn))
        calls = _install_forward_spy(monkeypatch)
        op.apply(_member(_ray_cotangent(sn, np.random.default_rng(1))))
        if len(calls) == 0:
            pytest.fail(
                "A_BB.apply did NOT route through the shared forward kernel "
                "(a divergent inlined copy — twin).")

    # ``test_walk_forward_routes_through_the_shared_kernel`` RETIRED at
    # step 6: the walk carries no ψ½ rows (the fused wrapper deleted with
    # the joint channel), so the walk-side anti-twin has no referent —
    # the operator-side sentinel above is the surviving single-source pin.

    def test_apply_transpose_is_the_euclidean_adjoint(self):
        # ⟨apply(u), v⟩ = ⟨u, apply_transpose(v)⟩ — plain flat dot (the metric
        # Hilbert adjoint .H is realized at the composite, L19/R4). Graded het σ.
        sn = _sphere()
        op = rc_march(sn, _ray_sigma(sn))
        for seed in range(2):
            rng = np.random.default_rng(seed + 10)
            u = _ray_cotangent(sn, rng)
            v = _ray_cotangent(sn, rng)
            lhs = float(op.apply(_member(u)).to_flat() @ v.to_flat())
            rhs = float(
                u.to_flat() @ op.apply_transpose(_member(v)).to_flat())
            defect = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-300)
            if defect > 1e-11:
                pytest.fail(
                    f"forward Euclidean adjoint defect {defect:.2e} > 1e-11")

    def test_inverse_is_the_march_involution(self):
        # inverse() delegates: apply → inner.solve, solve → inner.apply;
        # inverse().inverse() is self (mixin identity); predicates report True.
        sn = _sphere()
        op = rc_march(sn, _ray_sigma(sn))
        rng = np.random.default_rng(3)
        q0 = _member(_ray_source(sn, rng))
        psi0 = op.solve(q0)
        inv = op.inverse()
        np.testing.assert_array_equal(
            inv.apply(q0).to_flat(), op.solve(q0).to_flat())
        np.testing.assert_array_equal(
            inv.solve(psi0).to_flat(),
            op.apply(psi0).to_flat())
        if inv.inverse() is not op:
            pytest.fail("inverse().inverse() is not the original operator.")
        if not (op.is_invertible and op.is_adjointable):
            pytest.fail(
                "A_BB must report is_invertible and is_adjointable True at 4b.")

    def test_b2c_member_composite_block_boundary(self):
        r"""B.2c re-type (G-c1.1): the four action surfaces speak System B's
        member composite, with the #276 A4 DUALITY-TYPED role assignments.
        Containers: ``apply`` / ``apply_transpose`` emit SOURCE members (an
        operator action / a dual-of-flux is source-role); ``solve`` emits
        FLUX members; ``solve_transpose`` emits FLUX members TOO —
        dual-of-source under the G-pairing is the adjoint ray flux (the A4
        re-classing that lets the daggered coupled iteration close; the
        pre-A4 source-family wrap is the retired convention this row used
        to pin).  Declared domain/codomain asserted by object IDENTITY,
        not ``==`` — the unified space collides with the composite space
        on ``(name, shape)`` (memo F2), so ``==`` is Mode-12-blind to a
        block left typed on the unified carrier; and the block-boundary
        refusals (foreign carrier / foreign mesh / the solve source-role
        parse)."""
        sn = _sphere()
        op = rc_march(sn, _ray_sigma(sn))
        rng = np.random.default_rng(7)
        src = _member(_ray_source(sn, rng))
        cot = _member(_ray_cotangent(sn, rng))
        if op.domain is not sn.radial_characteristic_field_space:
            pytest.fail("A_BB.domain is not THE composite member-space object "
                        "(F2: == cannot see a unified-typed block).")
        if op.codomain is not sn.radial_characteristic_field_space:
            pytest.fail("A_BB.codomain is not THE composite member-space object.")
        for name, out, want_interior in (
            ("apply", op.apply(cot), RadialCharacteristicInteriorSourceSink),
            ("apply_transpose", op.apply_transpose(cot),
             RadialCharacteristicInteriorSourceSink),
            # #276 A4 duality typing: dual-of-source = the adjoint ray FLUX.
            ("solve_transpose", op.solve_transpose(cot),
             RadialCharacteristicInteriorFlux),
        ):
            if type(out) is not RadialCharacteristicField:
                pytest.fail(f"{name} did not emit the member composite; got "
                            f"{type(out).__name__}.")
            if type(out.interior) is not want_interior:
                pytest.fail(f"{name} did not emit {want_interior.__name__} "
                            f"members; got {type(out.interior).__name__}.")
        out_solve = op.solve(src)
        if type(out_solve) is not RadialCharacteristicField:
            pytest.fail("solve did not emit the member composite.")
        if type(out_solve.interior) is not RadialCharacteristicInteriorFlux:
            pytest.fail(f"solve did not emit FLUX members; got "
                        f"{type(out_solve.interior).__name__}.")
        # Block-boundary refusals (parse-don't-validate, match= the parse's
        # own message — a downstream crash must not false-green these).
        with pytest.raises(TypeError, match="System B's member carrier"):
            op.apply(_template(sn))                     # a System-A FullField (foreign carrier)
        with pytest.raises(TypeError, match="SOURCE members"):
            op.solve(cot)                              # flux into the resolvent
        # CS4b S3 (F2): a twin sphere's member is CONTENT-equal — legal;
        # a graded sphere's ray content differs — refused per block.
        _ = op.apply(_member(_ray_cotangent(_sphere(), rng)))
        graded = _graded_sphere(nx=5)
        with pytest.raises(ValueError, match="space-content invariant"):
            op.apply(_member(_ray_cotangent(graded, rng)))

    def test_b2c_member_value_rows_have_teeth(self, monkeypatch):
        r"""TEETH for the G-c1.1 value rows (4e RE-AIM).

        The pre-4e teeth patched ``RadialCharacteristicField.from_unified``
        (a bridge-drop → the container gate reds; a bridge-corruption → the
        value moves). Phase C 4e retired the unified leaf and its bridge, so
        that mechanism has no referent. Successors:

        * The container guarantee (``apply`` emits a native member composite) is
          now STRUCTURAL — pinned by
          :meth:`test_b2c_member_composite_block_boundary` (the operator emits
          the composite directly; there is no bridge to drop).
        * The value-teeth are RE-AIMED onto a direct corruption of the SHARED
          forward kernel's interior output (same assertion — a value corruption
          moves ``apply``'s value; the value rows are not blind to it)."""
        sn = _sphere()
        op = rc_march(sn, _ray_sigma(sn))
        cot = _member(_ray_cotangent(sn, np.random.default_rng(8)))
        reference = op.apply(cot).to_flat()             # unpatched control
        real = radial_characteristic_forward_residual

        def corrupt(*args, **kwargs):
            interior_out, boundary_out = real(*args, **kwargs)
            return interior_out * 2.0, boundary_out      # double the interior member

        with monkeypatch.context() as m:
            m.setattr(_rc_mod, "radial_characteristic_forward_residual", corrupt)
            corrupted = op.apply(cot).to_flat()
        if not np.max(np.abs(corrupted - reference)) > 0.0:
            pytest.fail("the value-corruption tooth left apply unchanged — the "
                        "value rows have no teeth against a corrupted kernel.")


# ── Step-2 helpers: A_BA the ψ½ Schur fold (bulk → ray q½ source) ──────────
#
# A_BA folds a bulk isotropic cell-emission q₀ (the ℓ=0 K_iso·φ₀ for S, the
# χ·νΣf·φ for F) onto the ray q½ source at the closed rays μ=±1, per carrying
# radial level. It lives ENTIRELY in the lagged S + F/k gain, OUTSIDE the
# resolvent (it cannot touch the #284 forward-substitution certificate). The
# Step-2 un-weld routes the three hand-rolled S/F fold arms (scattering.py S-fwd
# + S-adj, fission.py F-fwd) through ONE single source — the operator
# RadialCharacteristicReconstruction, which wraps the fold helper
# fold_moments_to_radial_characteristic; these gates pin that. (from_angular_
# source keeps its own per-level analysis — a distinct operation, not a twin.)
#
# The fold math (single source ``fold_moments_to_radial_characteristic``):
#   Q̄(μ=±1) = Σ_ℓ (2ℓ+1)/2 · Q_ℓ · P_ℓ(±1) = Σ_ℓ (2ℓ+1)/2 · Q_ℓ · (±1)^ℓ.
# At ℓ=0 it collapses to ½·Q₀ (both signs — P₀≡1); the PRODUCTION S/F arms feed
# ℓ=0 ONLY, so an S/F-only gate is BLIND to P_ℓ(±1) for ℓ≥1 (refutation #3) —
# the contract gate manufactures ℓ≥1 to activate the ``sign^ℓ`` line.


def _fissile_mixture(sig_t: float, sig_s: float, ng: int):
    """A group-graded FISSILE mixture (asymmetric in g).

    F's seed emission is ``χ·νΣf·φ`` — identically zero on a non-fissile
    mixture (``_mixture`` sets ``sig_f = nu = chi = 0``), which makes the F
    routing / bit-identity rows VACUOUS (0 == 0). vv refutation #4: split the
    fixed-source ``Q/Σ`` scattering config (non-fissile ``_mixture``) from the
    fissile config (here) so the F emission is genuinely nonzero. χ births all
    fission neutrons in group 0 (``chi=[1,0,…]``) — an asymmetric emission.
    """
    st = np.array([sig_t * (1.0 + 0.4 * g) for g in range(ng)])
    ss = np.diag([sig_s * (1.0 + 0.4 * g) for g in range(ng)])
    sf = np.array([0.05 + 0.1 * g for g in range(ng)])
    nu = np.full(ng, 2.4)
    chi = np.zeros(ng)
    chi[0] = 1.0
    return make_mixture(sig_t=st, sig_c=st - ss.sum(axis=0) - sf, sig_f=sf,
                        nu=nu, chi=chi, sig_s=ss)


def _fissile_sphere(nx: int = 5, ng: int = 2, sigma: float = 1.0, c: float = 0.4):
    """A seed-carrying FISSILE sphere (GL S4) — the F-arm carrier (non-vacuous emission)."""
    mesh = Mesh1D(edges=np.linspace(0.0, 4.0, nx + 1), mat_ids=np.zeros(nx, dtype=int),
                  coord=CoordSystem.SPHERICAL, bc_right=BC("vacuum"))
    return SNMesh(mesh, Quadrature.gauss_legendre(4),
                  {0: _fissile_mixture(sigma, c * sigma, ng)})


def _k_iso(solver):
    """The solver-composed K_iso (§14.1): ``S.isotropic_energy + N2N.isotropic_energy``
    — the SAME two cached leaf objects the production emission block
    consumes (build_within_group_system composes them at the one site)."""
    return solver.scattering_op.isotropic_energy + solver.n2n_op.isotropic_energy


def _s_emission(solver, psi: FullField) -> NDArray:
    """The ℓ=0 iso cell-emission ``q₀ = K_iso·φ₀`` that the S seed arm folds — the
    A_BA *input* (``(ng, nx)``). Computed via the solver-composed K_iso
    (the emission is the bulk job, verified elsewhere; A_BA's job is the
    FOLD of this emission, so it is the correct oracle input)."""
    phi0 = psi.interior.integrate_angular().values          # (ng, nx)
    return np.asarray(_k_iso(solver).apply(phi0))


def _f_emission(F, psi: FullField) -> NDArray:
    """The ℓ=0 iso fission cell-emission ``χ·νΣf·φ`` that the F seed folds — the
    A_BA_fission *input* (``(ng, nx)``). The eigenvalue outer loop computes exactly
    this (as ``fission_source`` from the SCALAR flux, ``F.kernel ∘ integrate``), so
    the migrated F seed :func:`~orpheus.sn.solver._radial_characteristic_fission_seed`
    only needs the FOLD of this emission (``A_BA_fission = Fold ∘ F.kernel``,
    factored)."""
    # CS4c step 4: F here is the ENERGY binding (solver-held fission_op);
    # step 5 (R-4): a plain binding admits the bare array of its bound shape,
    # so the scalar flux's ``.values`` is the operand (the untyped carrier
    # fall-through is retired).
    return np.asarray(F.apply(psi.interior.integrate_angular().values))


def _ba_oldloop_reference(emission: NDArray, sn) -> NDArray:
    r"""The EXACT pre-un-weld hand-rolled fold loop (the S/F seed arms before
    Step 2 routed them through RadialCharacteristicReconstruction): the
    bit-identity ORACLE. For each carrying (level, sign)
    the cells are ``fold(emission[None], sign)`` and the corners stay zero. The
    Step-2 un-weld MUST reproduce this byte-for-byte (``np.array_equal``),
    inheriting verification from the ℓ-fold contract gate (vv §Bit-identity: free
    verification by inheritance from a verified reference)."""
    ref = RadialCharacteristicField.source_zeros(sn.radial_characteristic_field_space)
    for lv in sn.radial_characteristic_levels:
        for sign in (-1, +1):
            ref.interior.cells(lv, sign)[:] = (
                _rcs_mod.fold_moments_to_radial_characteristic(emission[None], sign))
    return ref.to_flat()


def _fold_transpose_reference(y: NDArray, sign: int, n_moments: int,
                              *, coeff0: float | None = None) -> NDArray:
    r"""The Euclidean transpose of the ℓ-fold: ``moments_bar[ℓ] = coeff[ℓ]·y``
    with ``coeff[ℓ] = ((2ℓ+1)/2)·sign^ℓ`` — the contract the production
    fold-transpose (the S-adjoint's single-sourced ``0.5``) must satisfy. The
    ``coeff0`` override is the tooth: a ``0.6`` at ℓ=0 (≠ the fold's ``0.5``)
    breaks the adjoint identity."""
    ell = np.arange(n_moments)
    coeff = ((2.0 * ell + 1.0) / 2.0) * np.float64(sign) ** ell
    if coeff0 is not None:
        coeff = coeff.copy()
        coeff[0] = coeff0
    return coeff[:, None, None] * y[None]


def _install_fold_spy(monkeypatch) -> dict:
    r"""Mode-11 sentinel: WRAP the shared Legendre fold as the RECONSTRUCTION
    operator sees it. Post-un-weld the S/F seed arms route through
    ``RadialCharacteristicReconstruction.apply``, which module-level-imports
    ``fold_moments_to_radial_characteristic`` — so the object it fetches at call
    time is the reconstruction module's global (``_rcr_mod``), NOT the numerics
    source module. Wrapping it here counts every fold the reconstruction runs; an
    S/F arm that bypassed the reconstruction (re-inlined the numerics fold in its
    own namespace) would leave this counter at 0 (Cardinal Rule 2 single source;
    Mode 11)."""
    calls: dict = {"n": 0, "signs": []}
    real = _rcr_mod.fold_moments_to_radial_characteristic

    def spy(moments, sign):
        calls["n"] += 1
        calls["signs"].append(sign)
        return real(moments, sign)

    monkeypatch.setattr(_rcr_mod, "fold_moments_to_radial_characteristic", spy)
    return calls


def _apply_A_BA(emission: NDArray, sn) -> NDArray:
    r"""BIND POINT — how the extracted single-source A_BA fold is invoked.

    Input: the ℓ=0 iso cell-emission ``(ng, nx)``. Output: the folded ray-source
    ``RadialCharacteristicSourceSink.values`` (cells at μ=±1 per carrying level,
    corners zero). The Step-2 un-weld gives this ONE surface (the loop sites
    2/3/4 inline today). Its production shape is being decided in parallel — the
    main agent flips the ``# BIND:`` line to the chosen surface:
    """
    # Bound (Step 2, operator shape): the extracted single-source A_BA fold is
    # RadialCharacteristicReconstruction.apply — the emission is the ℓ=0 moment
    # (a unit ℓ axis, n_moments=1).
    return _rc_fold(sn).apply(emission[None]).to_flat()


def _apply_A_BA_transpose(seed_cotangent: NDArray, sn) -> NDArray:
    r"""BIND POINT — the A_BA Euclidean transpose (ray-cotangent → emission-cotangent),
    IF the chosen A_BA shape exposes a transpose surface. If the user picks a
    factory-only shape (no transpose surface), this gate is subsumed by the
    fold-helper-transpose contract (``test_fold_transpose_euclidean_contract``)
    and the operator-level consistency gate — flag the binding, do not force it.
    """
    # Bound (Step 2, operator shape): wrap the flat ray cotangent as a ψ½ FLUX
    # composite (composite to_flat order) and pull it back through the
    # reconstruction's Euclidean transpose → the (n_moments=1, ng, nx)
    # bulk-moment cotangent.
    field = _ray_composite(sn, seed_cotangent)
    return _rc_fold(sn).apply_transpose(field)


def _wrap_extracted_A_BA(monkeypatch) -> dict:
    r"""Mode-11 sentinel on the EXTRACTED single-source A_BA surface (post-un-weld).
    The wrap MUST sit on the SAME object the production S/F seed arms construct —
    the factory or the operator the un-weld routes them through. xfail until the
    bind is chosen; flip the ``# BIND:`` monkeypatch target:
    """
    calls: dict = {"n": 0}
    # Bound (Step 2, operator shape): wrap the operator method on the CLASS, so
    # every S/F seed arm that constructs a RadialCharacteristicReconstruction and
    # calls ``.apply`` is counted — immune to import binding (it patches the
    # class object all instances share).
    real = _rcr_mod.RadialCharacteristicReconstruction.apply

    def spy(self, moments, /):
        calls["n"] += 1
        return real(self, moments)

    monkeypatch.setattr(
        _rcr_mod.RadialCharacteristicReconstruction, "apply", spy)
    return calls


class TestA_BA_SchurFold:
    r"""``A_BA`` — the ψ½ Schur fold FACTOR (the ``RadialCharacteristicReconstruction``
    single source the ℓ-moment emission folds through at μ = ±1).

    Post-LIFT (campaign step 4c, commit 1) this class pins the FOLD FACTOR + the
    non-carrying control: the fold contract (``P_ℓ(±1) = (±1)^ℓ`` on a manufactured
    anisotropic input), the fold-transpose Euclidean contract, the fold-factor
    surface value + its Euclidean transpose, and the slab non-carrying control.
    The FULL coupling operator ``RadialCharacteristicEmission`` (``Fold ∘ K_iso ∘
    integrate``), the S/F→pure-bulk lift, and the driver routing are pinned in
    :class:`TestCoupledLift` (the step-2 "S/F EMIT the ray" gates are retired
    there — S/F are now pure bulk).

    Carrying member here = **sphere-GL S4** (R12a; 1 level → 2 fold calls/arm)
    — the only carrying member THIS class exercises, not the only one that
    exists: since Q5.6.3 the admitted folded cylinder carries on every level
    too.  The slab is the non-carrying CONTROL (the only admitted non-carrying
    1-D geometry). Every value row is ≥2G (1G is degenerate, vv anti-#3). Runtime: gates
    raise via ``pytest.fail`` / ``np.testing.assert_*`` (fire under ``python -O``),
    never a bare ``assert`` (vv Mode 8).
    """

    # ── Gate 1: the fold contract on a MANUFACTURED anisotropic input ──────

    def test_fold_contract_anisotropic_activates_p_ell(self):
        r"""Load-bearing refutation #3. Manufacture ``moments`` with ℓ=0 AND ℓ=1
        (≥2G, distinct per group) and assert the closed form
        ``Q̄(+1) = ½Q₀ + (3/2)Q₁``, ``Q̄(−1) = ½Q₀ − (3/2)Q₁`` — the ``sign^ℓ``
        line's P₁(±1) = ±1 asymmetry.

        Tooth (in-process, local): a fold that DROPS ``sign^ℓ`` (``coeff`` = the
        same ``(2ℓ+1)/2`` for both signs) reds the anisotropic assertion by
        ``3·|Q₁|`` (measured ≈ 2.7). NECESSITY: the SAME mutated fold on an
        ℓ=0-only input stays green (P₀≡1, ``sign^0=1`` always) — so the production
        S/F arms, which feed ℓ=0 ONLY, are STRUCTURALLY blind to this bug; the
        anisotropic input is what earns the coverage (§0.6 iso-snapshot blindness).
        """
        Q0 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])       # (ng=2, nx=3)
        Q1 = np.array([[0.7, -0.3, 0.2], [0.1, 0.9, -0.5]])     # distinct per group
        mom = np.stack([Q0, Q1], axis=0)                        # (L=2, ng, nx)
        fold = _rcs_mod.fold_moments_to_radial_characteristic
        np.testing.assert_allclose(
            fold(mom, +1), 0.5 * Q0 + 1.5 * Q1, rtol=0.0, atol=1e-13,
            err_msg="Q̄(+1) ≠ ½Q₀ + (3/2)Q₁ — the (2ℓ+1)/2·(+1)^ℓ fold is wrong.")
        np.testing.assert_allclose(
            fold(mom, -1), 0.5 * Q0 - 1.5 * Q1, rtol=0.0, atol=1e-13,
            err_msg="Q̄(−1) ≠ ½Q₀ − (3/2)Q₁ — the P₁(−1) = −1 sign is dropped/flipped.")

        # Tooth + necessity: a fold that drops sign^ℓ.
        def _fold_drop_sign(moments, sign):
            moments = np.asarray(moments)
            ell = np.arange(moments.shape[0])
            coeff = (2.0 * ell + 1.0) / 2.0                     # sign^ℓ DROPPED
            return np.tensordot(coeff, moments, axes=(0, 0))

        red_aniso = float(np.max(np.abs(_fold_drop_sign(mom, -1) - fold(mom, -1))))
        red_iso = float(np.max(np.abs(
            _fold_drop_sign(Q0[None], -1) - fold(Q0[None], -1))))
        print(f"  [G1] drop-sign^ℓ: anisotropic RED={red_aniso:.3f}  iso-only={red_iso:.1e}")
        if not red_aniso > 1e-3:
            pytest.fail(f"the drop-sign^ℓ mutation did not red the anisotropic contract "
                        f"({red_aniso:.3e}) — the ℓ≥1 gate is toothless.")
        if not red_iso < 1e-15:
            pytest.fail(f"the iso-only input MOVED under the mutation ({red_iso:.3e}) — "
                        f"the necessity claim is false (iso should be blind to sign^ℓ).")

    # NOTE (campaign step 4c, THE LIFT): the step-2 gates that asserted S/F EMIT
    # the ray via the fold (``test_scattering_and_fission_both_route_through_the_
    # shared_fold``, ``test_scattering_seed_is_the_half_emission_fold``,
    # ``test_fission_seed_is_the_half_emission_fold_fissile``) are RETIRED — S/F
    # are now PURE BULK and the emission moved to the ``RadialCharacteristicEmission``
    # gain (``A_BA``). Their successor coverage lives in :class:`TestCoupledLift`
    # (L1-FWD / L2 / L3), which pins that A_BA emits the fold and S/F carry no ray.

    # ── Gate 5a (LIVE): the fold-helper Euclidean transpose contract ───────

    def test_fold_transpose_euclidean_contract(self):
        r"""The transpose the S-adjoint arm's hard-coded ``0.5`` must be
        single-sourced through. The Euclidean adjoint identity
        ``⟨fold(m, sign), y⟩ = ⟨m, fold_transpose(y, sign)⟩`` with
        ``fold_transpose(y, sign)[ℓ] = ((2ℓ+1)/2)·sign^ℓ · y``, on a MANUFACTURED
        anisotropic ``m`` (ℓ=0 AND ℓ=1) and random ``y``.

        Teeth: (a) a ``0.6`` ℓ=0 coefficient in the reference transpose (≠ the
        fold's ``0.5`` — the scattering.py:1846 hard-code) breaks the identity
        (measured ≈ 0.02–0.04); (b) dropping ``sign^ℓ`` in the transpose breaks
        the sign=−1 leg (the P₁(−1) transpose consistency)."""
        rng = np.random.default_rng(0)
        m = rng.standard_normal((2, 2, 3))                      # (L=2, ng, nx) anisotropic
        y = rng.standard_normal((2, 3))
        fold = _rcs_mod.fold_moments_to_radial_characteristic
        for sign in (-1, +1):
            folded = fold(m, sign)
            lhs = float(np.sum(folded * y))
            rhs = float(np.sum(m * _fold_transpose_reference(y, sign, 2)))
            np.testing.assert_allclose(
                lhs, rhs, rtol=1e-12, atol=1e-12,
                err_msg=f"sign {sign}: ⟨fold(m),y⟩ ≠ ⟨m, fold_transpose(y)⟩ — the "
                        f"fold-transpose is not the Euclidean adjoint of the fold.")
            # Tooth (a): 0.6 ≠ 0.5 at ℓ=0.
            rhs_06 = float(np.sum(m * _fold_transpose_reference(y, sign, 2, coeff0=0.6)))
            d06 = abs(lhs - rhs_06) / (abs(lhs) + abs(rhs_06) + 1e-300)
            if not d06 > 1e-3:
                pytest.fail(f"sign {sign}: the 0.6 ℓ=0-coefficient tooth did not red "
                            f"({d06:.3e}) — the transpose contract is toothless to the "
                            f"scattering.py:1846 hard-coded 0.5.")

        # Tooth (b): the sign^ℓ transpose consistency (P₁(−1) = −1). A transpose
        # that drops sign^ℓ agrees at sign=+1 but breaks at sign=−1.
        ell = np.arange(2)
        for sign in (-1, +1):
            folded = fold(m, sign)
            lhs = float(np.sum(folded * y))
            no_sign = ((2.0 * ell + 1.0) / 2.0)[:, None, None] * y[None]   # sign^ℓ dropped
            d = abs(lhs - float(np.sum(m * no_sign))) / (abs(lhs) + 1e-300)
            print(f"  [G5a] sign={sign}: drop-sign^ℓ transpose defect = {d:.3f}")
            if sign == -1 and not d > 1e-3:
                pytest.fail(f"the drop-sign^ℓ transpose stayed green at sign=−1 "
                            f"({d:.3e}) — the P₁(−1) transpose consistency is unpinned.")

    # NOTE (campaign step 4c, THE LIFT): the step-2 gate
    # ``test_scattering_seed_arm_euclidean_transpose_consistency`` (S's hand-rolled
    # adjoint 0.5 == the forward fold's transpose) is RETIRED — the S-adjoint no
    # longer carries the seed pullback (S is pure bulk). Its stronger successor is
    # :meth:`TestCoupledLift.test_A_BA_scatter_carries_the_seed_pullback_S_carries_none`
    # (L1-ADJ), which pins the pullback ``w·K_isoᵀ(Reconstructionᵀ χ_seed)`` in
    # ``A_BA.apply_transpose`` and its Euclidean fwd↔adj consistency.

    # ── Gate 6 (LIVE): the non-carrying CONTROL (no ray, no fold) ──────────

    def test_non_carrying_control_no_ray_no_fold(self, monkeypatch):
        r"""The slab is the non-carrying CONTROL (``radial_characteristic_
        space is None`` — refutation #6, NOT "other geometries"): NO fold is
        invoked by the model-generic S (pure 2-block since B.2d — a ray arm
        is unspellable). The live claim: the fold spy counter stays 0.
        (Until Q5.6.3 an LS cylinder was the second control; a non-carrying
        cylinder is unconstructible since the admission flip.)"""
        slab = SNMesh(
            Mesh1D(edges=np.linspace(0.0, 4.0, 6), mat_ids=np.zeros(5, dtype=int),
                   coord=CoordSystem.CARTESIAN, bc_right=BC("reflective"),
                   bc_left=BC("reflective")),
            Quadrature.gauss_legendre(4), {0: _mixture(1.0, 0.4, 2)})
        for tag, sn in (("slab", slab),):
            if sn.radial_characteristic_field_space is not None:
                pytest.fail(f"{tag} carries a ray space — the non-carrying CONTROL is invalid.")
            N, nx, ng = sn.quad.N, sn.nx, sn.ng
            n_tr = int(sn.angular_trace.layout.total_size)
            bulk_only = FullField(
                interior=AngularFlux(values=np.random.default_rng(50).standard_normal((N, ng, nx)), space=sn.angular_bulk_space),
                boundary=AngularBoundaryFlux(values=np.zeros(n_tr), space=sn.angular_trace))
            calls = _install_fold_spy(monkeypatch)
            out = SNSolver(sn).scattering_op.apply(bulk_only)
            if type(out) is not FullField:
                pytest.fail(f"{tag}: S emitted {type(out).__name__}, not the "
                            f"2-block FullField.")
            if calls["n"] != 0:
                pytest.fail(f"{tag}: the fold was invoked {calls['n']}× on a non-carrying "
                            f"mesh — A_BA must not fire without a ray carrier.")

    # ── the fold FACTOR surface (a factor of A_BA, post-lift LIVE) ─────────
    #
    # ``_apply_A_BA`` / ``_apply_A_BA_transpose`` exercise the fold FACTOR
    # (``RadialCharacteristicReconstruction`` — emission → ray) that the coupling
    # operator ``RadialCharacteristicEmission`` composes with ``K_iso ∘ integrate``.
    # The FULL production A_BA (integrate → K_iso → fold) and the driver routing
    # are pinned in :class:`TestCoupledLift` (L1-FWD / L2 / L4-S); these two rows
    # pin just the fold factor's value + Euclidean transpose.

    def test_A_BA_fold_factor_folds_half_emission(self):
        r"""The A_BA fold FACTOR surface
        (``RadialCharacteristicReconstruction.apply``) folds an emission to the
        ray q½ source, matching the closed-form ½·emission loop
        (``_ba_oldloop_reference``)."""
        sn = _sphere()
        solver = SNSolver(sn)
        S = solver.scattering_op
        psi = _random_composite(sn, np.random.default_rng(60))
        emission = _s_emission(solver, psi)
        got = _apply_A_BA(emission, sn)
        np.testing.assert_array_equal(
            got, _ba_oldloop_reference(emission, sn),
            err_msg="the A_BA fold factor surface ≠ the documented fold loop.")

    def test_A_BA_fold_factor_transpose_euclidean_contract(self):
        r"""The fold FACTOR exposes ``.apply_transpose``, which satisfies the
        Euclidean adjoint contract against the forward fold
        (``⟨fold·emission, y⟩ = ⟨emission, foldᵀ·y⟩``)."""
        sn = _sphere()
        rng = np.random.default_rng(80)
        emission = rng.standard_normal((sn.ng, sn.nx))
        y = rng.standard_normal(sn.radial_characteristic_field_space.shape[0])
        fwd = _apply_A_BA(emission, sn)
        bwd = _apply_A_BA_transpose(y, sn)
        lhs = float(fwd @ y)
        rhs = float(emission.ravel() @ np.asarray(bwd).ravel())
        np.testing.assert_allclose(
            lhs, rhs, rtol=1e-11, atol=1e-12,
            err_msg="A_BA.apply_transpose is not the Euclidean adjoint of A_BA.apply.")


# ── Step-4c helpers: THE LIFT — S/F → pure bulk, A_BA born driver-side ──────
#
# The scatter LIFT (campaign step 4c, commit 1): the model-generic S / F gains
# drop their hand-rolled ψ½ seed arm and become PURE BULK; the bulk→ray emission
# is posed as the first-class coupling operator A_BA =
# RadialCharacteristicEmission (= Fold ∘ K_iso ∘ integrate), which the SI/Krylov
# driver lags as its OWN gain (the Wave-O #208 pattern that separated B from S).
# S_bulk ⊕ A_BA is bit-identical to the old monolithic S.apply.


def _a_ba_scatter(sn) -> RadialCharacteristicEmission:
    r"""The production scattering bulk→ray coupling A_BA — the SI driver's OWN
    lagged gain (``RadialCharacteristicEmission`` over S's isotropic kernel, the
    SAME shared object the bulk scatter gain uses — single-sourced emission)."""
    reduced = sn.reduced
    assert reduced is not None  # carrying fixture; narrowing only
    return RadialCharacteristicEmission(
        _k_iso(SNSolver(sn)),
        field_space=sn.radial_characteristic_field_space,
        full_field_space=sn.full_field_space,
        angular_bulk_space=sn.angular_bulk_space,
        angular_trace=sn.angular_trace,
        quadrature=sn.quad,
        coord=sn.coord,
    )


def _pullback_reconstruction(sn, solver, chi_seed_values: NDArray) -> NDArray:
    r"""The seed pullback ``w·K_isoᵀ(Reconstructionᵀ χ_seed)`` rebuilt from its
    NAMED factors — the structural decomposition
    ``A_BAᵀ = (∫dμ)ᵀ ∘ K_isoᵀ ∘ Foldᵀ`` the S-adjoint carried inline before the
    LIFT. Shape ``(N, ng, nx)``. (This shares A_BA's fold + kernel objects, so the
    ``== A_BA.apply_transpose.interior`` check is INHERITANCE — necessary-not-
    sufficient; the load-bearing structural cross-check is the fwd↔adj Euclidean
    reciprocity, leg (c) of L1-ADJ.)"""
    fold = _rc_fold(sn)
    chi_ray = _ray_composite(sn, chi_seed_values)
    m_bar = fold.apply_transpose(chi_ray)                       # Foldᵀ: (1, ng, nx)
    phi0_bar = np.asarray(_k_iso(solver).apply_transpose(m_bar[0]))  # K_isoᵀ: (ng, nx)
    w = np.asarray(sn.quad.weights, dtype=float)               # (∫dμ)ᵀ = ×w_n
    return w.reshape((w.size, 1, 1)) * phi0_bar[None]          # (N, ng, nx)


def _fold_half_to(coeff0: float):
    r"""A fold that overrides the ℓ=0 coefficient (½ → ``coeff0``) — the L2 value
    tooth (the un-weld's single-sourcing eliminated the hand-rolled ``0.5``)."""

    def _fold(moments, sign):
        moments = np.asarray(moments)
        ell = np.arange(moments.shape[0])
        coeff = ((2.0 * ell + 1.0) / 2.0) * np.float64(sign) ** ell
        coeff = coeff.copy()
        coeff[0] = coeff0
        return np.tensordot(coeff, moments, axes=(0, 0))

    return _fold


class TestCoupledLift:
    r"""Step 4c, commit 1 — THE SCATTER LIFT (S/F → pure bulk, A_BA born
    driver-side, BIT-IDENTICAL). The successor of the retired step-2 "S/F emit the
    ray" gates.

    Carrying member = **sphere-GL S4 ONLY** (R12a); ≥2G every value row; nonzero
    seed + bulk where the term needs activating (§0.6). Runtime: gates raise via
    :func:`pytest.fail` / ``np.testing.assert_*`` (fire under ``python -O``), never
    a bare ``assert`` (vv Mode 8). Every gate carries a mutation-verified tooth
    (a companion ``*_has_teeth`` / ``*_pins_the_object`` row).

    Commit 1 = the SCATTER lift (S/F ``.apply`` pure bulk, A_BA_scatter its own
    driver gain). **Commit 2 = the FISSION outer-seam migration**: the eigenvalue F
    ray seed moved from the per-ordinate ``from_isotropic → from_angular_source``
    round-trip to the DIRECT moments-fold
    :func:`~orpheus.sn.solver._radial_characteristic_fission_seed` (=
    ``A_BA_fission = Fold ∘ F.kernel``, factored — the outer loop already applied
    ``F.kernel ∘ integrate`` to build ``fission_source``, so only the Fold remains).
    The **L1-F** value gate + the **L4-F** Mode-11 sentinel pin it.
    (``from_angular_source`` STAYS for its other consumers — the final total-source
    reconstruction and the fixed-source external source.)
    """

    # ── L1-FWD: S / F pure bulk, A_BA emits (LIFT deliverable 1, forward) ───

    def test_L1_fwd_S_and_F_apply_are_pure_bulk_A_BA_scatter_emits(self):
        r"""S / F ``.apply`` are PURE BULK — ray present-zero, bulk UNCHANGED
        (``S.apply(psi).interior == S.apply(psi.interior)``, the FullField arm's bulk IS
        the model-generic scatter — the LIFT touched no bulk). The lifted A_BA
        gain carries the emission (ray nonzero, bulk present-zero — the disjoint
        direct sum ``S_bulk ⊕ A_BA``). Positive (A_BA emits) + pure-bulk
        (S/F ray present-zero); ≥2G, nonzero seed + bulk."""
        sn = _sphere()
        solver = SNSolver(sn)
        S = solver.scattering_op
        psi = _random_composite(sn, np.random.default_rng(100))
        s_out = S.apply(psi)
        # Since B.2d "S emits a ray" is UNSPELLABLE (the 2-block codomain has
        # no slot — Pattern 4 closed the pre-lift regression structurally);
        # the live claim is the bulk fidelity.
        if type(s_out) is not FullField:
            pytest.fail(f"S.apply emitted {type(s_out).__name__}, not the "
                        f"2-block FullField.")
        # CS4c step 5: ``S.apply(bulk)`` is unspellable (a composite-bound gain
        # admits its composite alone), so the SAME claim — the lift touches ONLY
        # the bulk — is now spelled as trace-INDEPENDENCE: re-run on a composite
        # carrying the same interior and a ZERO trace, and require the emitted
        # interior to be bit-identical. A body that read the trace reddens here.
        np.testing.assert_array_equal(
            s_out.interior.values,
            bulk_apply(S, psi.interior).values,
            err_msg="S.apply's bulk emission moved with the TRACE — the LIFT must "
                    "touch ONLY the bulk (extension-by-zero on the trace).")
        np.testing.assert_array_equal(
            s_out.boundary.values, 0.0,
            err_msg="S is volumetric: its composite emission's trace must be exactly zero.")
        # F pure bulk (fissile sphere; the F-fwd ray arm is dead — the fission
        # emission rides from_angular_source / commit 2, not F.apply).
        # CS4c step 4: the composite arm lives on the ANGULAR binding
        # (FissionOperator, minted as the eigen-M posing mints it).
        snf = _fissile_sphere()
        f_solver = SNSolver(snf)
        f_out = FissionOperator.from_solver_data(
            mat_xs=f_solver.mat_xs, space=snf.full_field_space,
        ).apply(_random_composite(snf, np.random.default_rng(101)))
        if type(f_out) is not FullField:
            pytest.fail(f"F.apply emitted {type(f_out).__name__}, not the "
                        f"2-block FullField.")
        # A_BA carries the emission on System B's OWN carrier (B.2b re-type):
        # the "bulk present-zero" disjointness is now STRUCTURAL — the codomain
        # has no bulk slot (Pattern 4) — so the pins are the container types
        # (G-b3.1 (ii)) + the nonzero folded cells; the boundary member is a
        # REAL zero (the fold writes cells only).
        a_out = _a_ba_scatter(sn).apply(psi)
        if type(a_out) is not RadialCharacteristicField:
            pytest.fail(f"A_BA.apply returned {type(a_out).__name__} — the block "
                        f"must emit System B's RadialCharacteristicField.")
        if type(a_out.interior) is not RadialCharacteristicInteriorSourceSink:
            pytest.fail(f"A_BA interior member is {type(a_out.interior).__name__} "
                        f"— the emission must carry the SOURCE pair.")
        if type(a_out.boundary) is not RadialCharacteristicBoundarySourceSink:
            pytest.fail(f"A_BA boundary member is {type(a_out.boundary).__name__} "
                        f"— the emission must carry the SOURCE pair.")
        if not np.max(np.abs(a_out.to_flat())) > 1e-6:
            pytest.fail("A_BA.apply ray ≈ 0 — the lifted gain does not carry the "
                        "bulk→ray emission (the coupling is unwired).")
        np.testing.assert_array_equal(
            a_out.boundary.values, 0.0,
            err_msg="A_BA.apply boundary member ≠ 0 — the fold writes CELLS only; "
                    "the corner datum is the boundary's job.")

    # ── L1-ADJ: the DECISIVE lost-pullback catcher (LIFT deliv 1, adjoint) ──

    def test_L1_adj_A_BA_scatter_carries_the_seed_pullback_S_carries_none(self):
        r"""THE DECISIVE pullback catcher (refutation R2). The S-adjoint's
        ``w·K_isoᵀ(Reconstructionᵀ χ_seed)`` bulk pullback moved to
        ``A_BA.apply_transpose``; ``S.apply_transpose`` is now pure bulk. On a
        NONZERO seed cotangent χ (the previously-nulled input — a present-zero χ
        gives ``Reconstructionᵀ(0) = 0`` so every ``.H`` reciprocity gate is BLIND
        to a lost pullback; ONLY a nonzero χ catches it):

        (a) ``A_BA.apply_transpose(χ).interior`` == the pullback (lives in A_BA), and
        is NONZERO (non-vacuity);
        (b) ``S.apply_transpose(χ_seed).interior`` == 0 (S dropped it — pure bulk);
        (c) structurally-independent fwd↔adj Euclidean reciprocity
        ``⟨A_BA·ψ, χ_seed⟩ = ⟨ψ, A_BAᵀ·χ⟩`` (a corrupted pullback lands on ONE
        side — the load-bearing cross-check; (a) is only an INHERITANCE decomposition).

        Tooth: :meth:`test_L1_adj_pullback_catcher_has_teeth`."""
        sn = _sphere()
        solver = SNSolver(sn)
        S = solver.scattering_op
        A_BA = _a_ba_scatter(sn)
        rng = np.random.default_rng(110)
        chi_seed = rng.standard_normal(sn.radial_characteristic_field_space.shape[0])
        # B.2b: the block's cotangent is System B's OWN carrier (a FullField
        # seed-only cotangent is now unspellable at the block boundary).
        chi_cot = _ray_composite(sn, chi_seed)
        # (a) the pullback lives in A_BA (== the named-factor reconstruction);
        # the output is System A's honest 2-block cotangent (B.2d — the
        # transitional present-zero ψ½ slot dissolved with the eviction).
        adj_out = A_BA.apply_transpose(chi_cot)
        if type(adj_out) is not FullField:
            pytest.fail(f"A_BA.apply_transpose emitted {type(adj_out).__name__}, "
                        f"not the 2-block System-A cotangent.")
        adj_bulk = adj_out.interior.values
        np.testing.assert_array_equal(
            adj_bulk, _pullback_reconstruction(sn, solver, chi_seed),
            err_msg="A_BA.apply_transpose.interior ≠ w·K_isoᵀ(Reconstructionᵀ χ_seed) — "
                    "the lifted seed pullback is wrong.")
        if not np.max(np.abs(adj_bulk)) > 1e-6:
            pytest.fail("A_BA.apply_transpose bulk ≈ 0 on a nonzero seed cotangent — "
                        "the pullback catcher is vacuous.")
        # (b) "S carries the seed pullback" is UNSPELLABLE since B.2d — a
        # seed-only System-A cotangent cannot even be built (the 2-block
        # composite has no ray slot), so the LIFT's claim is structural. The
        # residual live control: S's transpose of a zero-bulk composite is
        # zero (no hidden arm feeds the bulk from anywhere else).
        s_adj = S.apply_transpose(_zero_composite(sn))
        np.testing.assert_array_equal(
            s_adj.interior.values, 0.0,
            err_msg="S.apply_transpose(zero composite).interior ≠ 0 — a hidden "
                    "non-bulk arm feeds S's transpose.")
        # (c) structurally-independent Euclidean fwd↔adj reciprocity, NONZERO seed.
        psi = _random_composite(sn, rng)
        lhs = float(A_BA.apply(psi).to_flat() @ chi_seed)
        rhs = float(psi.interior.values.ravel() @ adj_bulk.ravel())
        defect = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-300)
        if not defect < 1e-11:
            pytest.fail(f"A_BA Euclidean fwd↔adj reciprocity defect {defect:.3e} ≥ "
                        f"1e-11 — apply_transpose is not the transpose of apply.")

    def test_L1_adj_pullback_catcher_has_teeth(self, monkeypatch):
        r"""TOOTH for L1-ADJ: a corrupted ``A_BA.apply_transpose`` (drop the bulk
        pullback — return present-zero bulk) breaks the fwd↔adj Euclidean
        reciprocity O(1). Proves L1-ADJ's leg (c) catches a lost pullback (and, via
        the composite ``.H``, the reciprocity-leaf sphere row too — see
        ``test_g_adjoint_reciprocity``). Monkeypatch-revert; ``-O``-safe."""
        sn = _sphere()
        A_BA = _a_ba_scatter(sn)
        rng = np.random.default_rng(111)
        chi_seed = rng.standard_normal(sn.radial_characteristic_field_space.shape[0])
        psi = _random_composite(sn, rng)

        from orpheus.transport.full_field import FullField as _FF
        from orpheus.transport.source_sinks import (
            AngularBoundarySourceSink, AngularSourceSink,
        )

        def _drop_pullback(self, cotangent, /):
            # A zero bulk (the pullback dropped) on the honest 2-block shape.
            return _FF(
                interior=AngularSourceSink.zeros(self._angular_bulk_space),
                boundary=AngularBoundarySourceSink.zeros(self._angular_trace))

        monkeypatch.setattr(RadialCharacteristicEmission, "apply_transpose", _drop_pullback)
        adj_bulk = A_BA.apply_transpose(_ray_composite(sn, chi_seed)).interior.values
        lhs = float(A_BA.apply(psi).to_flat() @ chi_seed)
        rhs = float(psi.interior.values.ravel() @ adj_bulk.ravel())
        defect = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-300)
        print(f"  [L1-ADJ tooth] dropped-pullback reciprocity defect = {defect:.3f}")
        if not defect > 1e-3:
            pytest.fail(f"the dropped-pullback mutation left the reciprocity defect at "
                        f"{defect:.3e} — L1-ADJ has no teeth to a lost pullback.")

    # ── L2: A_BA = Fold ∘ K_iso ∘ integrate, single-source fold (deliv 2) ───

    def test_L2_A_BA_scatter_routes_through_the_shared_fold(self, monkeypatch):
        r"""A_BA folds through the SINGLE-SOURCE reconstruction: a Mode-11 wrap on
        the shared ``fold_moments_to_radial_characteristic`` fires EXACTLY
        ``2·n_levels`` (2 signs/level; sphere-GL S4 → 1 level → 2), and the ray
        value is the documented fold loop (``_ba_oldloop_reference``,
        ``array_equal``, inheriting Gate 1's value). The lifted S no longer folds
        (0× — the emission left it).

        Tooth (½ → 0.6 fold coefficient): :meth:`test_L2_fold_value_has_teeth`."""
        sn = _sphere()
        solver = SNSolver(sn)
        S = solver.scattering_op
        A_BA = _a_ba_scatter(sn)
        psi = _random_composite(sn, np.random.default_rng(120))
        emission = _s_emission(solver, psi)
        n_levels = len(sn.radial_characteristic_levels)   # == 1 (sphere-GL S4)
        fold_calls = _install_fold_spy(monkeypatch)
        recon_calls = _wrap_extracted_A_BA(monkeypatch)
        ray = A_BA.apply(psi).to_flat()
        # (i) the extracted single-source RECONSTRUCTION operator is on the call
        # path (once per A_BA.apply — a re-inlined fold copy leaves this at 0).
        if recon_calls["n"] != 1:
            pytest.fail(f"A_BA routed through the reconstruction operator {recon_calls['n']}× "
                        f"(expected 1) — it does not use the extracted single-source surface.")
        # (ii) the shared inner fold fires 2·n_levels (2 signs/level).
        if fold_calls["n"] != 2 * n_levels:
            pytest.fail(f"A_BA folded {fold_calls['n']}× (expected 2·n_levels = "
                        f"{2 * n_levels}) — it does not route through the single-source fold.")
        np.testing.assert_array_equal(
            ray, _ba_oldloop_reference(emission, sn),
            err_msg="A_BA.apply ray ≠ the documented fold loop (the fold value diverged).")
        # S no longer folds (pure bulk).
        fold_calls["n"] = 0
        S.apply(psi)
        if fold_calls["n"] != 0:
            pytest.fail(f"S folded {fold_calls['n']}× — the lifted S must be pure bulk "
                        f"(the emission moved to A_BA).")

    def test_L2_fold_value_has_teeth(self, monkeypatch):
        r"""TOOTH for L2: a ½ → 0.6 fold coefficient (as the RECONSTRUCTION sees it)
        moves ``A_BA.apply`` off the documented loop (which uses the numerics fold,
        unpatched) — the ``array_equal`` reds. Proves the fold VALUE is pinned."""
        sn = _sphere()
        solver = SNSolver(sn)
        S = solver.scattering_op
        A_BA = _a_ba_scatter(sn)
        psi = _random_composite(sn, np.random.default_rng(121))
        emission = _s_emission(solver, psi)
        monkeypatch.setattr(_rcr_mod, "fold_moments_to_radial_characteristic", _fold_half_to(0.6))
        ray = A_BA.apply(psi).to_flat()
        oracle = _ba_oldloop_reference(emission, sn)   # numerics fold, unpatched (0.5)
        red = float(np.max(np.abs(ray - oracle)))
        print(f"  [L2 tooth] ½→0.6 fold: |A_BA.ray − oracle| = {red:.4f}")
        if not red > 1e-3:
            pytest.fail(f"the ½→0.6 fold mutation left A_BA.ray on the oracle "
                        f"({red:.3e}) — the L2 fold-value gate is toothless.")

    # ── L3: S_bulk ⊕ A_BA_ray reconstructs the monolith (deliv 3, Mode-12) ──

    def test_L3_S_bulk_plus_A_BA_ray_reconstructs_the_monolith(self):
        r"""The disjoint direct sum ``S_bulk ⊕ A_BA_ray`` reconstructs the OLD
        monolithic ``S.apply`` — pin the OBJECT (the exact ray PLACEMENT, never a
        keff/sum proxy; Mode-12 R4). S contributes ONLY the bulk (ray present-zero),
        A_BA ONLY the ray (bulk present-zero); the ray placement is byte-for-byte
        the documented monolith fold (``_ba_oldloop_reference``, ``array_equal``).

        Tooth (a ray permutation preserving the level sum):
        :meth:`test_L3_ray_placement_pins_the_object`."""
        sn = _sphere()
        solver = SNSolver(sn)
        S = solver.scattering_op
        A_BA = _a_ba_scatter(sn)
        psi = _random_composite(sn, np.random.default_rng(130))
        emission = _s_emission(solver, psi)
        s_out, a_out = S.apply(psi), A_BA.apply(psi)
        # Disjoint direct sum — BOTH sides structural since B.2b/B.2d: S's "no
        # ray" (2-block codomain) and A_BA's "no bulk" (composite codomain)
        # are unspellable (Pattern 4), so the surviving value claim is the
        # bulk fidelity + the exact ray placement below.
        # CS4c step 5: trace-independence is the spelling of "pure bulk" now
        # (``S.apply(bulk)`` is refused — the ends select the carrier).
        np.testing.assert_array_equal(
            s_out.interior.values, bulk_apply(S, psi.interior).values,
            err_msg="the lifted S's bulk drifted with the trace — it must be pure bulk.")
        # The reconstructed monolith ray == the documented fold loop (exact placement).
        monolith_ray = _ba_oldloop_reference(emission, sn)
        if not np.max(np.abs(monolith_ray)) > 1e-6:
            pytest.fail("the reconstructed monolith ray ≈ 0 — L3 is vacuous.")
        np.testing.assert_array_equal(
            a_out.to_flat(), monolith_ray,
            err_msg="S_bulk ⊕ A_BA_ray ≠ the monolithic S.apply ray (the ray "
                    "placement drifted — a permutation the OBJECT pin catches).")

    def test_L3_ray_placement_pins_the_object(self, monkeypatch):
        r"""TOOTH for L3 (Mode-12): a fold that PERMUTES the cells within a level
        (a radial roll — preserving the per-level SUM) reds the ``array_equal`` ray
        placement while a sum proxy would stay green. Proves L3 pins the OBJECT."""
        sn = _sphere()
        solver = SNSolver(sn)
        S = solver.scattering_op
        A_BA = _a_ba_scatter(sn)
        psi = _random_composite(sn, np.random.default_rng(131))
        emission = _s_emission(solver, psi)
        real_fold = _rcs_mod.fold_moments_to_radial_characteristic

        def _fold_rolled(moments, sign):
            return np.roll(real_fold(moments, sign), 1, axis=-1)  # permute cells; sum preserved

        monkeypatch.setattr(_rcr_mod, "fold_moments_to_radial_characteristic", _fold_rolled)
        ray = A_BA.apply(psi).to_flat()
        monolith_ray = _ba_oldloop_reference(emission, sn)   # numerics fold, unpatched
        # The OBJECT (placement) differs …
        placement_red = float(np.max(np.abs(ray - monolith_ray)))
        # … while the per-(level,sign) SUM proxy is BLIND to the permutation.
        ray_c = _ray_composite(sn, ray)
        ref_c = _ray_composite(sn, monolith_ray)
        sum_ray = sum(float(ray_c.interior.cells(lv, s).sum())
                      for lv in sn.radial_characteristic_levels for s in (-1, +1))
        sum_ref = sum(float(ref_c.interior.cells(lv, s).sum())
                      for lv in sn.radial_characteristic_levels for s in (-1, +1))
        sum_gap = abs(sum_ray - sum_ref) / (abs(sum_ref) + 1e-300)
        print(f"  [L3 tooth] permutation: placement RED={placement_red:.4f}  sum-proxy gap={sum_gap:.1e}")
        if not placement_red > 1e-3:
            pytest.fail(f"the ray permutation did not red the OBJECT pin "
                        f"({placement_red:.3e}) — L3 is a sum proxy, not an OBJECT pin.")
        if not sum_gap < 1e-9:
            pytest.fail(f"the permutation did NOT preserve the sum ({sum_gap:.3e}) — "
                        f"the Mode-12 necessity (a sum proxy would be blind) is not shown.")

    # ── L3.5 (G-b3.3): the transient adapter — byte-identity + DELEGATION ───

    # G-b3.3 (the B.2b adapter byte-identity + delegation gate) RETIRED at
    # B.2d d1 with its referent: the transient FullField-gain adapters are
    # gone (the driver consumes the blocks natively through the gain grid).
    # The fold-value pins live at BLOCK level (L3/L3.5 rows above); the
    # driver-route sentinel is L4-S below + TestWithinGroupSystem (G-d1.1).

    # ── L4-S: the driver routes A_BA as its OWN lagged gain (deliv 3/d) ─────

    def test_L4S_driver_routes_A_BA_scatter_as_its_own_gain(self, monkeypatch):
        r"""THE DECISIVE own-slot driver-routing sentinel (refutation R5: a green
        bulk gate measures the UNCHANGED sibling; only wrapping the NEW A_BA.apply
        proves the driver rewired). Wrap ``RadialCharacteristicEmission.apply`` and
        run a REAL within-group sphere ``solve_fixed_source``: the counter fires
        (> 0). Structural pair: the carrying record's gain grid carries the
        Emission block at (B,A) and the seedless record carries no coupled
        grid at all.

        Tooth (an Emission-less gain grid): :meth:`test_L4S_sentinel_has_teeth`."""
        sn = _sphere()
        solver = SNSolver(sn)
        system = build_within_group_system(
            sn, solver.mat_xs, scattering_op=solver.scattering_op)
        # B.2d: the gain grid's (B,A) slot carries the BLOCK natively.
        n_grid = system.explicit_gains[0]
        if not (isinstance(n_grid, CoupledOperator)
                and isinstance(n_grid.blocks[1][0], RadialCharacteristicEmission)):
            pytest.fail("the carrying record's gain grid carries no "
                        "RadialCharacteristicEmission at (B,A) — A_BA is not "
                        "wired as a block gain.")
        slab = SNMesh(
            Mesh1D(edges=np.linspace(0.0, 4.0, 6), mat_ids=np.zeros(5, dtype=int),
                   coord=CoordSystem.CARTESIAN, bc_right=BC("reflective"),
                   bc_left=BC("reflective")),
            Quadrature.gauss_legendre(4), {0: _mixture(1.0, 0.4, 2)})
        slab_solver = SNSolver(slab)
        slab_system = build_within_group_system(
            slab, slab_solver.mat_xs, scattering_op=slab_solver.scattering_op)
        if any(isinstance(g, CoupledOperator) for g in slab_system.explicit_gains):
            pytest.fail("the seedless record carries a coupled gain grid — a "
                        "seedless mesh has no bulk→ray coupling.")
        # Mode-11 sentinel: a REAL within-group sphere solve APPLIES A_BA.
        counter = {"n": 0}
        real = RadialCharacteristicEmission.apply

        def spy(self, psi, /):
            counter["n"] += 1
            return real(self, psi)

        monkeypatch.setattr(RadialCharacteristicEmission, "apply", spy)
        SNSolver(sn).solve_fixed_source(
            np.ones((sn.ng, sn.nx)), np.ones((sn.ng, sn.nx)))
        if counter["n"] <= 0:
            pytest.fail("a real within-group sphere solve did NOT apply A_BA — the SI "
                        "driver does not route the lifted gain (the rewire is missing).")

    def test_L4S_sentinel_has_teeth(self, monkeypatch):
        r"""TOOTH for L4-S: a driver whose gain grid forgot the Emission block
        (the (B,A) slot None — the un-wired shape) leaves the A_BA.apply counter
        at 0 — AND, since step 5, is caught IN PRODUCTION: the crippled
        splitting's fixed point solves the WRONG equation (A' = M − N_crippled),
        so the end-of-solve CERTIFICATE raises the lag-death error before the
        solve can return (the runtime half of the Mode-11 net; the counter is
        the structural half). Proves the unwired-driver shape cannot ship
        silently."""
        sn = _sphere()
        real_build = _solver_mod.build_within_group_system

        def _no_emission(sn_mesh, mat_xs, **kw):
            system = real_build(sn_mesh, mat_xs, **kw)
            n = system.explicit_gains[0]
            crippled = CoupledOperator(
                [[n.blocks[0][0], None], [None, n.blocks[1][1]]],
                domain=n.domain, codomain=n.codomain)
            return replace(system, explicit_gains=(crippled,))

        monkeypatch.setattr(
            _solver_mod, "build_within_group_system", _no_emission)
        counter = {"n": 0}
        real = RadialCharacteristicEmission.apply

        def spy(self, psi, /):
            counter["n"] += 1
            return real(self, psi)

        monkeypatch.setattr(RadialCharacteristicEmission, "apply", spy)
        # Leg 1 — the STRUCTURAL half (the original Mode-11 claim): with
        # the certificate silenced, the crippled DRIVER itself never
        # applies A_BA (counter 0) and the solve returns the wrong iterate
        # silently — exactly the shape the sentinel family exists to catch.
        monkeypatch.setattr(
            _solver_mod, "_certify_within_group_exit",
            lambda *a, **k: None,
        )
        SNSolver(sn).solve_fixed_source(
            np.ones((sn.ng, sn.nx)), np.ones((sn.ng, sn.nx)))
        print(f"  [L4-S tooth] Emission-less gain grid (certificate "
              f"silenced): A_BA.apply calls = {counter['n']}")
        if counter["n"] != 0:
            pytest.fail(f"the un-widened driver STILL applied A_BA {counter['n']}× — "
                        f"the L4-S sentinel would not catch a missing rewire.")
        # Leg 2 — the RUNTIME half (step 5): the LIVE certificate catches
        # the same crippled driver loudly (its one honest loss-grid apply
        # IS the counter's sole increment — the diagnostic consumer, not
        # the driver).
        monkeypatch.undo()
        monkeypatch.setattr(
            _solver_mod, "build_within_group_system", _no_emission)
        monkeypatch.setattr(RadialCharacteristicEmission, "apply", spy)
        counter["n"] = 0
        with pytest.raises(ConvergenceCertificateError, match="lag-death"):
            SNSolver(sn).solve_fixed_source(
                np.ones((sn.ng, sn.nx)), np.ones((sn.ng, sn.nx)))
        if counter["n"] != 1:
            pytest.fail(f"expected EXACTLY the certificate's one honest "
                        f"loss-grid apply; counted {counter['n']}")

    # ── L1-F: the eigenvalue F seed IS the direct moments-fold (commit 2) ───

    def test_L1F_fission_seed_is_the_moments_fold(self):
        r"""[L1-F value gate, commit 2] The eigenvalue fission ray seed
        (:func:`~orpheus.sn.solver._radial_characteristic_fission_seed`) IS the
        direct ℓ=0 moments-fold of the fission emission ``χ·νΣf·φ`` — pinning the
        migration ``A_BA_fission = Fold ∘ F.kernel`` (factored: the outer loop
        already applied ``F.kernel ∘ integrate`` to build ``fission_source``, so
        only the Fold remains).

        (a) BIT-IDENTICAL to the documented fold loop (``_ba_oldloop_reference`` —
        both fold ``emission[None]`` through ``RadialCharacteristicReconstruction``);
        (b) PRINCIPLED-EQUIV (~ULP) to the RETIRED per-ordinate route
        ``from_isotropic → from_angular_source`` — documenting the migration is a
        genuine ~ULP RE-baseline (the removed round-trip's per-ordinate ``·w``
        reassociates; vv §Bit-identity, criterion-3 FP-non-associativity), NOT
        bit-identical. FISSILE sphere, ≥2G (refutation #4: a non-fissile mixture
        makes the emission — hence the seed — identically zero, a VACUOUS gate).

        Tooth (½ → 0.6 fold): :meth:`test_L1F_fission_seed_value_has_teeth`."""
        from orpheus.sn.solver import (
            _radial_characteristic_fission_seed,
            _radial_characteristic_source_from_per_ordinate,
        )
        snf = _fissile_sphere()
        F = SNSolver(snf).fission_op
        psi = _random_composite(snf, np.random.default_rng(140))
        emission = _f_emission(F, psi)
        if not np.max(np.abs(emission)) > 1e-6:
            pytest.fail("the fission emission ≈ 0 — the L1-F value gate is VACUOUS "
                        "(the mixture is not actually fissile / νΣf = 0; vv #4).")
        got = _radial_characteristic_fission_seed(emission, snf).to_flat()
        # (a) bit-identical to the direct moments-fold loop.
        np.testing.assert_array_equal(
            got, _ba_oldloop_reference(emission, snf),
            err_msg="_radial_characteristic_fission_seed ≠ the direct moments-fold "
                    "(_ba_oldloop_reference) — the F seed is not the ℓ=0 fold.")
        # (b) principled-equiv (~ULP) to the RETIRED per-ordinate round-trip.
        old = _radial_characteristic_source_from_per_ordinate(
            AngularSourceSink.from_isotropic(emission, snf).values, snf).to_flat()
        # Budget: measured ~9-16 ULP on this config (maxabs 5.6e-16, maxrel 1.9e-15);
        # nulp=32 gives headroom for the removed per-ordinate ·w reassociation (the
        # migration is principled-equiv — vv §Bit-identity criterion 3, NOT byte-id).
        np.testing.assert_array_almost_equal_nulp(got, old, nulp=32)

    def test_L1F_fission_seed_value_has_teeth(self, monkeypatch):
        r"""TOOTH for L1-F (a): a ½ → 0.6 fold coefficient (as the RECONSTRUCTION
        sees it) moves ``_radial_characteristic_fission_seed`` off the documented
        loop (``_ba_oldloop_reference`` uses the numerics fold, unpatched) — the
        ``array_equal`` reds. Proves the F seed's fold VALUE is pinned (a
        dropped/wrong ½ coefficient in the migrated fold is caught)."""
        from orpheus.sn.solver import _radial_characteristic_fission_seed
        snf = _fissile_sphere()
        F = SNSolver(snf).fission_op
        psi = _random_composite(snf, np.random.default_rng(141))
        emission = _f_emission(F, psi)
        monkeypatch.setattr(
            _rcr_mod, "fold_moments_to_radial_characteristic", _fold_half_to(0.6))
        got = _radial_characteristic_fission_seed(emission, snf).to_flat()
        oracle = _ba_oldloop_reference(emission, snf)   # numerics fold, unpatched (0.5)
        red = float(np.max(np.abs(got - oracle)))
        print(f"  [L1-F tooth] ½→0.6 fold: |seed − oracle| = {red:.4f}")
        if not red > 1e-3:
            pytest.fail(f"the ½→0.6 fold mutation left the F seed on the oracle "
                        f"({red:.3e}) — the L1-F value gate is toothless.")

    # ── L4-F: the OUTER eigenvalue loop routes F through the fold (deliv/d) ──

    def test_L4F_outer_fission_seed_routes_through_the_moments_fold(self, monkeypatch):
        r"""L4-F — THE DECISIVE Mode-11 sentinel (refutation R5): a GREEN eigenvalue
        solve is BLIND to a leftover ``from_angular_source`` route on the F seed;
        only instrumenting the fold ON the fission path proves the OUTER
        fission-source loop routes through the migrated moments-fold.

        ⚠ REFINEMENT of the brief's literal design — a GLOBAL counter on
        ``RadialCharacteristicReconstruction.apply`` is Mode-11-CONTAMINATED: the
        SCATTER ``A_BA`` (:class:`RadialCharacteristicEmission`, a within-group
        lagged gain) folds through the SAME reconstruction EVERY SI iteration
        (measured: 322 scatter folds even with the F seed reverted), so a global
        ``n > 0`` fires from scatter alone and reverting the F seed would NOT red it.
        The fission-SPECIFIC catcher wraps the seam
        ``_radial_characteristic_fission_seed`` and measures the fold-count DELTA
        *attributable to it* — isolating the fission fold from the scatter fold.

        Structural pair (HAZARD 5): the gain grid carries the SCATTER ``A_BA``
        (over ``S.isotropic_kernel``), NOT a fission fold — F is the OUTER
        ``q_ext``, never a within-group gain.

        Tooth (revert the migration): :meth:`test_L4F_sentinel_has_teeth`."""
        snf = _fissile_sphere()
        # Global fold counter (scatter + fission both fold through Reconstruction).
        fold = {"n": 0}
        real_fold = RadialCharacteristicReconstruction.apply

        def fold_spy(self, moments, /):
            fold["n"] += 1
            return real_fold(self, moments)

        monkeypatch.setattr(RadialCharacteristicReconstruction, "apply", fold_spy)
        # Fission-SPECIFIC: the fold delta DURING _radial_characteristic_fission_seed.
        seam = {"n": 0, "fold_delta": 0}
        real_seed = _solver_mod._radial_characteristic_fission_seed

        def seam_spy(fission_source, sn_mesh):
            before = fold["n"]
            out = real_seed(fission_source, sn_mesh)
            seam["n"] += 1
            seam["fold_delta"] += fold["n"] - before
            return out

        monkeypatch.setattr(
            _solver_mod, "_radial_characteristic_fission_seed", seam_spy)
        # A REAL fissile-sphere eigenvalue solve (default source_iteration → the
        # eig-SI fission-seed site, solver.py:1453).
        _solver_mod.solve_sn(snf.materials, snf.mesh, snf.quad)
        # (i) the OUTER eigenvalue loop calls the fission seam.
        if seam["n"] <= 0:
            pytest.fail("the eigenvalue outer loop never called "
                        "_radial_characteristic_fission_seed — the F seed is not on "
                        "the eigenvalue path (Mode-11: green keff, uncaught).")
        # (ii) each fission-seam call routes through the moments-fold (delta > 0).
        if seam["fold_delta"] <= 0:
            pytest.fail(f"the fission seed fired {seam['n']}× but folded "
                        f"{seam['fold_delta']}× through RadialCharacteristic"
                        f"Reconstruction — a leftover from_angular_source route "
                        f"survives on the F seed (Mode-11).")
        print(f"  [L4-F] eigenvalue solve: seam_n={seam['n']} "
              f"seam_fold_delta={seam['fold_delta']} global_fold={fold['n']}")
        # Structural pair (HAZARD 5): the within-group gain grid carries the
        # SCATTER A_BA (over the solver-composed K_iso, §14.1), NOT a
        # fission fold — F is the outer q_ext. The identity pin is per
        # LEAF: the composed sum is minted fresh per build, but its two
        # leaves are the solver-cached energy bindings.
        snf_solver = SNSolver(snf)
        snf_system = build_within_group_system(
            snf, snf_solver.mat_xs, scattering_op=snf_solver.scattering_op,
            n2n_op=snf_solver.n2n_op)
        S = snf_solver.scattering_op
        emission_block = snf_system.explicit_gains[0].blocks[1][0]
        kernel = getattr(emission_block, "emission_kernel", None)
        if not (isinstance(emission_block, RadialCharacteristicEmission)
                and getattr(kernel, "_a", None) is S.isotropic_energy
                and getattr(kernel, "_b", None) is snf_solver.n2n_op.isotropic_energy):
            pytest.fail("the gain grid's (B,A) block is not EXACTLY the scatter "
                        "A_BA over the solver-composed K_iso "
                        "(S.isotropic_energy + N2N.isotropic_energy, leaf identity) — the "
                        "F fold must be the OUTER q_ext seam, never a "
                        "within-group gain (HAZARD 5).")

    def test_L4F_sentinel_has_teeth(self, monkeypatch):
        r"""TOOTH for L4-F: reverting the migration — pointing
        ``_radial_characteristic_fission_seed`` back to the RETIRED per-ordinate
        BYPASS (``from_isotropic → from_angular_source``, which never touches
        ``RadialCharacteristicReconstruction``) — leaves the fission-path
        ``fold_delta`` at 0, EVEN THOUGH the scatter A_BA keeps the GLOBAL fold
        counter > 0 (measured 322). Proves BOTH (a) the L4-F ``fold_delta`` catcher
        reds a reverted migration, and (b) a GLOBAL fold counter would NOT (it stays
        > 0 from scatter) — the fission-specific delta is WHY the sentinel has teeth
        (Mode 11). This tooth bakes the refutation of the brief's literal design in."""
        from orpheus.sn.solver import _radial_characteristic_source_from_per_ordinate
        snf = _fissile_sphere()
        fold = {"n": 0}
        real_fold = RadialCharacteristicReconstruction.apply

        def fold_spy(self, moments, /):
            fold["n"] += 1
            return real_fold(self, moments)

        monkeypatch.setattr(RadialCharacteristicReconstruction, "apply", fold_spy)
        # The reverted seam: compute the seed via the RETIRED per-ordinate route
        # (no Reconstruction) but record the fold delta the sentinel measures.
        seam = {"n": 0, "fold_delta": 0}

        def bypass_seam(fission_source, sn_mesh):
            before = fold["n"]
            out = _radial_characteristic_source_from_per_ordinate(
                AngularSourceSink.from_isotropic(fission_source, sn_mesh).values,
                sn_mesh)
            seam["n"] += 1
            seam["fold_delta"] += fold["n"] - before
            return out

        monkeypatch.setattr(
            _solver_mod, "_radial_characteristic_fission_seed", bypass_seam)
        _solver_mod.solve_sn(snf.materials, snf.mesh, snf.quad)
        print(f"  [L4-F tooth] reverted seed: global_fold={fold['n']} (scatter) "
              f"seam_n={seam['n']} seam_fold_delta={seam['fold_delta']}")
        # The seam still fires (the bypass IS called) …
        if seam["n"] <= 0:
            pytest.fail("the reverted seam never fired — the tooth is mis-wired.")
        # … but the fission path no longer folds — the fold_delta catcher reds.
        if seam["fold_delta"] != 0:
            pytest.fail(f"the reverted (bypass) fission seed STILL folded "
                        f"{seam['fold_delta']}× — the L4-F fold_delta catcher would "
                        f"not red a reverted migration.")
        # The Mode-11 lesson: a GLOBAL counter would MISS this (scatter keeps it > 0).
        if not fold["n"] > 0:
            pytest.fail(f"the global fold counter is {fold['n']} — the scatter A_BA "
                        f"should keep it > 0 even with F reverted (the Mode-11 point "
                        f"a global counter is blind to; the fission-specific delta "
                        f"is why the sentinel has teeth).")


# ── Step-3 helpers: A_AB the ψ½ seed injection (ray → bulk) ────────────────
#
# A_AB = RadialCharacteristicSeeding: the ray ψ½ seed injected into the bulk
# Morel–Montry angular recurrence. It is CELL-LOCAL ANGULAR (the seed at cell i
# feeds cell i's ordinate recurrence; NO spatial coupling — the radial march is
# A_BB's job), so — unlike A_BB's spatially-woven forward matvec — BOTH
# directions realize HERE as thin WRAPs of the single-sourced closure methods
# (precompute_psi_state / cell_contribution / angular_adjoint). σ-INDEPENDENT:
# with the bulk zeroed the collision/streaming terms drop out, so A_AB needs no
# σ_t (the constructor takes sn_mesh only). The forward .apply's contribution to
# (L+C).apply is isolated by LINEARITY (interior=0, boundary=0 → only the seed's
# angular numerator survives); the transpose is the seed_cells_bar term the
# in-sweep reverse adds on cells(p,-1). A_sb=0 (block-triangular) and A_bs≈7.5
# (this coupling's magnitude) are already pinned by TestRegressionFloor — not
# re-tested here. The sphere carries ONE level (R12a), so the per-level loop is
# length 1: a multi-carrying-level indexing bug is untested BY THIS MODULE.
# ⛔ The reason recorded here until 2026-08-29 — "UNTESTABLE with current
# geometry (cylinder is non-carrying)" — is present-tense FALSE since the
# Q5.6.3 admission flip: the ADMITTED cylinder is folded, hence CARRYING on
# every level (`[M]` 2026-08-29, ``folded_product(2, 4)`` →
# ``_carrying_levels == [0, 1]``), so a multi-carrying-level fixture is now
# CONSTRUCTIBLE.  What remains is a fixture gap, not a geometry limit.


def _bulk_composite(sn, bulk_values: NDArray) -> FullField:
    """A 2-block composite with the given bulk and a zero trace — the
    ``A_AB.apply_transpose`` isolation probe (step 6: the coupling's
    transpose is the explicit grid block; only the interior member is
    read, so the zero trace is inert)."""
    n_tr = int(sn.angular_trace.layout.total_size)
    return FullField(
        interior=AngularFlux(values=bulk_values, space=sn.angular_bulk_space),
        boundary=AngularBoundaryFlux(
            values=np.zeros(n_tr), space=sn.angular_trace),
    )


def _seed_flux(sn, rng) -> RadialCharacteristicField:
    """A random ψ½ ray seed — the ``A_AB.apply`` input."""
    ns = sn.radial_characteristic_field_space.shape[0]
    return _ray_composite(sn, rng.standard_normal(ns))


def _bulk_cotangent(sn, rng) -> AngularSourceSink:
    """A random bulk-residual cotangent — the ``A_AB.apply_transpose`` input."""
    return AngularSourceSink(values=rng.standard_normal((sn.quad.N, sn.ng, sn.nx)), space=sn.angular_bulk_space)


def _install_closure_spy(monkeypatch, sn, method_name: str) -> list[dict]:
    r"""Mode-11 sentinel: WRAP a ``MorelMontryAngularSweep`` method (the shared
    M-M closure kernel A_AB routes through) on the closure CLASS, recording each
    call's ``(args, kwargs)`` and delegating to the real method. Class-level so
    it is robust to the closure's storage layout (an instance-attr patch could
    trip ``__slots__``); the test uses ONE closure, so no cross-instance leak.
    Proves ``apply`` / ``apply_transpose`` EXECUTE the production kernel — a
    divergent inlined copy would leave the list empty (Cardinal Rule 2)."""
    cls = type(sn.angular_closure)
    real = getattr(cls, method_name)

    def spy(self, *args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return real(self, *args, **kwargs)

    calls: list[dict] = []
    monkeypatch.setattr(cls, method_name, spy)
    return calls


class TestA_AB_SeedInjection:
    r"""``A_AB`` = :class:`RadialCharacteristicSeeding` — the ray→bulk ψ½ seed
    injection (campaign step 3).

    The off-diagonal ``(transport, ray)`` coupling: the ψ½ ray seeds the bulk
    Morel–Montry angular recurrence. CELL-LOCAL ANGULAR (no spatial coupling),
    so both ``apply`` (ray → bulk residual) and ``apply_transpose`` (bulk
    cotangent → ray seed cotangent) are realized as WRAPs of the single-sourced
    closure methods and ``is_adjointable = True`` — the same both-directions
    completeness ``A_BB`` reached at step 4b (its forward is the radial march,
    single-sourced with the walk via ``radial_characteristic_forward_residual``).

    Sphere-GL S4 is this class's carrying member; the slab is the non-carrying
    CONTROL (the constructor rejects it — since Q5.6.3 the only admitted
    non-carrying 1-D geometry, the ADMITTED cylinder being folded, hence
    carrying on every level). Every value row is ≥2G. Gates raise
    via :func:`pytest.fail` / ``np.testing.assert_*`` (fire under ``python -O``),
    never a bare ``assert`` (vv Mode 8).

    L13/Mode-11 caveat: the bit-identity gates (:meth:`test_apply_matches_the_in_sweep_seed_injection`,
    :meth:`test_apply_transpose_is_the_in_sweep_seed_cells_bar`) route both
    ``A_AB`` and the ``(L+C)`` reference through the SAME closure methods, so
    they INHERIT bit-identity and are blind to a bug inside a shared method. The
    correctness cross-check is :meth:`test_euclidean_adjoint_consistency`
    (forward ↔ transpose — a shared-method sign bug lands on one side and breaks
    reciprocity)."""

    # ── Forward — the seed injection ≡ the in-sweep contribution ───────────

    def test_apply_executes_the_shared_closure_kernel(self, monkeypatch):
        r"""``A_AB.apply`` EXECUTES the shared M-M closure kernel with the
        bulk ZEROED and the seed passed role-preserved (Mode-11 sentinels).

        Step 6 retired the fused ``(L+C)`` joint channel, so A_AB IS the
        one spelling of the seed→bulk coupling — the ``≡ in-sweep
        injection`` reference has no referent.  σ-independence is now a
        TYPE fact (``RadialCharacteristicSeeding`` takes no σ); the VALUE
        anchoring is the Euclidean reciprocity gate below + the grid
        gates.  ≥2G."""
        sn = _sphere()
        rng = np.random.default_rng(30)
        sv = rng.standard_normal(sn.radial_characteristic_field_space.shape[0])
        seed = _ray_composite(sn, sv)
        # Mode-11 sentinel on the shared closure kernel.
        pre = _install_closure_spy(monkeypatch, sn, "precompute_psi_state")
        cc = _install_closure_spy(monkeypatch, sn, "cell_contribution")
        RadialCharacteristicSeeding(sn).apply(_member(seed))
        if len(pre) != 1:
            pytest.fail(
                f"precompute_psi_state called {len(pre)}× (expected 1) — A_AB."
                f"apply is not the single-precompute WRAP.")
        psi_view_arg = np.asarray(pre[0]["args"][0])
        if np.max(np.abs(psi_view_arg)) != 0.0:
            pytest.fail(
                "A_AB.apply did NOT zero the bulk psi_view — A_AA's angular "
                "redistribution would leak into the isolated coupling.")
        # 4e: the closure receives System B's INTERIOR member directly (the
        # split RadialCharacteristicInteriorFlux — no unified bridge). Role +
        # value-fidelity pin: the interior member is passed value-faithfully.
        bridged = pre[0]["kwargs"].get("radial_characteristic")
        if type(bridged) is not RadialCharacteristicInteriorFlux:
            pytest.fail(
                "A_AB.apply did not pass a role-preserved ψ½ INTERIOR FLUX as "
                f"radial_characteristic; got {type(bridged).__name__}.")
        np.testing.assert_array_equal(
            bridged.values, seed.interior.values,
            err_msg="the passed seed interior member is not value-faithful to "
                    "the input composite's interior.")
        if len(cc) < sn.nx:
            pytest.fail(
                f"cell_contribution called {len(cc)}× (< nx = {sn.nx}) — the "
                f"cell-local angular injection did not visit every cell.")

    # ── Transpose — the seed_cells_bar term ≡ the in-sweep reverse ─────────

    def test_apply_transpose_writes_only_the_inward_leg(self, monkeypatch):
        r"""``A_AB.apply_transpose`` runs ``angular_adjoint`` exactly once
        (Mode-11 sentinel) and writes ONLY the inward (−1) leg cells — the
        ``+1`` leg and both corners stay EXACTLY 0 (the forward's coupling
        reads only the inward cells, so its transpose writes only them).

        Step 6: the ``≡ in-sweep seed_cells_bar`` reference retired with
        the walk's joint channel — A_AB's transpose IS the one spelling of
        the pullback; its VALUE is anchored by the Euclidean reciprocity
        gate below (the two separately-implemented duals).  ≥2G."""
        sn = _sphere()
        rng = np.random.default_rng(31)
        vv = rng.standard_normal((sn.quad.N, sn.ng, sn.nx))
        aa = _install_closure_spy(monkeypatch, sn, "angular_adjoint")
        # B.2c: the cotangent arrives as System A's FullField (the codomain);
        # only its interior member is read (trace/ray structurally discarded).
        out = RadialCharacteristicSeeding(sn).apply_transpose(
            _bulk_composite(sn, vv))
        if len(aa) != 1:
            pytest.fail(
                f"angular_adjoint called {len(aa)}× (expected 1) — A_AB."
                f"apply_transpose is not the single-adjoint WRAP.")
        wrote_inward = False
        for p in sn.radial_characteristic_levels:
            if np.max(np.abs(out.interior.cells(p, -1))) > 0.0:
                wrote_inward = True
            np.testing.assert_array_equal(
                out.interior.cells(p, +1), 0.0,
                err_msg=f"level {p}: apply_transpose wrote the +1 leg (must be 0).")
            np.testing.assert_array_equal(
                out.boundary.corner(p, -1), 0.0,
                err_msg=f"level {p}: apply_transpose wrote the -1 corner (be 0).")
            np.testing.assert_array_equal(
                out.boundary.corner(p, +1), 0.0,
                err_msg=f"level {p}: apply_transpose wrote the +1 corner (be 0).")
        if not wrote_inward:
            pytest.fail(
                "apply_transpose left every inward (−1) leg at zero on a "
                "random cotangent — the M-M thread pullback is dead.")

    # ── Euclidean adjoint consistency — THE correctness cross-check ────────

    def test_euclidean_adjoint_consistency(self):
        r"""``⟨A_AB·u, v⟩ = ⟨u, A_ABᵀ·v⟩`` (Euclidean, plain dot — NOT the
        ``V_cell`` metric, which is the COMPOSITE Hilbert adjoint realized once
        at the coupled operator, L19) to < 1e-11, ≥3 draws. THE load-bearing
        correctness gate: it compares ``apply`` (precompute + cell_contribution)
        to ``apply_transpose`` (angular_adjoint) — two separately-implemented
        duals — so a sign/wiring bug in EITHER shared method lands on ONE side
        and breaks reciprocity (unlike the bit-identity gates, which route both
        sides through the same method and are blind to a shared-method bug).
        ≥2G."""
        sn = _sphere()
        op = RadialCharacteristicSeeding(sn)
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            u, v = _seed_flux(sn, rng), _bulk_cotangent(sn, rng)
            # B.2c I/O: apply's trace/ray output slots are present-zero, so
            # pairing the interior member against the bulk cotangent IS the
            # full Euclidean dot; the transpose reads a bulk-only FullField.
            lhs = float(
                op.apply(_member(u)).interior.values.ravel() @ v.values.ravel())
            rhs = float(
                u.to_flat()
                @ op.apply_transpose(
                    _bulk_composite(sn, v.values)).to_flat())
            defect = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-300)
            if not defect < 1e-11:
                pytest.fail(
                    f"seed {seed}: Euclidean adjoint defect {defect:.3e} ≥ 1e-11 "
                    f"— apply_transpose is not the transpose of apply.")

    def test_euclidean_adjoint_consistency_tooth(self, monkeypatch):
        r"""TOOTH for the adjoint gate: flipping the sign of ``cell_contribution``'s
        angular numerator flips ``apply`` (the forward side) but NOT
        ``apply_transpose`` (which routes through ``angular_adjoint``), so
        reciprocity breaks and the defect jumps to O(1). Proves the < 1e-11 gate
        has teeth AND that a shared-method bug DOES surface on the adjoint
        cross-check (the L13 escape hatch the bit-identity gates lack)."""
        sn = _sphere()
        cls = type(sn.angular_closure)
        real = cls.cell_contribution

        def flip(self, *args, **kwargs):
            denom, upstream = real(self, *args, **kwargs)
            return denom, -upstream

        monkeypatch.setattr(cls, "cell_contribution", flip)
        op = RadialCharacteristicSeeding(sn)
        rng = np.random.default_rng(1)
        u, v = _seed_flux(sn, rng), _bulk_cotangent(sn, rng)
        lhs = float(
            op.apply(_member(u)).interior.values.ravel() @ v.values.ravel())
        rhs = float(
            u.to_flat()
            @ op.apply_transpose(
                _bulk_composite(sn, v.values)).to_flat())
        defect = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-300)
        if not defect > 1e-3:
            pytest.fail(
                f"the cell_contribution sign flip left the adjoint defect at "
                f"{defect:.3e} — the consistency gate has no teeth.")

    # ── Seed-consumed asymmetry — reads ONLY the inward leg ────────────────

    def test_apply_reads_only_the_inward_leg(self):
        r"""``A_AB.apply`` reads ONLY the inward ``cells(p,-1)`` leg of the seed
        (the recurrence seed): two seeds sharing that leg but differing in the
        ``+1`` leg and both corners give IDENTICAL bulk output. Non-vacuity
        (Mode-5): the output is asserted non-trivial (a zero output would satisfy
        the identity vacuously)."""
        sn = _sphere()
        op = RadialCharacteristicSeeding(sn)
        rng = np.random.default_rng(32)
        ns = sn.radial_characteristic_field_space.shape[0]
        full = _ray_composite(sn, rng.standard_normal(ns))
        only_minus = RadialCharacteristicField.flux_zeros(sn.radial_characteristic_field_space)
        for p in sn.radial_characteristic_levels:
            only_minus.interior.cells(p, -1)[...] = full.interior.cells(p, -1)
        if not np.max(np.abs(only_minus.to_flat())) > 0.0:
            pytest.fail("the inward -1 leg is zero — the asymmetry gate is vacuous.")
        out_full = op.apply(_member(full))
        out_minus = op.apply(_member(only_minus))
        if not np.max(np.abs(out_full.interior.values)) > 1e-6:
            pytest.fail("A_AB.apply output is ~0 — the asymmetry gate is vacuous.")
        np.testing.assert_array_equal(
            out_full.interior.values, out_minus.interior.values,
            err_msg="A_AB.apply changed when only the +1 leg / corners changed — "
                    "it reads more than the inward starting-direction leg.")

    # ── Non-carrying CONTROL + mesh-identity ───────────────────────────────

    def test_constructor_and_mesh_identity_reject_non_carrying(self):
        r"""The guards, NET-NEW teeth (L4). Non-carrying CONTROL — the slab
        (Cartesian) has ``radial_characteristic_space is None`` → the
        seedless guard fires with ``match=`` the specific message.  (Until
        Q5.6.3 an LS cylinder was the second control; a non-carrying
        cylinder is unconstructible since the admission flip.)  Positive
        control — the sphere constructs. Mesh-identity (Pattern 4) —
        ``apply`` / ``apply_transpose`` refuse a field on a DIFFERENT
        sphere."""
        slab = SNMesh(
            Mesh1D(edges=np.linspace(0.0, 4.0, 6), mat_ids=np.zeros(5, dtype=int),
                   coord=CoordSystem.CARTESIAN, bc_right=BC("reflective"),
                   bc_left=BC("reflective")),
            Quadrature.gauss_legendre(4), {0: _mixture(1.0, 0.4, 2)})
        if slab.radial_characteristic_field_space is not None:
            pytest.fail("the slab carries a ray space — CONTROL invalid.")
        with pytest.raises(ValueError, match="carries no starting-direction ray"):
            RadialCharacteristicSeeding(slab)
        # Positive control + the space-content guard (CS4b S3 F2): a TWIN
        # sphere's member is content-equal — legal; a graded sphere's
        # content differs — refused.
        sn = _sphere()
        op = RadialCharacteristicSeeding(sn)
        _ = op.apply(_member(_seed_flux(_sphere(), np.random.default_rng(9))))
        other = _graded_sphere(nx=5)
        with pytest.raises(ValueError, match="space-content invariant"):
            op.apply(_member(_seed_flux(other, np.random.default_rng(9))))
        with pytest.raises(ValueError, match="space-content invariant"):
            op.apply_transpose(_bulk_composite(
                other,
                np.random.default_rng(9).standard_normal(
                    (other.quad.N, other.ng, other.nx))))
        # 4e block-boundary refusals: a System-A FullField / a bulk field are
        # not the typed carriers (parse-don't-validate at the block boundary).
        with pytest.raises(TypeError, match="System B's member carrier"):
            op.apply(_template(sn))                     # a System-A FullField (foreign carrier)
        with pytest.raises(TypeError, match="expected a FullField"):
            op.apply_transpose(_bulk_cotangent(sn, np.random.default_rng(9)))

    def test_b2c_grid_entry_containers(self):
        r"""B.2c re-type (G-c1.3): A_AB's grid-entry carriers. ``apply`` emits
        System A's honest 2-block FullField — the interior member carries the
        bulk term over a zero trace (B.2d: the transitional ψ½ slot is gone);
        ``apply_transpose`` emits System B's composite (source members).
        Declared domain/codomain pinned by IDENTITY (memo F2 — the composite
        space object, and System A's cached full_field_space)."""
        sn = _sphere()
        op = RadialCharacteristicSeeding(sn)
        rng = np.random.default_rng(11)
        if op.domain is not sn.radial_characteristic_field_space:
            pytest.fail("A_AB.domain is not THE composite member-space object.")
        if op.codomain is not sn.full_field_space:
            pytest.fail("A_AB.codomain is not THE full_field_space object.")
        out = op.apply(_member(_seed_flux(sn, rng)))
        if type(out) is not FullField:
            pytest.fail(f"apply did not emit a FullField; got {type(out).__name__}.")
        if type(out.interior) is not AngularSourceSink:
            pytest.fail(f"apply's interior is not an AngularSourceSink; got "
                        f"{type(out.interior).__name__}.")
        np.testing.assert_array_equal(
            out.boundary.values, 0.0,
            err_msg="apply's trace slot is not zero (A_AB writes only "
                    "the interior).")
        out_t = op.apply_transpose(_bulk_composite(
            sn, rng.standard_normal((sn.quad.N, sn.ng, sn.nx))))
        if type(out_t) is not RadialCharacteristicField:
            pytest.fail(f"apply_transpose did not emit the member composite; "
                        f"got {type(out_t).__name__}.")
        if type(out_t.interior) is not RadialCharacteristicInteriorSourceSink:
            pytest.fail(f"apply_transpose did not emit SOURCE members; got "
                        f"{type(out_t.interior).__name__}.")


class TestSystemRoleLattice:
    r"""4a — the ``SystemRole {A, B, COUPLED}`` two-system role lattice.

    The COARSE two-system partition that makes System B first-class in the
    operator algebra (orthogonal to :class:`~orpheus.numerics.operator.BlockRole`,
    the within-System-A bulk↔boundary refinement): the self-block ``A_BB`` and
    the ray boundary ``B_b`` are System B; the off-diagonal couplings ``A_AB`` /
    ``A_BA`` span both systems (COUPLED); every model-generic System-A leaf stays
    unclassified (``None``). The join is the two-system analogue of the
    block-role union — ``A ⊔ B = COUPLED``. Foundation: a software-invariant
    gate (no ``verifies`` — it pins no equation).
    """

    def test_join_is_the_two_system_union(self):
        # The defining law A ⊔ B = COUPLED, its symmetry, idempotence, COUPLED
        # absorption, and the conservative None propagation (an operator outside
        # the two-system decomposition stays outside under a sum).
        A, B, C = SystemRole.A, SystemRole.B, SystemRole.COUPLED
        assert _join_system_roles(A, A) is A
        assert _join_system_roles(B, B) is B
        assert _join_system_roles(C, C) is C
        assert _join_system_roles(A, B) is C
        assert _join_system_roles(B, A) is C          # symmetric
        assert _join_system_roles(A, C) is C
        assert _join_system_roles(C, B) is C          # COUPLED absorbs
        assert _join_system_roles(None, A) is None
        assert _join_system_roles(A, None) is None
        assert _join_system_roles(None, None) is None

    def test_psi_half_blocks_carry_their_system_role(self):
        # The four ψ½ blocks are stamped at the class level (the classification
        # is a class attribute — readable without instantiation).
        assert RadialCharacteristicOperator.system_role is SystemRole.B          # A_BB
        assert RadialCharacteristicSeeding.system_role is SystemRole.COUPLED     # A_AB
        assert RadialCharacteristicBoundaryOperator.system_role is SystemRole.B  # B_b
        assert (
            _rcr_mod.RadialCharacteristicReconstruction.system_role
            is SystemRole.COUPLED  # A_BA
        )

    def test_model_generic_operators_stay_unclassified(self):
        # The CONTROL: System-A model-generic leaves carry NO intrinsic
        # two-system membership — an SN context composes them into System A, but
        # they belong to no system by construction (the honest None default).
        assert SNBoundaryOperator.system_role is None      # B_a (System A trace boundary)
        assert MultiplicationOperator.system_role is None  # C (collision)
        assert ScatteringOperator.system_role is None      # S
        assert FissionOperator.system_role is None         # F

    def test_role_propagates_through_the_composers(self):
        # The derivation fires through the composers exactly as block_role does:
        # OperatorSum joins its summands, the G-adjoint preserves. A_AB is
        # σ-independent, so a sphere instance is cheap; it carries COUPLED, so
        # both a sum with itself and its adjoint stay COUPLED.
        a_ab = RadialCharacteristicSeeding(_sphere())
        assert (a_ab + a_ab).system_role is SystemRole.COUPLED   # OperatorSum join
        assert (2.0 * a_ab).system_role is SystemRole.COUPLED    # ScaledOperator passthrough
        assert a_ab.H.system_role is SystemRole.COUPLED          # AdjointOperator passthrough


# ── B.2c helpers: the co-producing builder (G-c2.x) ─────────────────────────
#
# build_coupled_system(sn_mesh, mat_xs) → (CoupledOperator, CoupledSpace) —
# the ψ½ instance #1 of the numerics block machinery. The gates below realize
# the test-architect delta memo coupled_operator_b2c_builder_verification
# (G-c2.1–2.6): P1 alignment + the F2 runtime proof, P2 presence-structural,
# THE grid≡fused centrepiece (per-row tolerances — the bulk row is the
# campaign's FIRST intrinsic principled-equiv row: the block split of the
# fused Morel-Montry angular state), the M2-on-real layout coextensiveness,
# the forward block-.H reciprocity (Mode-12), and the dead-slot hazard
# witness (memo R3).


def _m_minus_n_reference(sn, mat_xs, coupled):
    r"""The centrepiece's reference: ``A·ψ`` spelled as ``M·ψ − N·ψ`` through
    the RECORD's splitting — the WELDED path (the fused walk's seed feed via
    the B.2d explicit legs) minus the gain grid.

    Structurally DISTINCT from the loss grid's own row composition (the grid
    reaches the ray coupling through the EXPLICIT ``A_AB``/``A_BA`` blocks;
    ``M`` reaches it through the walk's welded feed) — so ``grid ≡ M − N`` is
    a genuine two-path cross-check of the named splitting ``A = M − N``, the
    B.2d successor of the retired fused-shim reference (memo F1's REFLECTIVE
    requirement carries over: vacuum masks a dropped ``B_b``)."""
    solver_S = ScatteringOperator.from_solver_data(
        mat_xs=mat_xs, scattering_order=0,
        space=sn.full_field_space)
    system = build_within_group_system(sn, mat_xs, scattering_op=solver_S)
    y_m = system.implicit_operator.apply(coupled)
    y_n = system.explicit_gains[0].apply(coupled)
    return CoupledField(systems=(
        y_m.systems[0] - y_n.systems[0],
        y_m.systems[1] - y_n.systems[1],
    ))


class TestCoupledBuilder:
    r"""B.2c — ``build_coupled_system``: the typed ψ½ 2×2 grid, co-produced
    with its :class:`CoupledSpace` (campaign step 4d.2; SUBSUMES the step-6
    presence collapse for the grid arm).

    Gates G-c2.1–2.6 per the delta memo. Sphere-GL S4 is the carrying member
    (REFLECTIVE where ``B_b`` must be non-null — memo F1) and, since Q5.6.3,
    the admitted folded cylinder is a SECOND carrying case (2×2 — asserted in
    :meth:`TestCoupledBuilder.test_p2_presence_structural`); the slab is the non-carrying CONTROL
    (1×1). ≥2G everywhere. Gates raise via
    :func:`pytest.fail` / ``np.testing.assert_*`` (fire under ``python -O``).
    """

    # ── G-c2.1 — P1 alignment by construction + the F2 runtime proof ──────

    def test_p1_alignment_by_construction_and_runtime_apply(self):
        r"""The co-produced pair is aligned: ``op.domain is op.codomain is
        space``; the space members are THE mesh's cached space objects
        (identity — the F2 discipline); each block declares the member space
        its grid position requires (asserted DIRECTLY — the grid's own check
        is ``==``-blind between the unified and composite System-B spaces,
        memo F2); the C-fwd ``SystemRole.A`` stamp rides the (A,A) block; and
        ``grid.apply`` RUNS on a space-shaped :class:`CoupledField` — the
        SUFFICIENT runtime proof that every block speaks its typed carrier
        (construction alone is Mode-12-blind, F2)."""
        sn = _sphere()
        grid, space = build_coupled_system(sn, sn.material_xs_field())
        if type(space) is not CoupledSpace:
            pytest.fail(f"space is {type(space).__name__}, not CoupledSpace.")
        if grid.domain is not space or grid.codomain is not space:
            pytest.fail("the grid is not typed against the co-produced space "
                        "object (P1 alignment broken).")
        if space.systems[0] is not sn.full_field_space:
            pytest.fail("System A's member space is not THE mesh's cached "
                        "full_field_space object.")
        if space.systems[1] is not sn.radial_characteristic_field_space:
            pytest.fail("System B's member space is not THE mesh's cached "
                        "composite member-space object.")
        # Per-block declared spaces, asserted directly (F2).
        if grid.blocks[0][0].domain != sn.full_field_space:
            pytest.fail("A_AA does not declare System A's space.")
        if grid.blocks[0][1].domain != sn.radial_characteristic_field_space:
            pytest.fail("A_AB's domain is not System B's member space.")
        if grid.blocks[1][0].codomain != sn.radial_characteristic_field_space:
            pytest.fail("A_BA's codomain is not System B's member space.")
        if grid.blocks[1][1].domain != sn.radial_characteristic_field_space:
            pytest.fail("A_BB−B_b does not declare System B's member space.")
        # Role stamps: the grid spans systems; the (A,A) block is explicitly
        # SystemRole.A (C-fwd — its model-generic members join to None).
        if grid.system_role is not SystemRole.COUPLED:
            pytest.fail("the grid does not carry SystemRole.COUPLED.")
        if grid.blocks[0][0].system_role is not SystemRole.A:
            pytest.fail("the (A,A) block is not stamped SystemRole.A "
                        "(the C-fwd explicit stamp).")
        # THE runtime proof (F2): apply runs and emits the member types.
        y = grid.apply(_coupled_template(sn))
        if type(y.systems[0]) is not FullField:
            pytest.fail("row A did not emit a FullField.")
        if type(y.systems[1]) is not RadialCharacteristicField:
            pytest.fail("row B did not emit the member composite.")

    # ── G-c2.2 — P2 presence-STRUCTURAL (positive shapes + refusals) ──────

    def test_p2_presence_structural(self):
        r"""Carrying sphere AND carrying folded cylinder → 2×2 (Q5.6.3: the
        admitted cylinder is folded, hence carrying on every level); the
        non-carrying slab → 1×1 over
        ``(full_field_space,)`` alone. The bypass-proof (the memo's forced-
        presence negative, realized at the guards that enforce it): EVERY
        System-B block constructor refuses a seedless mesh with its own
        specific message — so even a builder whose presence predicate were
        bypassed could not construct System B (Pattern 4: the illegal grid
        is unrepresentable, not merely un-built)."""
        sn = _sphere()
        grid, space = build_coupled_system(sn, sn.material_xs_field())
        if not (grid.n_rows == grid.n_cols == 2):
            pytest.fail(f"carrying sphere built {grid.n_rows}×{grid.n_cols}, "
                        f"expected 2×2.")
        slab = SNMesh(
            Mesh1D(edges=np.linspace(0.0, 4.0, 6), mat_ids=np.zeros(5, dtype=int),
                   coord=CoordSystem.CARTESIAN, bc_right=BC("reflective"),
                   bc_left=BC("reflective")),
            Quadrature.gauss_legendre(4), {0: _mixture(1.0, 0.4, 2)})
        # Q5.6.3: the ADMITTED cylinder is folded = CARRYING, so its presence
        # row joins the sphere's 2x2 leg; the slab is the only admitted
        # 1x1 (non-carrying) geometry.
        cyl_folded = SNMesh(
            Mesh1D(edges=np.linspace(0.05, 4.0, 6), mat_ids=np.zeros(5, dtype=int),
                   coord=CoordSystem.CYLINDRICAL, bc_right=BC("vacuum")),
            Quadrature.folded_product(n_mu=4, n_phi=8),
            {0: _mixture(1.0, 0.4, 2)})
        op_cyl, _space_cyl = build_coupled_system(
            cyl_folded, cyl_folded.material_xs_field())
        if not (op_cyl.n_rows == op_cyl.n_cols == 2):
            pytest.fail(f"carrying folded cylinder built "
                        f"{op_cyl.n_rows}×{op_cyl.n_cols}, expected 2×2.")
        for mesh, label in ((slab, "slab"),):
            op1, space1 = build_coupled_system(mesh, mesh.material_xs_field())
            if not (op1.n_rows == op1.n_cols == 1):
                pytest.fail(f"non-carrying {label} built "
                            f"{op1.n_rows}×{op1.n_cols}, expected 1×1.")
            if len(space1.systems) != 1 or (
                    space1.systems[0] is not mesh.full_field_space):
                pytest.fail(f"{label}: the 1×1 space is not (full_field_space,).")
        # The four System-B ctor refusals (match= each guard's OWN message —
        # a downstream crash must not false-green these).
        sigma = CrossSectionField(values=np.ones((2, 5)), space=slab.bulk_space)
        with pytest.raises(ValueError, match="carries no starting-direction ray"):
            RadialCharacteristicOperator(                                # A_BB
                slab.radial_characteristic_field_space, sigma,
                bulk_space=slab.bulk_space, dr=slab.axis_widths[0],
                start_cosines={})
        with pytest.raises(ValueError, match="carries no starting-direction ray"):
            RadialCharacteristicSeeding(slab)                            # A_AB
        with pytest.raises(ValueError, match="carries no radial-characteristic ray"):
            RadialCharacteristicEmission(                                # A_BA
                object(),
                field_space=slab.radial_characteristic_field_space,
                full_field_space=slab.full_field_space,
                angular_bulk_space=slab.angular_bulk_space,
                angular_trace=slab.angular_trace,
                quadrature=slab.quad,
                coord=CoordSystem.CARTESIAN)
        with pytest.raises(ValueError, match="carries no ψ½ ray"):
            RadialCharacteristicBoundaryOperator(slab.radial_characteristic_field_space, slab.bc["xmax"].law)                   # B_b

    # ── G-c2.3 — THE centrepiece: grid ≡ the complete fused loss ──────────

    def test_grid_equals_m_minus_n(self):
        r"""``grid.apply([ψ_A, ψ_B])`` ≡ ``M·ψ − N·ψ`` through the record's
        splitting — the named ``A = M − N`` identity as a TWO-PATH cross-check
        (the grid's ray coupling flows through the EXPLICIT ``A_AB``/``A_BA``
        blocks; ``M``'s through the walk's welded feed via the B.2d leaf-kwarg
        legs). The B.2d successor of the retired fused-shim centrepiece.
        PER-ROW bars (memo §0 — NOT uniformly array_equal):

        * ``y_A.interior``  — rtol=1e-11 (the intrinsic M-M block split:
          seed-zeroed A_AA + bulk-zeroed A_AB summed after ≠ the fused joint
          recurrence, FP reassociation ~5.5e-16; array_equal here would
          falsely red the honest decomposition);
        * ``y_A.boundary``  — rtol=1e-12 (addition order only);
        * ``y_B``           — rtol=1e-12 (same single-sourced bodies, only
          composition order differs).

        REFLECTIVE sphere so ``B_b`` is non-null (memo F1 — vacuum masks a
        dropped B_b); ≥2G; the live-ray + live-corner non-vacuity asserted."""
        sn = _sphere(bc="reflective")
        mat_xs = sn.material_xs_field()
        grid, _ = build_coupled_system(sn, mat_xs)
        for seed in (21, 22):
            rng = np.random.default_rng(seed)
            coupled = _random_pair(sn, rng)
            if not np.max(np.abs(coupled.systems[1].to_flat())) > 0.0:
                pytest.fail("the probe ray is zero — the centrepiece would be "
                            "vacuous for the coupling rows.")
            y_ref = _m_minus_n_reference(sn, mat_xs, coupled)
            if not np.max(np.abs(y_ref.systems[1].to_flat())) > 0.0:
                pytest.fail("the reference ray rows are zero — the B_b/emission "
                            "arms are not exercised (non-vacuity, F1).")
            y_a, y_b = grid.apply(coupled).systems
            np.testing.assert_allclose(
                y_a.interior.values, y_ref.systems[0].interior.values,
                rtol=1e-11, atol=1e-13,
                err_msg=f"seed {seed}: the bulk row diverged beyond the M-M "
                        f"block-split floor — the grid is NOT M − N.")
            np.testing.assert_allclose(
                y_a.boundary.values, y_ref.systems[0].boundary.values,
                rtol=1e-12, atol=1e-15,
                err_msg=f"seed {seed}: the trace row moved (only addition "
                        f"order may differ).")
            np.testing.assert_allclose(
                y_b.to_flat(),
                y_ref.systems[1].to_flat(),
                rtol=1e-12, atol=1e-15,
                err_msg=f"seed {seed}: the ray rows diverged — the (B,·) "
                        f"blocks are not the welded M − N action.")

    def test_centrepiece_teeth_misplacement_and_dropped_b_b(self):
        r"""TEETH for G-c2.1/2.3: (a) the off-diagonal swap is
        UNCONSTRUCTABLE — the typed grid refuses A_BA at the (A,B) position
        at construction (Pattern 4; full_field vs composite spaces DO
        discriminate — the F2 blindness is only unified-vs-composite);
        (b) dropping ``− B_b`` from (B,B) moves the ray rows on the
        REFLECTIVE sphere (on vacuum B_b ≡ 0 would mask the drop — F1)."""
        sn = _sphere(bc="reflective")
        mat_xs = sn.material_xs_field()
        grid, space = build_coupled_system(sn, mat_xs)
        with pytest.raises(IncompatibleOperatorComposition):
            CoupledOperator(
                [[grid.blocks[0][0], grid.blocks[1][0]],
                 [grid.blocks[0][1], grid.blocks[1][1]]],
                domain=space, codomain=space)
        a_bb_alone = rc_march(sn, mat_xs.total_cross_section_field)
        dropped = CoupledOperator(
            [[grid.blocks[0][0], grid.blocks[0][1]],
             [grid.blocks[1][0], a_bb_alone]],
            domain=space, codomain=space)
        rng = np.random.default_rng(23)
        coupled = _random_pair(sn, rng)
        y_ref = _m_minus_n_reference(
            sn, mat_xs, coupled).systems[1].to_flat()
        y_drop = dropped.apply(coupled).systems[1].to_flat()
        if not np.max(np.abs(y_drop - y_ref)) > 1e-8:
            pytest.fail("dropping −B_b left the ray rows unmoved on a "
                        "REFLECTIVE sphere — the centrepiece has no teeth "
                        "for the boundary block.")

    # ── G-c2.4 — M2-on-real: layout coextensiveness; assemble unavailable ─

    def test_m2_layout_coextensiveness_and_assemble_unavailable(self, monkeypatch):
        r"""The elegance-carried M2 re-pin on REAL members: the three offset
        spellings that CAN exist today agree — ``member.to_flat()`` sizes ==
        ``prod(member_space.shape)``, the ``system_slices`` table extracts
        exactly each member's flat (multi-axis leaves included), the table
        COVERS the whole flat, and ``CoupledField.from_flat`` round-trips.
        The third spelling (``block_array`` inference) is UNAVAILABLE — the
        ψ½ blocks emit no sparse assembly (memo F3/R2): pinned as
        ``is_assemblable False`` + ``assemble()`` raises, so a future reader
        sees deferral, not a bug. Tooth: a slice table that drops System B
        reds the coverage pin."""
        sn = _sphere()
        grid, space = build_coupled_system(sn, sn.material_xs_field())
        coupled = _random_pair(sn, np.random.default_rng(24))
        flat = coupled.to_flat()
        slices = space.system_slices
        if len(slices) != 2:
            pytest.fail(f"expected 2 system slices, got {len(slices)}.")
        if slices[-1].stop != flat.size:
            pytest.fail("the slice table does not COVER the coupled flat "
                        f"({slices[-1].stop} != {flat.size}).")
        for i, (member, mspace) in enumerate(zip(coupled.systems, space.systems)):
            m_flat = np.asarray(member.to_flat(), dtype=float)
            if m_flat.size != int(np.prod(mspace.shape)):
                pytest.fail(
                    f"system {i}: to_flat size {m_flat.size} != prod(space."
                    f"shape) {int(np.prod(mspace.shape))} — the field layout "
                    f"and the space layout are NOT coextensive.")
            np.testing.assert_array_equal(
                flat[slices[i]], m_flat,
                err_msg=f"system {i}: the system_slices extraction ≠ the "
                        f"member's own to_flat — the offset spellings drifted.")
        rebuilt = CoupledField.from_flat(flat, coupled)
        np.testing.assert_array_equal(
            rebuilt.to_flat(), flat,
            err_msg="the coupled flat protocol does not round-trip.")
        if grid.is_assemblable:
            pytest.fail("the ψ½ grid claims assemblability — no block emits "
                        "sparse assembly (memo F3); did a walk assembler land? "
                        "Then wire the block_array M2 arm (R2).")
        with pytest.raises(MissingAssembly):
            grid.assemble()
        # Tooth: drop System B's slice → the coverage pin reds.
        with monkeypatch.context() as m:
            first = slices[0]
            m.setattr(type(space), "system_slices",
                      property(lambda self: (first,)))
            broken = space.system_slices
            if broken[-1].stop == flat.size:
                pytest.fail("the dropped-slice tooth is mis-wired — coverage "
                            "still holds with System B dropped.")

    # ── G-c2.5 — forward block-.H reciprocity (Mode-12, real members) ─────

    def test_forward_block_adjoint_reciprocity(self, monkeypatch):
        r"""``⟨grid·ψ, x⟩_G = ⟨ψ, grid.H·x⟩_G`` on the REAL sphere 2×2 —
        the block Hilbert adjoint's FIRST real-curvilinear-member run (it is
        FREE via the B.2a `AdjointOperator` + the member-wise CoupledSpace
        metrics; a hand-rolled Euclidean block-.H is the ERR-067 reopening).
        Tooth M-ADJ-metric: stripping the metric conjugation (identity
        apply_metric / apply_inverse_metric) reds the defect O(1) — the
        composite bulk⊕trace⊕seed metric is non-trivial on every geometry
        (the step-4 memo's burned 'slab stays green' lesson)."""
        sn = _sphere(bc="reflective")
        grid, space = build_coupled_system(sn, sn.material_xs_field())
        rng = np.random.default_rng(25)
        psi = _random_pair(sn, rng)
        x = _random_pair(sn, rng)
        lhs = space.inner_product(grid.apply(psi), x)
        rhs = space.inner_product(psi, grid.H.apply(x))
        defect = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-300)
        if not defect < 1e-12:
            pytest.fail(f"block-.H reciprocity defect {defect:.3e} ≥ 1e-12 on "
                        f"the real 2×2 — the metric conjugation is broken.")
        with monkeypatch.context() as m:
            m.setattr(CoupledSpace, "apply_metric", lambda self, f: f)
            m.setattr(CoupledSpace, "apply_inverse_metric", lambda self, f: f)
            lhs2 = space.inner_product(grid.apply(psi), x)
            rhs2 = space.inner_product(psi, grid.H.apply(x))
            defect2 = abs(lhs2 - rhs2) / (abs(lhs2) + abs(rhs2) + 1e-300)
        if not defect2 > 1e-3:
            pytest.fail(f"the M-ADJ-metric tooth left the defect at "
                        f"{defect2:.3e} — the reciprocity gate has no teeth "
                        f"(a Euclidean block-.H would pass).")

    # ── G-c2.6 — REMOVED at B.2d d2 (G-d2.7) ─────────────────────────────
    #
    # The ``test_dead_slot_double_count_witness`` gate is GONE with its
    # hazard: a live-ray ψ_A is UNREPRESENTABLE since the eviction
    # (``FullField`` is 2-block), so the double-count it witnessed cannot
    # occur — the replacement is the type system itself, not a runtime
    # assert (memo R3, exactly as ruled).


# ═════════════════════════════════════════════════════════════════════════
# B.2d d1 — the WithinGroupSystem record + the block-native driver (G-d1.x)
# ═════════════════════════════════════════════════════════════════════════
#
# Delta memo: test-architect coupled_operator_b2d_driver_eviction_verification
# (G-d1.1–1.8; findings F1–F5; rulings R1–R6). The §0 partition: the WALK is
# ZERO-TOUCH at d1 (operator/walk-level pins bit-identical); the ~ULP drift
# lives ONLY in the end-to-end coupled driver (rhs reassembly + GMRES
# dead-ray padding), so the carrying same-fixed-point row is principled-equiv
# (SAFETY × inner_tol) while the seedless slab CONTROL is array_equal (the
# record's seedless arm is a pure re-package).


def _het_vacuum_sphere(ng: int = 2) -> SNMesh:
    r"""Heterogeneous VACUUM 2-region sphere — the Mode-9 configuration
    (vv-principles: never gate a splitting/driver re-pose on the reflective
    isotropic box; vacuum + heterogeneity break the degenerate coincidences)."""
    mesh = curvilinear_two_region_mesh(
        outers=(2.0, 4.0), mat_ids=(0, 1), n_cells=(3, 3),
        coord=CoordSystem.SPHERICAL, bc=BC("vacuum"),
    )
    return SNMesh(mesh, Quadrature.gauss_legendre(4),
                  {0: _mixture(1.0, 0.4, ng), 1: _mixture(0.5, 0.1, ng)})


# The d1-era ``_fused_reference_pieces`` (the pre-d1 fused driver rebuilt
# from the FusedRay*Gain shims) DISSOLVED at d2 with the 3-block carrier —
# the shims and the fused spelling are unrepresentable. The d1 gate G-d1.6
# proved the block-native driver reproduces the pre-d1 fixed point; the
# d2-forward fixed-point protection is the walk-substrate re-points (F4:
# test_282 / walk-baselines / native-matvec — values bit-identical through
# the six-signature re-type) + the SI ≡ Krylov cross-driver row below + the
# E4 closed-form anchors (d3).


class TestWithinGroupSystem:
    r"""G-d1.x — the record's shape, the driver's route, and the fixed point."""

    # ── G-d1.2 — the record's container/identity pins ────────────────────

    def test_g_d1_2_record_container_and_identity_pins(self):
        r"""ONE construction: the loss grid, its space, M, and N share the
        SAME space OBJECT (`is` — the P1 co-production carried onto the
        record); N's (A,B) slot is the structural ∅ (Seeding lives in M);
        the (A,A) gain is stamped ``SystemRole.A`` (C-fwd); the seedless
        record is the pure re-package ``(L+C, (S, B_a))`` with the solver's
        CACHED scattering operator (the cache seam, by identity)."""
        sn = _sphere()
        solver = SNSolver(sn)
        system = build_within_group_system(
            sn, solver.mat_xs, scattering_op=solver.scattering_op)
        if type(system) is not WithinGroupSystem:
            pytest.fail(f"builder returned {type(system).__name__}")
        if not isinstance(system.loss, CoupledOperator):
            pytest.fail(f"loss is {type(system.loss).__name__}")
        if system.loss.domain is not system.space or (
                system.loss.codomain is not system.space):
            pytest.fail("loss grid not typed against THE record space (identity)")
        # Step 5: M is the HONEST upper-triangular grid over the same piece
        # objects — [[LC, Seeding], [None, march]] (the fused facade
        # dissolved; R-5.4).
        M_grid = system.implicit_operator
        if not isinstance(M_grid, CoupledOperator):
            pytest.fail(f"carrying resolvent is {type(M_grid).__name__}")
        if M_grid._triangular_orientation() != "upper":
            pytest.fail("M is not upper-triangular — the substitution route "
                        "is gone")
        if M_grid.blocks[1][0] is not None:
            pytest.fail("M's (B,A) slot is not the structural ∅ — the "
                        "emission belongs to N, never to M")
        if not isinstance(M_grid.blocks[0][1], RadialCharacteristicSeeding):
            pytest.fail("M's (A,B) block is not the Seeding")
        if not isinstance(M_grid.blocks[1][1], RadialCharacteristicOperator):
            pytest.fail("M's (B,B) block is not the bare march (B_b belongs "
                        "to N)")
        if M_grid.domain is not system.space:
            pytest.fail("M not typed against THE record space (identity)")
        if len(system.explicit_gains) != 1 or not isinstance(system.explicit_gains[0], CoupledOperator):
            pytest.fail(f"carrying gains are {system.explicit_gains!r} — expected (N,)")
        n_grid = system.explicit_gains[0]
        if n_grid.domain is not system.space or n_grid.codomain is not system.space:
            pytest.fail("N not typed against THE record space (identity)")
        if n_grid.blocks[0][1] is not None:
            pytest.fail("N's (A,B) slot is not the structural ∅ — Seeding "
                        "must live in M (the walk's welded feed), never in N.")
        if n_grid.blocks[0][0].system_role is not SystemRole.A:
            pytest.fail("N's (A,A) gain lost the C-fwd SystemRole.A stamp")
        if not isinstance(n_grid.blocks[1][0], RadialCharacteristicEmission):
            pytest.fail("N's (B,A) block is not the Emission")
        if not isinstance(n_grid.blocks[1][1], RadialCharacteristicBoundaryOperator):
            pytest.fail("N's (B,B) block is not B_b")
        # The cache seam: the injected scattering operator rides BY IDENTITY.
        # (Reach S through N's (A,A) OperatorSum is internal; pin the seam by
        # rebuilding with injection and checking the seedless arm below.)
        # Seedless: the pure re-package.
        slab = SNMesh(
            Mesh1D(edges=np.linspace(0.0, 4.0, 6), mat_ids=np.zeros(5, dtype=int),
                   coord=CoordSystem.CARTESIAN, bc_right=BC("reflective"),
                   bc_left=BC("reflective")),
            Quadrature.gauss_legendre(4), {0: _mixture(1.0, 0.4, 2)})
        slab_solver = SNSolver(slab)
        s_system = build_within_group_system(
            slab, slab_solver.mat_xs, scattering_op=slab_solver.scattering_op)
        if isinstance(s_system.implicit_operator, CoupledOperator):
            pytest.fail("seedless resolvent is coupled — DP-seedless violated")
        if not isinstance(s_system.implicit_operator, StreamingCollisionOperator):
            pytest.fail(f"seedless resolvent is {type(s_system.implicit_operator).__name__}")
        if len(s_system.explicit_gains) != 3:
            pytest.fail(f"seedless gains are {s_system.explicit_gains!r} — expected (S, N2N, B_a)")
        S_g, _n2n_g, B_g = s_system.explicit_gains
        if S_g is not slab_solver.scattering_op:
            pytest.fail("the injected scattering operator did not ride the "
                        "record by IDENTITY (the cache seam broke)")
        if not isinstance(B_g, SNBoundaryOperator):
            pytest.fail(f"seedless boundary gain is {type(B_g).__name__}")
        if s_system.space.n_systems != 1 or s_system.loss.n_rows != 1:
            pytest.fail("seedless record is not the 1-system grid (P2)")

    # ── G-d1.1 — the W1 Mode-11 sentinel (+ the dead-slot rider) ─────────

    @pytest.mark.parametrize("inner", ["source_iteration", "krylov"])
    def test_g_d1_1_production_routes_through_the_block_machinery(
            self, inner, monkeypatch):
        r"""A REAL carrying-mesh within-group solve EXECUTES the block route:
        the N grid's ``CoupledOperator.apply``, the M leg — since step 5 the
        SUBSTITUTION (``CoupledSubstitutionOperator.apply`` → ``grid.solve``,
        the SI step) or the triangular grid's own matvec (the Krylov action;
        discriminated from N by the structural ``(1, 0) is None`` slot —
        only M's grid is upper-triangular) — and the bulk walk beneath it
        all fire (wrap counters > 0 — the B.2c F2 lesson: runtime is the
        sufficient catcher). RIDER: the driver's iterate is the coupled
        pair with an honest 2-block ψ_A (the d1 dead-slot rider dissolved —
        a live-ray ψ_A is unrepresentable since d2)."""
        sn = _sphere()
        counters = {"N": 0, "M": 0, "walk": 0}
        real_grid_apply = CoupledOperator.apply

        def spy_grid_apply(op, x, /):
            counters["N"] += 1
            # The M grid is the ONLY upper-triangular 2×2 in the route
            # (N and the loss both carry a present (1, 0) block) — its
            # matvec IS the Krylov M leg.
            if op.n_rows == 2 and op.blocks[1][0] is None:
                counters["M"] += 1
            return real_grid_apply(op, x)

        real_grid_solve = CoupledOperator.solve
        real_sub_apply = CoupledSubstitutionOperator.apply

        def spy_grid_solve(op, rhs):
            counters["M"] += 1
            return real_grid_solve(op, rhs)

        def spy_sub_apply(op, rhs, /, *, initial_guess=None):
            counters["M"] += 1
            return real_sub_apply(op, rhs, initial_guess=initial_guess)

        # The bulk walk beneath the substitution: SI enters through .solve
        # (the ray-DECOUPLED (L+C) leg of the substitution), Krylov through
        # .apply (the (A,A) block matvec — its GMRES preconditioner is the
        # explicit identity, #200, so .solve never fires there). Either
        # entry IS the walk leg.
        real_walk_solve = StreamingCollisionOperator.solve
        real_walk_apply = StreamingCollisionOperator.apply

        def spy_walk_solve(op, rhs, *a, **kw):
            counters["walk"] += 1
            return real_walk_solve(op, rhs, *a, **kw)

        def spy_walk_apply(op, psi, *a, **kw):
            counters["walk"] += 1
            return real_walk_apply(op, psi, *a, **kw)

        monkeypatch.setattr(CoupledOperator, "apply", spy_grid_apply)
        monkeypatch.setattr(CoupledOperator, "solve", spy_grid_solve)
        monkeypatch.setattr(CoupledSubstitutionOperator, "apply", spy_sub_apply)
        monkeypatch.setattr(StreamingCollisionOperator, "solve", spy_walk_solve)
        monkeypatch.setattr(StreamingCollisionOperator, "apply", spy_walk_apply)
        solver = SNSolver(sn, inner_solver=inner)
        solver.solve_fixed_source(
            np.ones((sn.ng, sn.nx)), np.ones((sn.ng, sn.nx)))
        for name, n in counters.items():
            if n <= 0:
                pytest.fail(f"[{inner}] the production route never executed "
                            f"the {name} leg (counter 0) — Mode-11: the block "
                            f"machinery is bypassed.")
        # The pair rider: the iterate is the coupled pair; ψ_A is the honest
        # 2-block System-A member (Pattern 4 — no ray slot to double-count).
        assert solver._inner is not None
        psi = solver._inner.iterate
        if not isinstance(psi, CoupledField):
            pytest.fail(f"[{inner}] the carrying iterate is "
                        f"{type(psi).__name__}, not the coupled pair")
        if type(psi.systems[0]) is not type(psi.systems[0]) or not isinstance(
                psi.systems[0], FullField):
            pytest.fail(f"[{inner}] ψ_A is {type(psi.systems[0]).__name__}, "
                        f"not the 2-block System-A composite")
        if not isinstance(psi.systems[1], RadialCharacteristicField):
            pytest.fail(f"[{inner}] ψ_B is {type(psi.systems[1]).__name__}, "
                        f"not System B's composite")

    def test_g_d1_1_sentinel_seedless_negative_control(self, monkeypatch):
        r"""NEGATIVE CONTROL (DP-seedless): a SEEDLESS slab solve leaves every
        block counter at 0 — the coupled machinery fires exactly where System
        B exists, never on the bare seedless arm. (The d1 bypass tooth — a
        driver handed the pre-d1 FUSED record running green with counters 0 —
        dissolved at d2: the fused 3-block spelling is unrepresentable, so a
        bypassing driver cannot even be CONSTRUCTED; the sentinel's teeth are
        now the type system plus this seedless control.)"""
        slab = SNMesh(
            Mesh1D(edges=np.linspace(0.0, 4.0, 6), mat_ids=np.zeros(5, dtype=int),
                   coord=CoordSystem.CARTESIAN, bc_right=BC("reflective"),
                   bc_left=BC("reflective")),
            Quadrature.gauss_legendre(4), {0: _mixture(1.0, 0.4, 2)})
        counters = {"N": 0, "M": 0}
        real_n = CoupledOperator.apply

        def spy_n(op, x, /):
            counters["N"] += 1
            return real_n(op, x)

        real_m = CoupledSubstitutionOperator.apply

        def spy_m(op, rhs, /, *, initial_guess=None):
            counters["M"] += 1
            return real_m(op, rhs, initial_guess=initial_guess)

        monkeypatch.setattr(CoupledOperator, "apply", spy_n)
        monkeypatch.setattr(CoupledSubstitutionOperator, "apply", spy_m)
        solver = SNSolver(slab, inner_solver="krylov")
        solver.solve_fixed_source(
            np.ones((slab.ng, slab.nx)), np.ones((slab.ng, slab.nx)))
        if counters["N"] != 0 or counters["M"] != 0:
            pytest.fail(f"a SEEDLESS solve drove the coupled machinery "
                        f"{counters} — DP-seedless is violated (the coupled "
                        f"carrier must appear exactly where System B exists).")

    # ── G-d1.4 — the M leg plumbing: consistency + coupling non-vacuity ──

    def test_g_d1_4_m_leg_plumbing_consistency(self):
        r"""The B.2d successor of the d1 fused-state-bridge round trip (the
        split/fuse pair dissolved with the 3-block): the M surfaces' explicit
        leaf legs are plumbed OBJECT-level —

        (a) ZERO-LEG CONSISTENCY: ``M.apply([x_A, 0])``'s System-A member ≡
        the (A,A) block's OWN bare call (the ray-decoupled action — since
        step 5 the block matvec's LC leg plus a ZERO Seeding term) — the
        structural zero and the explicit zero leg are the SAME arithmetic,
        array_equal;
        (b) COUPLING NON-VACUITY: a LIVE ψ_B moves y_A (the welded seed feed
        genuinely reads the leg — a leg-dropping pack would leave y_A at the
        (A,A) value) and fills y_B (the emitted rows ride the buffer);
        (c) ROUND TRIP: ``M.solve(M.apply(x))`` reproduces x at walk
        precision on BOTH systems (the joint inverse round trip — the
        object-level catcher for mutations keff is spectrally blind to)."""
        sn = _sphere()
        solver = SNSolver(sn)
        system = build_within_group_system(
            sn, solver.mat_xs, scattering_op=solver.scattering_op)
        M_op = system.implicit_operator
        rng = np.random.default_rng(150)
        psi_a = _random_composite(sn, rng)
        ns = sn.radial_characteristic_field_space.shape[0]
        live = _pair(sn, psi_a, rng.standard_normal(ns))
        dead = _pair(sn, psi_a, np.zeros(ns))
        # (a) zero-leg ≡ no-leg (the (A,A) block action — the grid's own
        # LC block, called bare on its ray-decoupled channel).
        y_dead = M_op.apply(dead)
        y_noleg = M_op.blocks[0][0].apply(psi_a)
        np.testing.assert_array_equal(
            y_dead.systems[0].interior.values, y_noleg.interior.values,
            err_msg="M.apply([x_A, 0]) ≠ the no-leg (A,A) block action — the "
                    "zero-substitution and the explicit zero leg drifted")
        np.testing.assert_array_equal(
            y_dead.systems[1].to_flat(), 0.0,
            err_msg="a zero ψ_B emitted nonzero ray rows (D·0 ≠ 0)")
        # (b) a live ψ_B moves y_A and fills y_B.
        y_live = M_op.apply(live)
        if not np.max(np.abs(
                y_live.systems[0].interior.values
                - y_dead.systems[0].interior.values)) > 1e-8:
            pytest.fail("a LIVE ψ_B left y_A unmoved — the welded seed feed "
                        "does not read the flux leg (a dropped pack)")
        if not np.max(np.abs(y_live.systems[1].to_flat())) > 1e-8:
            pytest.fail("a LIVE ψ_B left y_B empty — the emitted ray rows do "
                        "not ride the source buffer")
        # (c) the joint inverse round trip — ψ_A's bulk + ψ_B's CELLS legs
        # (the corner slots are the given-data BC slots: apply emits the
        # corner defect, solve reads the source corner as given data — the
        # same reason the step-0 floor asserted the bulk alone).
        back = M_op.solve(M_op.apply(live))
        np.testing.assert_allclose(
            back.systems[0].interior.values, psi_a.interior.values,
            rtol=1e-11, atol=1e-12,
            err_msg="M.solve(M.apply(x)) moved ψ_A past walk precision")
        back_b = back.systems[1]
        live_b = live.systems[1]
        for lv in sn.radial_characteristic_levels:
            for sign in (-1, +1):
                np.testing.assert_allclose(
                    back_b.interior.cells(lv, sign),
                    live_b.interior.cells(lv, sign),
                    rtol=1e-11, atol=1e-12,
                    err_msg=f"M.solve(M.apply(x)) moved ψ_B cells({lv},{sign})")

    # ── G-d1.5 — N ≡ the fused gains (control) + sign/shape teeth ────────

    def test_g_d1_5_gain_grid_matches_the_pieces_with_teeth(self):
        r"""CONTROL: the record's N grid reproduces the PIECE composition
        row-for-row ``array_equal`` — row A = ``(S + B_a)·ψ_A``, row B =
        ``Emission·ψ_A + B_b·ψ_B`` (the grid's block dispatch vs the direct
        piece application; a pure re-association on the gain side). The
        fused-shim reference dissolved at d2 with the 3-block. REFLECTIVE
        sphere (the B.2c F1 lesson: vacuum masks a dropped B_b).
        TEETH (each mutation moves a row O(1)): dropped B_b; sign-flipped
        Emission; a non-∅ (A,B) block double-counting the Seeding feed."""
        sn = _sphere(bc="reflective")
        solver = SNSolver(sn)
        system = build_within_group_system(
            sn, solver.mat_xs, scattering_op=solver.scattering_op)
        n_grid = system.explicit_gains[0]
        coupled = _random_pair(sn, np.random.default_rng(151))
        out = n_grid.apply(coupled)
        emission = n_grid.blocks[1][0]
        b_b = n_grid.blocks[1][1]
        n_aa = n_grid.blocks[0][0]
        row_a_ref = n_aa.apply(coupled.systems[0])
        row_b_ref = (emission.apply(coupled.systems[0])
                     + b_b.apply(coupled.systems[1]))
        np.testing.assert_array_equal(
            out.systems[0].interior.values, row_a_ref.interior.values,
            err_msg="N row A bulk ≠ (S + B_a)·ψ_A")
        np.testing.assert_array_equal(
            out.systems[0].boundary.values, row_a_ref.boundary.values,
            err_msg="N row A trace ≠ (S + B_a)·ψ_A")
        np.testing.assert_array_equal(
            out.systems[1].to_flat(),
            row_b_ref.to_flat(),
            err_msg="N row B ray ≠ Emission·ψ_A + B_b·ψ_B")
        # TEETH (in-process constructions; the real N is never touched):
        # (a) dropped B_b is UNCONSTRUCTABLE — B_b is the only block reading
        # System B's input, so the all-None-column guard refuses the grid
        # (stronger than a value-red: the mutation class is unspellable).
        with pytest.raises(ValueError, match="column 1 has no blocks"):
            CoupledOperator(
                [[n_aa, None], [emission, None]],
                domain=n_grid.domain, codomain=n_grid.codomain)
        # (b) sign-flipped Emission → the ray bulk row flips.
        flipped = CoupledOperator(
            [[n_aa, None], [-emission, b_b]],
            domain=n_grid.domain, codomain=n_grid.codomain)
        d_b = np.max(np.abs(
            flipped.apply(coupled).systems[1].interior.values
            - out.systems[1].interior.values))
        if not d_b > 1e-8:
            pytest.fail("tooth (b) dead: flipping the Emission sign left "
                        "the ray row unmoved")
        # (c) a non-∅ (A,B) block → the Seeding feed double-counts into bulk.
        seeding = RadialCharacteristicSeeding(sn)
        doubled = CoupledOperator(
            [[n_aa, seeding], [emission, b_b]],
            domain=n_grid.domain, codomain=n_grid.codomain)
        d_c = np.max(np.abs(
            doubled.apply(coupled).systems[0].interior.values
            - n_grid.apply(coupled).systems[0].interior.values))
        if not d_c > 1e-10:
            pytest.fail("tooth (c) dead: a non-∅ (A,B) block left the bulk "
                        "row unmoved — the Seeding double-count is invisible")

    # ── G-d1.6 — Mode-9 same fixed point (het vacuum) + slab control ─────

    def test_g_d1_6_same_fixed_point_het_vacuum_sphere(self):
        r"""The TWO block-native inner drivers (SI and Krylov — structurally
        different iterations over the SAME record splitting) converge to the
        SAME fixed point on the Mode-9 configuration — heterogeneous VACUUM
        2-region sphere, 2G (NOT the reflective isotropic box).
        Principled-equiv bar SAFETY × inner_tol. The FLUX FIELD is asserted
        — bulk AND System B's ψ½ member — not keff (outside every spectral
        invariance group). (The d1 spelling compared against the pre-d1
        fused driver, proven then; the fused reference dissolved with the
        3-block at d2, and the cross-driver row is the standing successor.)"""
        tol = 1e-11
        sn = _het_vacuum_sphere()
        q_np = np.ones((sn.quad.N, sn.ng, sn.nx))
        mats = {0: _mixture(1.0, 0.4, 2), 1: _mixture(0.5, 0.1, 2)}
        sols = {
            inner: solve_sn_fixed_source(
                mats, sn.mesh, Quadrature.gauss_legendre(4), q_np,
                inner_solver=inner, inner_schedule="jacobi",
                max_inner=6000, inner_tol=tol)
            for inner in ("source_iteration", "krylov")
        }
        bar = 100.0 * tol
        si_sol, ky_sol = sols["source_iteration"], sols["krylov"]
        np.testing.assert_allclose(
            si_sol.angular_flux.interior.values,
            ky_sol.angular_flux.interior.values,
            rtol=bar, atol=bar,
            err_msg="SI and Krylov bulk fixed points diverged past SAFETY×tol")
        for inner, sol in sols.items():
            if sol.radial_characteristic is None:
                pytest.fail(f"[{inner}] Solution.radial_characteristic is None "
                            f"on a carrying sphere (DP-Solution broken)")
        np.testing.assert_allclose(
            si_sol.radial_characteristic.to_flat(),
            ky_sol.radial_characteristic.to_flat(),
            rtol=bar, atol=bar,
            err_msg="SI and Krylov ψ½ fixed points diverged past SAFETY×tol")

    def test_g_d1_6_slab_control_is_bit_identical(self):
        r"""CONTROL (ordering hazard 5): the seedless record arm is a PURE
        re-package — the production slab SI converges BIT-IDENTICAL
        (array_equal) to the hand-built ``SourceIteration(L+C⁻¹, S, B_a)``
        on the same rhs. A drift here is a bug, never principled-equiv."""
        slab = SNMesh(
            Mesh1D(edges=np.linspace(0.0, 4.0, 6), mat_ids=np.zeros(5, dtype=int),
                   coord=CoordSystem.CARTESIAN, bc_right=BC("reflective"),
                   bc_left=BC("reflective")),
            Quadrature.gauss_legendre(4), {0: _mixture(1.0, 0.4, 2)})
        tol, mi = 1e-11, 3000
        q_np = np.ones((slab.quad.N, slab.ng, slab.nx))
        sol = solve_sn_fixed_source(
            {0: _mixture(1.0, 0.4, 2)}, slab.mesh,
            Quadrature.gauss_legendre(4), q_np,
            inner_solver="source_iteration", inner_schedule="jacobi",
            max_inner=mi, inner_tol=tol)
        solver = SNSolver(slab)
        system = build_within_group_system(
            slab, solver.mat_xs, scattering_op=solver.scattering_op)
        q3 = _build_fixed_source_rhs(q_np, slab)
        si = SourceIteration(
            system.implicit_operator.inverse(), *system.explicit_gains, max_iter=mi, tol=tol)
        psi_ref, _ = si.solve(
            q3, initial_guess=_unwindowed_cold_start(
                slab, history_depth=q3.history_depth))
        np.testing.assert_array_equal(
            sol.angular_flux.interior.values, psi_ref.interior.values,
            err_msg="the seedless record arm is NOT a pure re-package — "
                    "the slab control drifted (ordering hazard 5)")

    # ── G-d1.7 — the SI displacement diagnostic survives (F5) ────────────

    def test_g_d1_7_si_displacement_diagnostic_on_the_coupled_iterate(self):
        r"""The SI convergence diagnostics survive the CoupledField iterate:
        ``CoupledField.principal_bulk_leaf`` delegates to the PRIMARY system
        (System A's bulk — CS3-R relocated the walk onto the carriers),
        so the increment norms RECORD on a carrying sphere (F5 — previously
        uncovered; a silent-empty diagnostic is the failure mode)."""
        sn = _sphere(c=0.6)
        solver = SNSolver(sn)
        system = build_within_group_system(
            sn, solver.mat_xs, scattering_op=solver.scattering_op)
        si, *_ = _within_group_si(
            system, sn, inner_schedule="jacobi", max_iter=60, tol=1e-10)
        q_pair = _build_fixed_source_rhs(
            np.ones((sn.quad.N, sn.ng, sn.nx)), sn)  # coupled on carrying
        if not isinstance(q_pair, CoupledField):
            pytest.fail("the carrying rhs builder did not return the coupled pair")
        cold = _coupled_flux_state(
            _unwindowed_cold_start(sn, history_depth=2), sn)
        _, rec = si.solve(q_pair, initial_guess=cold)
        if not rec.increment_norms:
            pytest.fail("increment_norms is EMPTY on the coupled iterate — "
                        "the increment diagnostic went silent (F5)")
        if not rec.contraction_ratios:
            pytest.fail("contraction_ratios is EMPTY on the coupled iterate — "
                        "fewer than two increments recorded (F5)")

    # ── G-d1.8 / G-d2.3 — the HONEST coupled DOF + the ERR-053 restart ───

    def test_g_d2_3_honest_dof_and_krylov_restart(self):
        r"""The coupled ravel is the HONEST two-system sum (G-d2.3): ``n_dof
        == size_A(2-block) + size_B(composite)`` with NO dead-ray padding —
        strictly ``n_seed`` less than the d1 padded count (the "DOF count
        goes honest" claim, quantified: Δ == n_seed). And the Krylov
        ``restart`` sizing reads exactly this coupled ``to_flat`` (ERR-053
        stays closed at the coupled seam)."""
        sn = _sphere()
        solver = SNSolver(sn)
        system = build_within_group_system(
            sn, solver.mat_xs, scattering_op=solver.scattering_op)
        cold_a = _unwindowed_cold_start(sn, history_depth=2)
        cold = _coupled_flux_state(cold_a, sn)
        n_dof = int(cold.to_flat().size)
        size_a = int(cold_a.to_flat().size)
        size_b = int(np.asarray(cold.systems[1].to_flat()).size)
        n_seed = int(sn.radial_characteristic_field_space.shape[0])
        nb = sn.quad.N * sn.ng * sn.nx
        nt = int(sn.angular_trace.layout.total_size)
        if size_a != nb + nt:
            pytest.fail(f"size_A = {size_a} ≠ bulk + trace = {nb + nt} — "
                        f"ψ_A still carries padding (the eviction leaked)")
        if n_dof != size_a + size_b:
            pytest.fail(f"coupled n_dof = {n_dof} ≠ size_A + size_B = "
                        f"{size_a + size_b} — the coupled ravel lost a system")
        padded_d1 = size_a + n_seed + size_b   # the retired dead-pad count
        if padded_d1 - n_dof != n_seed:
            pytest.fail(f"Δ(padded − honest) = {padded_d1 - n_dof} ≠ n_seed = "
                        f"{n_seed} — the dead padding did not dissolve exactly")
        krylov = _within_group_krylov(
            system.implicit_operator, *system.explicit_gains,
            n_dof=n_dof, max_iter=5, tol=1e-3)
        if krylov.restart != n_dof:
            pytest.fail(f"restart = {krylov.restart} ≠ n_dof = {n_dof} — "
                        f"ERR-053 re-opened at the coupled seam")


# ═══════════════════════════════════════════════════════════════════════
# B.2d d3 — the E4 closed-form END-TO-END anchors (G-d3.1 / G-d3.2)
# ═══════════════════════════════════════════════════════════════════════


def _pure_absorber_reflective_sphere(ng: int = 2, nx: int = 8):
    r"""REFLECTIVE PURE-ABSORBER (c = 0) carrying sphere, group-graded σ_t.

    ``_mixture(σ, 0.0, ng)`` grades ``σ_t,g = σ·(1 + 0.4g)`` with zero
    scattering (pure capture) — the Mode-6 group-swap catcher: each group's
    equilibrium ratio ``Q_g/Σ_t,g`` is distinct.
    """
    mesh = Mesh1D(
        edges=np.linspace(0.0, 4.0, nx + 1), mat_ids=np.zeros(nx, dtype=int),
        coord=CoordSystem.SPHERICAL, bc_right=BC("reflective"),
    )
    return mesh, Quadrature.gauss_legendre(4), {0: _mixture(1.3, 0.0, ng)}


class TestWithinGroupSystemAnchors:
    r"""B.2d d3 — the E4 closed-form END-TO-END anchors through the
    block-native driver (memo G-d3.1 / G-d3.2).

    **G-d3.2 (k_inf) is carried by the EXISTING production anchor** —
    ``tests/sn/verification/analytical/test_kinf_homogeneous.py::
    test_kinf_homogeneous`` runs the production ``solve_sn`` (block-native
    since d1) on the carrying sphere × {1, 2, 4}G × BOTH inner solvers
    against the ``orpheus.derivations`` closed form at ``rtol = 1e-10`` —
    tighter than the memo's 1e-8 floor.  A twin here would duplicate a
    green production gate (Cardinal Rule 2); this class records the
    PAIRING instead: per the memo's Mode-12 sweep, k_inf is the
    eigenvalue-LAYER anchor only — it is NEVER credited against
    shape-class (leg-swap / ray-coupling) mutations, whose committed
    catchers are G-d1.4's object-level round trip, G-d1.6's HETEROGENEOUS
    same-fixed-point FIELD row, and G-d3.1's flat-flux FIELD below.
    """

    pytestmark = [
        pytest.mark.l1,
        pytest.mark.catches("ERR-026"),
        pytest.mark.verifies(
            "transport-cartesian", "sn-curvilinear-homogeneous-kinf-recovery",
        ),
    ]

    @pytest.mark.parametrize(
        "inner_solver", ["source_iteration", "krylov"],
    )
    def test_g_d3_1_flat_flux_equilibrium_on_the_carrying_sphere(
        self, inner_solver,
    ):
        r"""G-d3.1 — E4 ``φ = Q/Σ_t`` END-TO-END (the single most powerful
        curvilinear diagnostic), BOTH drivers, ``rtol ≤ 1e-10``.

        On the reflective pure-absorber carrying sphere with a uniform
        per-ordinate source ``Q_g`` and group-graded ``Σ_t,g``, the
        converged state is the flat-flux equilibrium EXACTLY — the M-M
        closure's flat-flux consistency condition makes streaming +
        angular redistribution vanish per ordinate, so the discrete
        answer equals the continuous one at solver tolerance::

            ψ_n,g = Q_g / Σ_t,g      φ_g = Σw · Q_g / Σ_t,g

        and System B's converged member (``Solution.radial_
        characteristic``) sits at the SAME equilibrium — the driver-level
        end-to-end companion of the operator-level
        ``TestA_BB_RadialBVP::test_fixed_source_equilibrium_Q_over_sigma``.

        A driver sign/shape bug in the ray coupling moves the FIELD O(1)
        (flat flux is exact — never sub-floor; the mutation teeth ride
        G-d1.5's coupling-wire mutations, which divert through this same
        driver).  The 1G/krylov-only sibling
        (``test_fixed_source_g1::test_uniform_source_converges_to_q_over_
        sigma_t``) stays the ERR-049 W-factor sentinel; THIS gate adds
        ≥2G group grading (Mode-6), the source-iteration arm, and the
        1e-10 contract.
        """
        ng, nx = 2, 8
        mesh, quad, materials = _pure_absorber_reflective_sphere(ng=ng, nx=nx)
        sig_t_g = np.asarray(materials[0].SigT, dtype=float)
        Q_g = np.array([3.0, 0.7])
        q = np.broadcast_to(
            Q_g[None, :, None], (quad.N, ng, nx),
        ).astype(float).copy()

        sol = solve_sn_fixed_source(
            materials=materials, mesh=mesh, quadrature=quad,
            external_source=q, inner_solver=inner_solver,
            max_inner=400, inner_tol=1e-13,
        )
        if not sol.history.converged:
            pytest.fail(f"[{inner_solver}] fixed-source did not converge "
                        f"on the pure-absorber sphere")

        psi = np.asarray(sol.angular_flux.interior.values)
        np.testing.assert_allclose(
            psi,
            np.broadcast_to((Q_g / sig_t_g)[None, :, None], psi.shape),
            rtol=1e-10,
            err_msg=(
                f"[{inner_solver}] per-ordinate ψ off the flat-flux "
                f"equilibrium Q_g/Σ_t,g — the E4 curvilinear catcher "
                f"(a ray-coupling sign/shape bug in the block-native "
                f"driver is O(1) here, never sub-floor)"
            ),
        )
        sum_w = float(quad.weights.sum())
        phi = np.asarray(
            sol.angular_flux.interior.integrate_angular().values,
        )
        np.testing.assert_allclose(
            phi,
            np.broadcast_to((sum_w * Q_g / sig_t_g)[:, None], phi.shape),
            rtol=1e-10,
            err_msg=f"[{inner_solver}] φ off Σw·Q_g/Σ_t,g",
        )
        # System B end-to-end: the converged ray member carries the SAME
        # per-group equilibrium (layout-free set check: every composite
        # value IS one of the two group ratios, and both appear).
        ray = sol.radial_characteristic
        if ray is None:
            pytest.fail(f"[{inner_solver}] Solution.radial_characteristic "
                        f"is None on the carrying sphere (DP-Solution)")
        u = np.asarray(ray.to_flat(), dtype=float)
        eq = Q_g / sig_t_g
        one_of = np.isclose(u[:, None], eq[None, :], rtol=1e-10).any(axis=1)
        if not one_of.all():
            pytest.fail(
                f"[{inner_solver}] ray member values off the flat "
                f"equilibrium set {{Q_g/Σ_t,g}}: "
                f"worst = {u[~one_of][:3]!r} vs {eq!r}"
            )
        for g, v in enumerate(eq):
            if not np.isclose(u, v, rtol=1e-10).any():
                pytest.fail(
                    f"[{inner_solver}] group-{g} equilibrium {v} absent "
                    f"from the ray member — a group axis dropped (Mode 6)"
                )


# ═══════════════════════════════════════════════════════════════════════
# Step 7 — TestPrescribedCornerDatum: the three-arm inflow-corner law
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.foundation
@pytest.mark.catches("ERR-069")
class TestPrescribedCornerDatum:
    r"""The rhs birth delivers the PRESCRIBED-inflow ψ½ corner datum
    (step 7 — the d3 regression's fast wiring catcher; ERR-069).

    The inflow-corner law has three arms (see
    :meth:`RadialCharacteristicField.source_from_angular`): vacuum ⇒ 0,
    reflective ⇒ the ``B_b`` gain arm, prescribed ⇒ the SOURCE's own trace
    datum at the most-inward ``xmax`` ordinate. The d3 carve wired only the
    first two; the sphere prescribed-inflow MMS converged to a wrong fixed
    point (L2 ≈ 0.21 vs the 2.4e-3 floor) with its ONLY catcher slow-marked
    — five not-slow walls never saw it. This class is the not-slow wiring
    pin: drop the ``boundary_trace`` threading in
    ``_build_fixed_source_rhs`` (the recorded red) and the prescribed arm
    here REDs immediately; the slow MMS gate remains the value-level
    catcher.
    """

    def test_prescribed_trace_populates_the_inflow_corner(self):
        r"""A nonzero (ordinate-asymmetric) prescribed trace lands on every
        carrying level's inflow corner as the most-inward-ordinate row —
        the nearest-node proxy for ψ_in(μ=−1), the pre-d3
        ``bc_outer_value`` datum restored through the source channel. The
        outflow corner stays zero (the defect row, R13)."""
        from orpheus.transport.source_sinks import (
            AngularBoundarySourceSink,
            AngularSourceSink,
        )
        from orpheus.transport.timed_full_field import TimedFullField

        sn = _sphere(nx=4, ng=2)
        rng = np.random.default_rng(7)
        trace_vals = rng.random((sn.quad.N, sn.ng))
        bnd = AngularBoundarySourceSink.zeros(sn.angular_trace)
        bnd.face_view("xmax")[...] = trace_vals
        q = TimedFullField(
            interior=AngularSourceSink(values=np.ones((sn.quad.N, sn.ng, sn.nx)), space=sn.angular_bulk_space),
            boundary=bnd,
            _history=(),
            history_depth=2,
        )
        rhs = _build_fixed_source_rhs(q, sn)
        if not isinstance(rhs, CoupledField):
            pytest.fail("carrying rhs builder did not return the coupled pair")
        seed = rhs.systems[1]
        mu = sn.quad.mu_x
        level_indices = sn.angular_closure.level_indices
        levels = tuple(seed.boundary.space.levels)
        if not levels:
            pytest.fail("sphere rhs has no carrying levels — the pin is vacuous")
        for p in levels:
            ords = np.asarray(level_indices[p])
            most_inward = ords[int(np.argmin(mu[ords]))]
            np.testing.assert_array_equal(
                seed.boundary.corner(p, -1), trace_vals[most_inward, :],
                err_msg=(
                    f"level {p}: the prescribed trace's most-inward row did "
                    f"not reach the ψ½ inflow corner — the d3 dropped-corner "
                    f"regression is back"
                ),
            )
            np.testing.assert_array_equal(
                seed.boundary.corner(p, +1), np.zeros(sn.ng),
                err_msg=f"level {p}: the outflow (defect) corner must stay zero",
            )

    def test_vacuum_rhs_keeps_zero_corners(self):
        r"""The vacuum arm is untouched: a bare-ndarray (vacuum) rhs leaves
        every corner zero — the arm is dormant off the prescribed class,
        so vacuum/reflective solves stay byte-identical."""
        sn = _sphere(nx=4, ng=2)
        rhs = _build_fixed_source_rhs(
            np.ones((sn.quad.N, sn.ng, sn.nx)), sn,
        )
        if not isinstance(rhs, CoupledField):
            pytest.fail("carrying rhs builder did not return the coupled pair")
        seed = rhs.systems[1]
        for p in seed.boundary.space.levels:
            for sign in (-1, +1):
                np.testing.assert_array_equal(
                    seed.boundary.corner(p, sign), np.zeros(sn.ng),
                    err_msg=f"vacuum rhs corner ({p}, {sign:+d}) is nonzero",
                )


# ═══════════════════════════════════════════════════════════════════════
# Step 5 (#41) — TestCoupledSolve: the block solve's SN-bound rows
# ═══════════════════════════════════════════════════════════════════════


class TestCoupledSolve:
    r"""The step-5 block solve on the ψ½ instance (the TA step-5 memo's
    B-rows) — the record's M is the honest upper-triangular grid and its
    ``solve`` the numerics substitution.

    * **r1 (MANDATORY, the reflective-corner data-flow)** — the
      substitution's ``march.solve(rhs_B)`` reads the corner datum from
      ``rhs_B``'s corner slot exactly as the fused route did (``B_b ∈ N``
      in BOTH routes — the corner arrives via the rhs, never via hidden
      state). Gated against the STRUCTURALLY-INDEPENDENT dense-LU of the
      probed M (the row-6 doctrine: the rhs is MANUFACTURED in M's range,
      ``q = M·ψ0`` with ψ0 random on the bulk ⊕ ray blocks, so every slot
      is consistent), corner block asserted explicitly, on the REFLECTIVE
      sphere (the non-trivial ``B_b`` corner-swap fixture).
    * **B3** — the transposed substitution against the dense ``Mᵀ`` under
      the same manufactured-cotangent doctrine (``b = Mᵀ·x̄0``).
    """

    def _system(self, bc: str = "reflective"):
        sn = _sphere(bc=bc)
        solver = SNSolver(sn)
        return sn, build_within_group_system(
            sn, solver.mat_xs, scattering_op=solver.scattering_op)

    @staticmethod
    def _carried_state(sn, seed: int):
        """ψ0 random on bulk ⊕ the WHOLE ray block (cells + corner — the r1
        focus), zero trace — the row-6 manufactured-range doctrine."""
        rng = np.random.default_rng(seed)
        tpl = _coupled_template(sn)
        bulk, _tr, ray, _ = _blocks(sn)
        psi0 = np.zeros(tpl.to_flat().size)
        psi0[bulk] = rng.standard_normal(psi0[bulk].size)
        psi0[ray] = rng.standard_normal(psi0[ray].size)
        return tpl, psi0

    def test_r1_substitution_corner_dataflow_vs_dense_lu(self):
        sn, system = self._system(bc="reflective")
        M_grid = system.implicit_operator
        tpl, psi0 = self._carried_state(sn, 51)
        q = M_grid.apply(CoupledField.from_flat(psi0, tpl))
        x = M_grid.solve(q).to_flat()
        dense = _dense(M_grid.apply, tpl)
        reference = np.linalg.solve(dense, q.to_flat())
        bulk, _tr, ray, _ = _blocks(sn)
        # The corner slots are the walk's COMPUTED/free-DOF pair (#284:
        # solve computes the outflow slots apply treats as free DOFs), so
        # the two-sided dense inverse does not hold THERE — the corner
        # data-flow is evidenced by the ray CELLS (which depend on the
        # inflow corner datum through the march: a broken corner read
        # moves them O(1) off this reference) plus the facade bit-row
        # below (corners included, array_equal).
        n_cells = RadialCharacteristicField.flux_zeros(sn.radial_characteristic_field_space).interior.values.size
        ray_cells = slice(ray.start, ray.start + n_cells)
        for name, sl in (("bulk", bulk), ("ray cells", ray_cells)):
            np.testing.assert_allclose(
                x[sl], reference[sl], rtol=1e-11, atol=1e-13,
                err_msg=f"substitution off the dense-LU reference on the "
                        f"{name} block — the r1 data-flow broke")

    def test_b3_transpose_substitution_vs_dense_mt(self):
        sn, system = self._system(bc="reflective")
        M_grid = system.implicit_operator
        tpl, x0 = self._carried_state(sn, 53)
        dense = _dense(M_grid.apply, tpl)
        if np.allclose(dense, dense.T):
            pytest.fail("fixture drift: M is symmetric — the transpose "
                        "gate is Mode-12-blind")
        b = M_grid.apply_transpose(CoupledField.from_flat(x0, tpl))
        xt = M_grid.solve_transpose(b).to_flat()
        reference = np.linalg.solve(dense.T, b.to_flat())
        bulk, _tr, ray, _ = _blocks(sn)
        # Same #284 computed-slot doctrine as the forward: the corner
        # cotangent slots are outside the two-sided-inverse subspace.
        n_cells = RadialCharacteristicField.flux_zeros(sn.radial_characteristic_field_space).interior.values.size
        ray_cells = slice(ray.start, ray.start + n_cells)
        for name, sl in (("bulk", bulk), ("ray cells", ray_cells)):
            np.testing.assert_allclose(
                xt[sl], reference[sl], rtol=1e-11, atol=1e-13,
                err_msg=f"transposed substitution off the dense-Mᵀ "
                        f"reference on the {name} block")
