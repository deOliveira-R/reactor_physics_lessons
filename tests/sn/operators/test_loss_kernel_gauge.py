r"""``LossKernelGauge`` — the closed-form projector onto :math:`\ker(L+C-S-B)`.

**What has to be true, and how each is checked without using the gauge as its
own oracle.** The construction is a closed form, so a gate that only asked
"does it project onto what it built?" would be vacuous. Every substantive claim
here is pinned against something the construction does not touch:

============================================  ==========================================
claim                                          independent reference
============================================  ==========================================
the dimension is right                         the combinatorial counting law
                                               (:func:`predicted_kernel_dimension`,
                                               which builds no vector), AND a dense
                                               SVD of the assembled :math:`A`
the vectors really are in :math:`\ker A`       the production matvec, per mode, ALL modes
the deviation it removes is the REAL one       a production Gauss-Seidel solve, against
                                               the analytic uniform trace
it is not a universal absorber                 the Jacobi control, which already lands on
                                               the correct member and must be left alone
============================================  ==========================================

**The dense-SVD ground truth, in full.** Four configurations re-run the SVD
live (:func:`test_the_dimension_matches_a_DENSE_SVD_of_the_assembled_operator`);
the construction was originally validated against **13**, and the rest are
recorded here because their fixtures are too large to re-solve every run and
the table is otherwise only in an untracked scratch memo. ``mine`` = the
closed form's column count, ``law`` = the combinatorial count computed by a
separate function that builds no vector — **both of these are** :math:`R`
**alone**; ``T`` = the tangential slots; ``svd`` = dense SVD of the assembled
:math:`A`, which sees the WHOLE kernel, so the table's own identity is
:math:`{\rm svd} = T + R`. ``gap`` =
:math:`\lVert(I - P_{\rm mine})P_{\rm svd}\rVert_2`, meaningful only where
:math:`T = 0` (see below).

====================================  ====================  ====  ====  ====  ====  ===========  =========
case                                  quadrature            mine   law     T   svd      max res        gap
====================================  ====================  ====  ====  ====  ====  ===========  =========
d2 (3,4) refl abs ng=2                level_symmetric(S4)     12    12     0    12     2.80e-16  1.288e-14
d2 (3,4) refl abs ng=2                level_symmetric(S2)      4     4     0     4     2.21e-16  2.220e-14
d2 (5,6) refl c=0.9 ng=3              level_symmetric(S4)     18    18     0    18     4.16e-16  2.049e-14
d2 (3,4) refl abs ng=2                product(4,4)             0     0   224   224  n/a (empty)        n/a
d2 (3,4) refl abs ng=2                product(8,8)            16    16   448   464     9.90e-16        n/a
d2 (3,4) refl abs ng=2                lebedev(11)             18    18   224   242     8.08e-16        n/a
d2 (3,4) GRADED refl abs ng=2         level_symmetric(S4)     12    12     0    12     5.37e-16  1.416e-14
d2 (3,4) x-VAC y-refl ng=2            level_symmetric(S4)      0     0     0     0     0.00e+00        n/a
d3 (2,2,2) refl abs ng=1              level_symmetric(S4)     33    33     0    33     1.50e-16  9.801e-15
d3 (2,2,3) refl abs ng=1              level_symmetric(S4)     39    39     0    39     1.39e-16  1.018e-14
d3 (2,2,2) refl abs ng=2              level_symmetric(S2)     22    22     0    22     1.49e-16  5.790e-15
d3 (2,2,2) refl abs ng=1              product(4,4)             8     8   128   136     4.00e-16        n/a
d3 (2,2,2) xy-refl z-VAC ng=1         level_symmetric(S4)     12    12     0    12     0.00e+00  1.526e-14
====================================  ====================  ====  ====  ====  ====  ===========  =========

Note the structural rows: ``product(4,4)`` at d=2 is **pure T** (224
tangential, ``R = 0``, so the construction correctly returns an EMPTY basis and
``max res`` has no modes to range over), and the mixed-BC row is a negative
control where the correct answer is 0 — an empty basis rather than manufactured
spurious modes.

⛔ **This table was WRONG when it landed at** ``f934ff57``, **and the correction
is instructive.** Every T-bearing row recorded :math:`T + R` under the ``mine``
and ``law`` columns — ``224 / 464 / 242 / 136`` where the true :math:`R` is
``0 / 16 / 18 / 8``. It contradicted its own prose three lines below (which said
``R = 0``) and ``docs/theory/methods/sn/cartesian_multid.rst`` (which was
correct). ``[M]`` re-measured 2026-08-15: **4 of 4** T-bearing rows, not the
3 first reported — the ``d3 (2,2,2) product(4,4)`` row is the one an audit
scanning only the d=2 block misses. The ``gap`` column was wrong for the same
reason and is now ``n/a`` there *by construction*: :math:`P_{\rm mine}` spans
:math:`R` and :math:`P_{\rm svd}` spans :math:`T \oplus R`, so a whole-space
projector distance between them is :math:`\approx 1` and says nothing about the
closed form. Containment — the claim that actually matters — is carried by
``max res`` on every row and by
:func:`test_EVERY_basis_vector_is_annihilated_by_the_production_matvec`.
The gate that would have caught this is
``tests/sn/operators/test_loss_nullspace_reflective_box.py``'s T+R row, which
did not exist until Step 7; every ``_SINGULAR`` fixture here is
``level_symmetric``, where :math:`T = 0` and the error is unspellable.

⚠ **The load-bearing gate is**
:func:`test_each_block_gram_is_the_identity_which_is_what_earns_DIAGONAL`.
``[M]`` the *raw pair generators* — the natural thing to ship — have a Gram that
is ``0.000e+00`` off-diagonal at :math:`d = 2` and **``4.05e-01``** at
``(3,4,5)``. The d=2 reading is vacuous (:math:`\kappa(\{x,y\}) = 1`, so a
:math:`1\times1` Gram is diagonal for free), and a suite that only exercised
:math:`d = 2` would certify ``DIAGONAL`` on a basis that is 43 % off-diagonal
where it matters — producing a silently mis-normalised projector, because
:attr:`~orpheus.numerics.frame.FrameBase.gram_inverse`'s row-sum probe equals the true
diagonal only when :math:`MR` is diagonal. **Every d=3 row in this file exists
for that reason; do not trim them to speed the suite up.**
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.numerics.manifold import IndexSet
from orpheus.derivations.common.xs_library import make_mixture
from orpheus.geometry import BC
from orpheus.numerics.basis import GramStructure
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.coupled_system import build_within_group_system
from orpheus.sn.operators.loss_kernel_gauge import (
    GaugeFreedom,
    LossKernelGauge,
    _anova_dimension,
    _reflection_orbits,
    gauge_freedom,
    predicted_kernel_dimension,
)
from orpheus.sn.solver import SNSolver, _as_sn_mesh, _unwindowed_cold_start
from orpheus.transport.mesh.axis import AxisMesh
from orpheus.transport.spatial.linear_discontinuous import LinearDiscontinuous

_R, _V = BC("reflective"), BC("vacuum")


# ─────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────
def _mixture(ng: int = 2, c: float = 0.0, fissile: bool = False):
    sig_t = np.linspace(0.8, 1.6, ng)
    sig_s = np.zeros((ng, ng))
    np.fill_diagonal(sig_s, c * sig_t)
    sig_f = 0.3 * sig_t if fissile else np.zeros(ng)
    chi = np.zeros(ng)
    if fissile:
        chi[0] = 1.0
    return make_mixture(
        sig_t=sig_t, sig_c=sig_t - sig_s.sum(axis=0) - sig_f,
        sig_f=sig_f, nu=np.full(ng, 2.4) if fissile else np.zeros(ng),
        chi=chi, sig_s=sig_s,
    )


def _graded(extent: float, n: int, stretch: float = 1.0) -> np.ndarray:
    """Geometric cell widths — ``stretch == 1.0`` is uniform."""
    widths = stretch ** np.arange(n)
    return np.concatenate([[0.0], np.cumsum(widths / widths.sum() * extent)])


def _mesh(cells, bcs, *, ng=2, quad=None, c=0.0, stretch=1.0,
          fissile=False, scheme=None):
    extents = (1.0, 2.0, 3.0)[: len(cells)]
    axes = tuple(
        AxisMesh(edges=_graded(e, n, stretch), bc_low=lo, bc_high=hi)
        for e, n, (lo, hi) in zip(extents, cells, bcs)
    )
    return _as_sn_mesh(
        axes, quad or Quadrature.level_symmetric(sn_order=4),
        {0: _mixture(ng, c, fissile)}, scheme=scheme,
    )


def _loss_system(sn_mesh):
    """``(system, template)`` — the PRODUCTION splitting the SI driver iterates."""
    solver = SNSolver(sn_mesh, inner_solver="source_iteration")
    system = build_within_group_system(
        sn_mesh, solver.mat_xs, scattering_op=solver.scattering_op,
    )
    return system, _unwindowed_cold_start(sn_mesh, history_depth=0)


def _apply_loss(system, template, flat: np.ndarray) -> np.ndarray:
    r""":math:`A x = (\text{implicit} - \sum \text{gains})\,x`."""
    x = type(template).from_flat(flat, template)
    out = system.implicit_operator.apply(x)
    for gain in system.explicit_gains:
        out = out - gain.apply(x)
    return out.to_flat()


def _embed_trace(template, trace: np.ndarray) -> np.ndarray:
    """A trace vector as a full-field flat vector with zero bulk."""
    return np.concatenate(
        [np.zeros(template.interior.values.size), trace])


#: ``(label, cells, bcs, kwargs)``. The d=3 rows are NOT optional — see the
#: module docstring.
_SINGULAR = [
    ("d2 (3,4) LS4 ng=2", (3, 4), [(_R, _R)] * 2, {}),
    ("d2 (5,6) LS4 ng=3 c=0.9", (5, 6), [(_R, _R)] * 2, {"ng": 3, "c": 0.9}),
    ("d2 (3,4) GRADED 1.7x", (3, 4), [(_R, _R)] * 2, {"stretch": 1.7}),
    ("d2 (3,4) LS2 ng=1", (3, 4), [(_R, _R)] * 2,
     {"ng": 1, "quad": Quadrature.level_symmetric(sn_order=2)}),
    ("d3 (2,2,2) LS4 ng=1", (2, 2, 2), [(_R, _R)] * 3, {"ng": 1}),
    ("d3 (2,2,3) LS4 ng=1", (2, 2, 3), [(_R, _R)] * 3, {"ng": 1}),
]

#: Small enough that a dense SVD of ``A`` is viable as a second oracle.
_DENSE_SVD_VIABLE = {
    "d2 (3,4) LS4 ng=2", "d2 (3,4) GRADED 1.7x", "d2 (3,4) LS2 ng=1",
    "d3 (2,2,2) LS4 ng=1",
}


# ─────────────────────────────────────────────────────────────────────
# 1. the dimension, against two independent oracles
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.foundation
@pytest.mark.verifies("dd-null-counting-law")
@pytest.mark.parametrize("label,cells,bcs,kwargs", _SINGULAR,
                         ids=[row[0] for row in _SINGULAR])
def test_the_dimension_matches_the_combinatorial_counting_law(
        label, cells, bcs, kwargs):
    r"""``gauge.dimension`` == the law, computed by a route that builds nothing.

    :func:`predicted_kernel_dimension` walks the ANOVA combinatorics of the
    theory and returns an integer; the gauge walks the pair generators and takes
    an SVD. They share the orbit decomposition and nothing else, so agreement is
    real evidence — and it is the ONLY available oracle at production sizes,
    where the dense SVD is not viable.
    """
    mesh = _mesh(cells, bcs, **kwargs)
    gauge = LossKernelGauge.for_mesh(mesh)
    assert gauge.dimension == predicted_kernel_dimension(mesh) > 0


@pytest.mark.foundation
@pytest.mark.verifies("dd-null-counting-law")
@pytest.mark.parametrize(
    "label,cells,bcs,kwargs",
    [row for row in _SINGULAR if row[0] in _DENSE_SVD_VIABLE],
    ids=[row[0] for row in _SINGULAR if row[0] in _DENSE_SVD_VIABLE],
)
def test_the_dimension_matches_a_DENSE_SVD_of_the_assembled_operator(
        label, cells, bcs, kwargs):
    """The numerical oracle: rank-deficiency of ``A`` itself.

    ``level_symmetric`` places no tangential ordinate, so ``T`` is empty and the
    whole kernel is the component ``R`` the gauge builds — which is what makes
    this an equality rather than an inequality. The ``product`` case, where
    ``T`` is large, is covered separately.
    """
    mesh = _mesh(cells, bcs, **kwargs)
    system, template = _loss_system(mesh)
    n_dof = template.to_flat().size

    dense = np.empty((n_dof, n_dof))
    unit = np.zeros(n_dof)
    for column in range(n_dof):
        unit[column] = 1.0
        dense[:, column] = _apply_loss(system, template, unit)
        unit[column] = 0.0
    singular = np.linalg.svd(dense, compute_uv=False)
    rank_deficiency = int(np.sum(singular < 1e-10 * singular[0]))

    assert LossKernelGauge.for_mesh(mesh).dimension == rank_deficiency


@pytest.mark.foundation
def test_the_counting_law_reproduces_the_two_closed_form_specialisations():
    r"""``ng * N/4`` at d=2 (mesh-INDEPENDENT) and ``ng*(N/8)*(2*sum(n)-1)`` at d=3.

    Pinned as literals derived from the theory, so a change to the orbit
    decomposition cannot quietly move both the law and the construction
    together.
    """
    n_ordinates = Quadrature.level_symmetric(sn_order=4).weights.size
    for cells in [(3, 4), (5, 6), (2, 2)]:          # d=2 is mesh-independent
        mesh = _mesh(cells, [(_R, _R)] * 2, ng=2)
        assert predicted_kernel_dimension(mesh) == 2 * (n_ordinates // 4)

    for cells in [(2, 2, 2), (2, 2, 3), (3, 4, 5)]:
        mesh = _mesh(cells, [(_R, _R)] * 3, ng=1)
        assert predicted_kernel_dimension(mesh) == (
            (n_ordinates // 8) * (2 * sum(cells) - 1)
        )


@pytest.mark.foundation
def test_the_anova_dimension_is_the_separable_equation_solution_count():
    r""":math:`\kappa(\{a,b\}) = 1` and :math:`\kappa(\{a,b,c\}) = n_a+n_b+n_c-1`.

    The two specialisations the theory names, checked against the general
    formula — the two-term case says "a function of :math:`i_y` plus a function
    of :math:`i_x` vanishes ⟹ both are constant", which is why the d=2 kernel
    carries no mesh freedom at all.
    """
    for n_a, n_b in [(2, 2), (3, 4), (5, 7), (1, 9)]:
        assert _anova_dimension((n_a, n_b)) == 1
    for cells in [(2, 2, 2), (3, 4, 5), (2, 3, 7)]:
        assert _anova_dimension(cells) == sum(cells) - 1


# ─────────────────────────────────────────────────────────────────────
# 2. the vectors are in ker A — every one of them, via the production matvec
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.foundation
@pytest.mark.verifies("dd-null-sawtooth")
@pytest.mark.parametrize("label,cells,bcs,kwargs", _SINGULAR,
                         ids=[row[0] for row in _SINGULAR])
def test_EVERY_basis_vector_is_annihilated_by_the_production_matvec(
        label, cells, bcs, kwargs):
    r"""``max ||A phi|| / ||phi||`` over ALL modes — not a sample (`vv` #13).

    What each fixture is here to catch, **measured by mutation** rather than
    asserted (an unwitnessed fixture is `plan-authoring` §6c):

    * **GRADED 1.7x** pins that the face area is a *function of the transverse
      index*, :math:`A_a(i_\perp) = \prod_{b \neq a} h_b(i_b)`. ``[M]``
      replacing :math:`h_b(i_b)` with :math:`h_b(0)` — so the area becomes a
      constant — reddens **exactly these two graded rows and nothing else**.
      ⚠ It does NOT pin the area's *presence*: deleting the factor outright
      reddens all six rows, because the extents ``(1.0, 2.0, 3.0)`` carry
      unequal cell counts, so :math:`h_x \neq h_y` on a uniform mesh too.
    * **LS2 ng=1 / LS4 ng=2 / ng=3 c=0.9** cover the ordinate count, the group
      count and a scattering ratio, all of which cancel out of the derivation
      and must therefore change nothing.
    * **d3 (2,2,3)** carries an ODD cell count, and that is load-bearing:
      ``[M]`` a mutation that drops the :math:`(-1)^{n_a}` far-face sign
      reddens ``(2,2,3)`` but **not** ``(2,2,2)``, where every :math:`n_a` is
      even and the dropped factor is :math:`+1` anyway (`vv` #13's
      congruence-class trap, in this suite's own fixture set).
    """
    mesh = _mesh(cells, bcs, **kwargs)
    gauge = LossKernelGauge.for_mesh(mesh)
    system, template = _loss_system(mesh)
    n_trace = int(np.prod(mesh.angular_trace.shape))

    worst = 0.0
    modes = 0
    for block in gauge.blocks:
        for mode in range(block.basis.table.shape[1]):
            trace = np.zeros(n_trace)
            trace[block.gather.indices] = block.basis.table[:, mode]
            full = _embed_trace(template, trace)
            residual = np.linalg.norm(_apply_loss(system, template, full))
            worst = max(worst, residual / np.linalg.norm(full))
            modes += 1
    assert modes == gauge.dimension
    assert worst < 1e-13, f"{label}: worst residual {worst:.3e} over {modes} modes"


# ─────────────────────────────────────────────────────────────────────
# 3. ⭐ the orthonormality that EARNS the DIAGONAL declaration
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.foundation
@pytest.mark.parametrize("label,cells,bcs,kwargs", _SINGULAR,
                         ids=[row[0] for row in _SINGULAR])
def test_each_block_gram_is_the_identity_which_is_what_earns_DIAGONAL(
        label, cells, bcs, kwargs):
    r"""⭐ :math:`\Phi^{\mathsf T} G \Phi = I` per block — the load-bearing gate.

    :attr:`LossKernelBasis.gram_structure` declares ``DIAGONAL``, and
    :attr:`~orpheus.numerics.frame.FrameBase.gram_inverse` computes its diagonal by the
    row-sum probe ``analysis(reconstruction(ones))``, which equals the true
    diagonal ONLY if :math:`MR` is diagonal. So the declaration is a promise
    about this matrix, and if it were false the projector would be silently
    mis-normalised rather than loudly wrong.

    ⚠ ``[M]`` the raw pair generators — what the derivation produces before the
    :math:`\sqrt G`-weighted SVD — read ``0.000e+00`` off-diagonal at d=2 and
    **``4.05e-01``** at ``(3,4,5)``. The d=2 reading is *vacuous*
    (:math:`\kappa(\{x,y\}) = 1`: one mode per orbit, so the Gram is
    :math:`1\times1`). **The d=3 rows are the only ones that can fail this.**
    """
    mesh = _mesh(cells, bcs, **kwargs)
    gauge = LossKernelGauge.for_mesh(mesh)
    metric = np.asarray(mesh.angular_trace.inner_product_weights, dtype=float)

    saw_multimode = False
    for block in gauge.blocks:
        table = block.basis.table
        assert block.basis.gram_structure is GramStructure.DIAGONAL
        gram = table.T @ (metric[block.gather.indices][:, None] * table)
        saw_multimode |= table.shape[1] > 1
        assert np.allclose(gram, np.eye(table.shape[1]), atol=1e-12), (
            f"{label}: block {block.basis.orbit}/g{block.basis.group} Gram is "
            f"off the identity by {np.abs(gram - np.eye(len(gram))).max():.3e}"
        )
    if len(cells) >= 3:
        assert saw_multimode, (
            "a d=3 fixture must carry a block with >1 mode, or this gate is "
            "the vacuous d=2 reading in disguise"
        )


@pytest.mark.foundation
def test_the_frame_gram_probe_agrees_with_the_true_gram():
    """The frame's own row-sum shortcut returns all ones, as orthonormality implies.

    Closes the loop between the declaration and the machinery that consumes it:
    this asserts the value ``FrameBase.gram_inverse`` actually computes, not the matrix
    we believe it stands for.
    """
    from orpheus.numerics.frame import GalerkinFrame
    from orpheus.numerics.measure import DiscreteMeasure

    mesh = _mesh((2, 2, 2), [(_R, _R)] * 3, ng=1)
    gauge = LossKernelGauge.for_mesh(mesh)
    assert gauge.blocks, "fixture is no longer singular"
    metric = np.asarray(mesh.angular_trace.inner_product_weights, dtype=float)
    for block in gauge.blocks:
        indices = block.gather.indices
        # ⛔ RE-KEYED 2026-09-02 (#429 tracker 2.2). The probe measure named
        # its own manifold (``IndexSet(label="probe")``) while the block's
        # basis eats ``index(sn_trace_orbit(...)_g0)`` — two different point
        # sets, which the frame's new G0 arrow refuses. The probe was never
        # about the LABEL: it integrates the block's own trace metric over the
        # block's own indices, so it must name the basis's manifold. Naming
        # the same object on both halves is what production does
        # (``frame.py:818`` binds one ``points``).
        frame = GalerkinFrame(
            block.basis,
            DiscreteMeasure(nodes=indices.astype(float),
                            weights=metric[indices], support=block.basis.domain),
        )
        diagonal = np.asarray(frame.gram_inverse.diagonal)
        np.testing.assert_allclose(diagonal, np.ones_like(diagonal), atol=1e-12)


# ─────────────────────────────────────────────────────────────────────
# 4. it is a projector, and a G-self-adjoint one
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.foundation
@pytest.mark.parametrize("label,cells,bcs,kwargs", _SINGULAR,
                         ids=[row[0] for row in _SINGULAR])
def test_it_is_an_idempotent_G_self_adjoint_projector(
        label, cells, bcs, kwargs):
    r""":math:`\Pi^2 = \Pi` and :math:`\langle \Pi x, y\rangle_G =
    \langle x, \Pi y\rangle_G`.

    The defining laws of a :math:`G`-orthogonal projector, asserted as laws
    rather than inferred from the construction.
    """
    mesh = _mesh(cells, bcs, **kwargs)
    gauge = LossKernelGauge.for_mesh(mesh)
    metric = np.asarray(mesh.angular_trace.inner_product_weights, dtype=float)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(metric.size)
    y = rng.standard_normal(metric.size)

    projected = gauge.apply(x)
    assert np.linalg.norm(projected) > 1e-6, "degenerate fixture: nothing projected"
    np.testing.assert_allclose(gauge.apply(projected), projected, atol=1e-12)
    assert float(np.sum(metric * gauge.apply(x) * y)) == pytest.approx(
        float(np.sum(metric * x * gauge.apply(y))), rel=1e-11
    )


@pytest.mark.foundation
def test_the_blocks_have_DISJOINT_supports_and_the_type_refuses_otherwise():
    """The direct-sum precondition, and the guard that enforces it.

    A sum of block projectors is a projector only if no two blocks claim the
    same trace DOF; overlapping supports would make ``apply`` order-dependent
    AND non-idempotent, in a way no smoothness check would notice.
    """
    mesh = _mesh((2, 2, 2), [(_R, _R)] * 3, ng=2)
    gauge = LossKernelGauge.for_mesh(mesh)
    assert len(gauge.blocks) > 1
    seen = np.concatenate([b.gather.indices for b in gauge.blocks])
    assert np.unique(seen).size == seen.size

    with pytest.raises(ValueError, match="DISJOINT"):
        LossKernelGauge(
            (gauge.blocks[0], gauge.blocks[0]), mesh.angular_trace)


@pytest.mark.foundation
def test_no_inverse_and_it_is_spelled_by_ABSENCE():
    """A projector onto a proper subspace is not invertible.

    Spelled the house way — ``is_invertible is False`` AND no ``inverse``
    method, so misuse is a static error rather than a runtime one.
    """
    gauge = LossKernelGauge.for_mesh(_mesh((3, 4), [(_R, _R)] * 2))
    assert gauge.is_invertible is False
    assert not hasattr(gauge, "inverse")
    assert gauge.is_adjointable is True
    assert gauge.domain is gauge.codomain


# ─────────────────────────────────────────────────────────────────────
# 5. residual-neutrality — no convergence certificate may move
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.foundation
@pytest.mark.verifies("sn-loss-kernel-gauge-projection")
@pytest.mark.parametrize("label,cells,bcs,kwargs", _SINGULAR,
                         ids=[row[0] for row in _SINGULAR])
def test_gauging_cannot_move_any_convergence_certificate(
        label, cells, bcs, kwargs):
    r""":math:`A(\psi - \Pi\psi) = A\psi`, because :math:`\Pi\psi \in \ker A`.

    Asserted, not assumed. This is what makes the gauge safe to fire at a
    converged exit: the residual, the balance projection and every other
    functional of :math:`A\psi` are bit-unaffected.
    """
    mesh = _mesh(cells, bcs, **kwargs)
    gauge = LossKernelGauge.for_mesh(mesh)
    system, template = _loss_system(mesh)
    rng = np.random.default_rng(1)
    trace = rng.standard_normal(int(np.prod(mesh.angular_trace.shape)))
    bulk = rng.standard_normal(template.interior.values.size)

    raw = np.concatenate([bulk, trace])
    gauged = np.concatenate([bulk, gauge.gauge(trace)])
    before = _apply_loss(system, template, raw)
    after = _apply_loss(system, template, gauged)
    assert np.linalg.norm(after - before) / np.linalg.norm(before) < 1e-13


# ─────────────────────────────────────────────────────────────────────
# 6. ⭐ THE FLAGSHIP — and its negative control
# ─────────────────────────────────────────────────────────────────────
def _uniform_source_fixture(cells, ng=2):
    r"""All-reflective box + uniform isotropic source, whose exact answer is flat.

    :math:`\psi = Q / (\Sigma_t \sum w)` everywhere, bulk and trace — an
    analytic reference that owes nothing to the gauge.
    """
    from orpheus.sn.solver import _build_fixed_source_rhs

    mesh = _mesh(cells, [(_R, _R)] * len(cells), ng=ng)
    system, template = _loss_system(mesh)
    total_weight = float(mesh.quad.weights.sum())
    sig_t = np.asarray(mesh.materials[0].SigT, dtype=float)
    per_group = 1.0 / (total_weight * sig_t)

    interior = np.zeros(template.interior.values.shape)
    interior[:] = 1.0 / total_weight
    source = _build_fixed_source_rhs(interior, mesh)

    n_bulk = template.interior.values.size
    exact = np.zeros(template.to_flat().size)
    block = np.zeros(template.interior.values.shape)
    for g in range(ng):
        block[:, g, ...] = per_group[g]
    exact[:n_bulk] = block.ravel()
    for slot in mesh.angular_trace.layout.faces.values():
        face = np.zeros(slot.shape)
        for g in range(ng):
            face[:, g, ...] = per_group[g]
        exact[n_bulk + slot.offset:
              n_bulk + slot.offset + slot.flat_size] = face.ravel()
    return mesh, system, template, source, exact, n_bulk


def _solve(system, mesh, template, source, schedule):
    from orpheus.numerics.iteration import SourceIteration
    from orpheus.sn.solver import _select_si_splitting

    # Named, not splatted: the selector decides the BOUNDARY half of the
    # splitting only, and the gains are named here the way the driver names
    # them (S, N₂ₙ, boundary gain — §14.1, B LAST), so nothing can mis-bind.
    scattering, n2n, boundary = system.explicit_gains
    base, boundary_gain = _select_si_splitting(
        system.implicit_operator, boundary, mesh, schedule,
    )
    iteration = SourceIteration(base.inverse(), scattering, n2n, boundary_gain,
                                max_iter=400_000, tol=1e-13)
    zero = type(template).from_flat(
        np.zeros(template.to_flat().size), template)
    solution, record = iteration.solve(source, initial_guess=zero)
    assert record.converged, f"{schedule} did not converge"
    return solution.to_flat()


@pytest.mark.foundation
@pytest.mark.parametrize("cells", [(3, 4), (3, 3)])
def test_the_gauge_recovers_the_PHYSICAL_trace_from_a_production_solve(cells):
    r"""⭐ The flagship: Gauss-Seidel's 6 % trace error collapses to round-off.

    The deviation is produced by the **production** boundary Gauss-Seidel
    splitting on a fixture whose exact answer is analytic, so neither side of
    the comparison comes from this module. ``[M]`` at ``(3,4)``:
    ``6.09e-02 -> 1.19e-13``, with **100.0000 %** of the deviation inside the
    closed-form span.
    """
    mesh, system, template, source, exact, n_bulk = _uniform_source_fixture(cells)
    gauge = LossKernelGauge.for_mesh(mesh)
    psi = _solve(system, mesh, template, source, "gauss_seidel")

    reference = np.linalg.norm(exact[n_bulk:])
    deviation = psi[n_bulk:] - exact[n_bulk:]
    before = np.linalg.norm(deviation) / reference
    after = np.linalg.norm(gauge.gauge(psi[n_bulk:]) - exact[n_bulk:]) / reference

    assert before > 1e-3, (
        f"fixture no longer exhibits the defect (deviation {before:.3e}) — if "
        f"the splitting was fixed, this gate has nothing to catch"
    )
    assert after < 1e-11, f"{before:.3e} -> {after:.3e}"
    # The whole deviation is kernel content, so the projector is the right tool.
    in_span = (np.linalg.norm(gauge.apply(deviation))
               / np.linalg.norm(deviation))
    assert in_span == pytest.approx(1.0, abs=1e-6)


@pytest.mark.foundation
def test_it_is_NOT_a_universal_absorber_the_jacobi_control():
    r"""⭐ The negative control (`vv` #19) — Jacobi is already right, and stays right.

    Jacobi lands on the correct member of the solution manifold, so its residual
    deviation is pure round-off and lies essentially OUTSIDE ``ker A``. A
    projector that "improved" it would be absorbing whatever it is handed.
    ``[M]`` ``1.7741e-13 -> 1.7741e-13``, with only 0.3 % of the round-off
    deviation in span.

    Without this leg, the flagship above is satisfied by ``gauge = identity map
    onto the exact answer``, which is not a thing but is exactly the shape of
    error a green-only suite cannot see.
    """
    mesh, system, template, source, exact, n_bulk = _uniform_source_fixture((3, 4))
    gauge = LossKernelGauge.for_mesh(mesh)
    psi = _solve(system, mesh, template, source, "jacobi")

    reference = np.linalg.norm(exact[n_bulk:])
    deviation = psi[n_bulk:] - exact[n_bulk:]
    before = np.linalg.norm(deviation) / reference
    assert before < 1e-11, "jacobi is supposed to already be correct here"

    after = np.linalg.norm(gauge.gauge(psi[n_bulk:]) - exact[n_bulk:]) / reference
    assert after == pytest.approx(before, rel=1e-6), (
        "the gauge moved a solution that was already the canonical member"
    )
    in_span = (np.linalg.norm(gauge.apply(deviation))
               / np.linalg.norm(deviation))
    assert in_span < 0.05, (
        f"{in_span:.4f} of pure round-off reads as kernel content — the "
        f"projector is absorbing more than it should"
    )


# ─────────────────────────────────────────────────────────────────────
# 7. the predicate — every way of NOT having a gauge
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.foundation
def test_one_vacuum_axis_removes_the_freedom():
    """`[M]` (#344) a single vacuum face collapses ``dim ker A`` from 12 to 0.

    An undamped face mode with only one closed axis pair has nowhere to return
    from — the loop does not close.
    """
    mesh = _mesh((3, 4), [(_V, _V), (_R, _R)])
    verdict = gauge_freedom(mesh)
    assert not verdict.present and not verdict.undetermined
    assert "1 reflective axis pair" in verdict.because
    assert LossKernelGauge.for_mesh(mesh).dimension == 0
    assert predicted_kernel_dimension(mesh) == 0


@pytest.mark.foundation
def test_a_damping_closure_removes_the_freedom_at_the_ROOT():
    r"""Linear-discontinuous on the IDENTICAL box has no kernel.

    This is the remedy the warning is obliged to name: the freedom is a property
    of the *closure*, not of the geometry, so changing the discretization
    removes it at the root rather than projecting it out afterwards. ``[M]``
    (#344) ``dim ker A == 0`` for LD on the same all-reflective box where
    diamond gives 12.
    """
    mesh = _mesh((3, 4), [(_R, _R)] * 2, scheme=LinearDiscontinuous())
    verdict = gauge_freedom(mesh)
    assert not verdict.present and not verdict.undetermined
    assert "DAMPS" in verdict.because
    assert mesh.reflective_axis_pairs == 2, (
        "the geometry conjunct must still be satisfied, or this row proves "
        "nothing about the CLOSURE conjunct"
    )
    assert LossKernelGauge.for_mesh(mesh).dimension == 0


@pytest.mark.foundation
def test_one_dimension_has_no_face_mode_and_therefore_no_gauge():
    """``d = 1`` falls out of the spectral radius — no special case anywhere."""
    mesh = _mesh((5,), [(_R, _R)])
    assert not gauge_freedom(mesh).present
    assert LossKernelGauge.for_mesh(mesh).dimension == 0


@pytest.mark.foundation
def test_an_UNDETERMINED_closure_does_NOT_gauge_and_says_why():
    """The third state: unclassified is not "no freedom".

    ``[M]`` LD at ``ndim = 3`` cannot be driven (its ``assemble_inflow_axis``
    handles ``axis in {0, d-1}`` only). The ruled behaviour is to warn loudly
    and NOT gauge — silently treating it as DAMPED would skip the gauge on a
    scheme that was never examined.
    """
    mesh = _mesh((2, 2, 2), [(_R, _R)] * 3, ng=1, scheme=LinearDiscontinuous())
    verdict = gauge_freedom(mesh)
    assert verdict.undetermined and not verdict.present
    assert "could not be classified" in verdict.because
    assert LossKernelGauge.for_mesh(mesh).dimension == 0


@pytest.mark.foundation
def test_the_verdict_refuses_to_be_both_present_and_undetermined():
    """An unclassified closure yields no verdict about its face mode."""
    with pytest.raises(ValueError, match="cannot be both"):
        GaugeFreedom(present=True, undetermined=True, because="impossible")


@pytest.mark.foundation
def test_a_quadrature_not_closed_under_a_mirror_is_REFUSED():
    """A reflective BC needs the mirror partner to exist in the rule.

    Not a limitation of this construction — the reflected ordinate has nowhere
    to land, so the boundary condition itself is unrealisable. Refusing here is
    cheaper than producing modes that are not in ``ker A``.
    """
    from types import SimpleNamespace

    mesh = _mesh((3, 4), [(_R, _R)] * 2)
    mu_x = np.asarray(mesh.quad.mu_x, dtype=float).copy()
    mu_x[0] += 0.25                     # its mirror partner no longer exists

    # A stub carrying exactly what `_reflection_orbits` reads — a real mesh
    # cannot hold this state, which is the point: the refusal must fire before
    # anything downstream trusts the orbit decomposition.
    #
    # ⚠ `reflective_axes` is read from the MESH, not recomputed here, and that
    # is deliberate: the criterion is single-sourced on
    # `SNMesh.reflective_axes` (2026-08-15 — it used to be a twin inside this
    # module). A surrogate must honour the contract it stands in for, so it
    # forwards the real mesh's answer rather than re-deriving one; the thing
    # this stub perturbs is the QUADRATURE, and nothing else.
    broken = SimpleNamespace(
        quad=SimpleNamespace(
            mu_x=mu_x,
            mu_y=np.asarray(mesh.quad.mu_y, dtype=float),
            mu_z=np.asarray(mesh.quad.mu_z, dtype=float),
        ),
        face_labels=mesh.face_labels,
        bc=mesh.bc,
        reflective_axes=mesh.reflective_axes,
    )
    with pytest.raises(ValueError, match="not closed under the axis-0 mirror"):
        _reflection_orbits(broken)  # type: ignore[arg-type]  # deliberate stub

    # …and the unmodified rule is accepted, so the guard is not always-on.
    assert _reflection_orbits(mesh)


# ─────────────────────────────────────────────────────────────────────
# 8. the kernel is geometry-only, and the mesh caches it
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.foundation
def test_the_basis_never_reads_a_CROSS_SECTION():
    """`[M]` (#344) an absorber and a fissile mixture give the identical basis.

    The Stratum-1 claim that licenses caching on the mesh and reusing one build
    across every group, outer and eigenvalue iterate. Asserted bit-exactly:
    every factor in the mode is a mesh, quadrature or boundary quantity, so
    anything less than bit-identity means a cross-section leaked in.
    """
    absorber = LossKernelGauge.for_mesh(_mesh((3, 4), [(_R, _R)] * 2))
    fissile = LossKernelGauge.for_mesh(
        _mesh((3, 4), [(_R, _R)] * 2, fissile=True, c=0.4))

    assert absorber.dimension == fissile.dimension > 0
    for left, right in zip(absorber.blocks, fissile.blocks):
        np.testing.assert_array_equal(left.gather.indices, right.gather.indices)
        np.testing.assert_array_equal(left.basis.table, right.basis.table)


@pytest.mark.foundation
def test_the_mesh_caches_the_gauge():
    """One build per mesh — the setup cost must not ride the per-solve path."""
    mesh = _mesh((3, 4), [(_R, _R)] * 2)
    assert mesh.loss_kernel_gauge is mesh.loss_kernel_gauge
    assert mesh.loss_kernel_gauge.dimension == predicted_kernel_dimension(mesh)


# ─────────────────────────────────────────────────────────────────────
# 9. component T — out of scope, and LEFT ALONE rather than mangled
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.foundation
def test_the_TANGENTIAL_component_is_untouched_not_annihilated():
    r"""``T`` lies in :math:`\ker G`, so :math:`(I - \Pi)t = t` — exactly.

    ``product(4,4)`` puts **half** the trace rows on tangential ordinates, where
    the metric is exactly zero and no minimum-:math:`G`-norm representative
    exists. The right behaviour is to leave them alone, and it is what falls out:
    :math:`Gt = 0` makes :math:`t` :math:`G`-orthogonal to every mode, so the
    projection sees nothing.

    ⚠ A suite using only ``level_symmetric`` (zero tangential rows) is blind to
    this whole class.
    """
    mesh = _mesh((3, 4), [(_R, _R)] * 2,
                 quad=Quadrature.product(n_mu=4, n_phi=4))
    gauge = LossKernelGauge.for_mesh(mesh)
    metric = np.asarray(mesh.angular_trace.inner_product_weights, dtype=float)

    tangential = metric == 0.0
    assert tangential.sum() > 0, "fixture has no tangential rows to test"

    field = np.zeros(metric.size)
    field[tangential] = np.random.default_rng(2).standard_normal(
        int(tangential.sum()))
    np.testing.assert_array_equal(gauge.gauge(field), field)

    # And no block ever claims a zero-metric DOF in the first place.
    for block in gauge.blocks:
        assert np.all(metric[block.gather.indices] > 0.0)


@pytest.mark.foundation
@pytest.mark.parametrize("label, cells, bcs, kwargs", _SINGULAR,
                         ids=[row[0] for row in _SINGULAR])
def test_each_blocks_frame_names_ONE_manifold_on_both_halves(
        label, cells, bcs, kwargs):
    r"""⭐ The production gauge's basis and measure agree on the point set.

    Every block is a :class:`~orpheus.numerics.frame.GalerkinFrame` over a
    :class:`LossKernelBasis` and a
    :class:`~orpheus.numerics.measure.DiscreteMeasure` built four lines apart
    in ``_build_gauge_blocks``. Until 2026-09-01 the two named the same point
    set in **two spellings** — the measure tagged the bare label
    ``sn_trace_orbit(…)_g…`` while the basis wrapped it as ``index(…)`` — and
    tracker 2.1 pinned that divergence in
    ``tests/numerics/test_basis_domain.py::test_d6``, explicitly so that
    tracker 2.0c *"must come back here and cannot resolve it by accident"*.

    ⛔ **It was resolved by accident, and this gate is why that is now
    visible.** 2.0c made the measure read ``support=basis.domain``, which
    closes the divergence by construction — and a mutation battery then
    measured the repair as **BLIND**: replacing that expression with a wrong
    ``IndexSet`` reddened **nothing tree-wide**, because ``test_d6`` asserts
    the *basis*'s half and no gate reached the production *measure*'s.

    The observable path is the block's own operator wiring: the gather's
    codomain IS ``measure.space``, whose name is ``L2[{support.name}]``. So
    this compares the two halves through production plumbing rather than by
    re-reading the constructor. `[M]` the mutation now reddens every row.

    (``vv-principles`` #17: the verdict of a battery is a TABLE, and an arm
    that reddens nothing is a claim with no witness — not a claim that holds.)
    """
    gauge = LossKernelGauge.for_mesh(_mesh(cells, bcs, **kwargs))
    assert gauge.blocks, f"{label}: no blocks — the fixture is not singular"
    for block in gauge.blocks:
        domain = block.basis.domain
        assert isinstance(domain, IndexSet)
        assert block.gather.codomain is not None
        assert block.gather.codomain.name == f"L2[{domain.name}]", (
            f"{label}: block {block.basis.orbit}/g{block.basis.group} — the "
            f"basis says its functions live on {domain.name!r} while its "
            f"measure induced {block.gather.codomain.name!r}. One frame, two "
            f"manifolds."
        )
