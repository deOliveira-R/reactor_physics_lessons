r"""Foundation gates for the two splitting-iteration primitives — #448.

``lagged_source`` and ``fixed_point_step`` entered
``orpheus.numerics.iteration.__all__`` with the #448 carve, as the ONE
spelling of the map every SN driver iterates:

.. math::

   G(\psi) \;=\; M^{-1}\Bigl(q_{\rm ext} + \sum_i N_i\,\psi\Bigr),
   \qquad A = M - \sum_i N_i .

``SourceIteration.solve`` composes ``lagged_source`` to convergence; the SN
eigenvalue finalize and the windowed fixed-source finalize evaluate
``fixed_point_step`` **once** on a converged iterate, which is what makes the
returned flux solve the equation the caller is told about.  Before #448 the
eigenvalue finalize hand-rolled that source and the twin drifted — the whole
of ERR-083.

⛔ **They had ZERO direct call sites in ``tests/`` when they landed** (qa F-6).
Every gate on them was end-to-end, through an SN solve, so the primitives'
own contract — what ``lagged_source`` returns, what ``fixed_point_step``
composes, that the iterate is threaded as ``initial_guess`` — was pinned only
by the answer of a 20-cell transport solve.  This module is the term tier.

────────────────────────────────────────────────────────────────────────────
CLAIM LEDGER
────────────────────────────────────────────────────────────────────────────
========================================  =============  ==========================
row                                       kind           reference
========================================  =============  ==========================
``…lagged_source_is_q_plus_the_gains``    IDENTITY       hand-written ``q + ΣNψ``
``…step_is_the_inverse_of_that_source``   IDENTITY       ``inv(M) @ rhs`` (a second
                                                         LAPACK route, stated as
                                                         procedural not structural)
``…a_fixed_point_is_a_fixed_point``       **THEOREM**    ``solve(M − ΣN, q)`` — the
                                                         ASSEMBLED equation; the
                                                         only structurally
                                                         independent row here
``…a_non_fixed_point_MOVES``              ACTIVATION     vv #19 — the row above is
                                                         green for a broken step
                                                         that returns its argument
``…the_iterate_is_threaded_as_a_seed``    CONTRACT       a recording double
``…source_iteration_drives_the_same…``    SINGLE-SOURCE  a spy on the shared body
========================================  =============  ==========================

⚠ **What the third row buys and the second does not.**  Row 2's reference
(``inv(M) @ rhs``) is *procedurally* independent of the SUT's route
(``getrs`` via ``np.linalg.solve``) and NOT structurally independent — both
invert the same matrix I handed them.  It pins the COMPOSITION (that the
inverse is applied to the lagged source and not to ``q_ext`` alone), which is
the only thing a 3-line composition primitive can get wrong.  Row 3 is the
real claim: it solves a DIFFERENT equation (the assembled ``A = M − ΣN``, one
dense solve) and requires the splitting algebra to be right for the two to
meet.  A swapped sign on the gains, a dropped gain, or ``M`` in place of
``A`` all separate them by O(1).

`[M]` 2026-09-06: this module runs in **0.06 s** (5×5 dense, one seed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from orpheus.numerics import iteration as _iteration
from orpheus.numerics.iteration import (
    SourceIteration,
    fixed_point_step,
    lagged_source,
)
from orpheus.numerics.operator import InverseOperator, LinearOperator

if TYPE_CHECKING:  # the annotation only — no runtime import cost
    from orpheus.numerics.space import FunctionSpace

_N = 5
_SEED = 20260906


class _Matrix(LinearOperator):
    """Dense-matrix double — the same shape ``test_iteration.py`` uses."""

    @property
    def domain(self) -> "FunctionSpace | None":
        """A deliberately UNBOUND probe: it DECLARES the unbound state.

        The S4-amendment's fourth sanctioned answer (a documented
        ``Optional``): this double carries a bare matrix and no space, and
        says so, rather than inheriting a silence the base no longer
        supplies.
        """
        return None

    @property
    def codomain(self) -> "FunctionSpace | None":
        """See :attr:`domain` — unbound by declaration."""
        return None

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=float)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.matrix @ x

    def solve(self, b: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.matrix, b)

    @property
    def is_invertible(self) -> bool:
        return True

    def inverse(self) -> InverseOperator:
        return InverseOperator(self)


class _RecordingInverse:
    """A ``SupportsSeededApply`` double that records what it was handed.

    Deliberately NOT a ``LinearOperator``: the primitive's static contract is
    the seeded-apply signature alone, so the double should satisfy exactly
    that and nothing more — a double that is *more* than the contract cannot
    show the contract is enough.
    """

    def __init__(self, inner: _Matrix) -> None:
        self.inner = inner
        self.calls: list[tuple[np.ndarray, object]] = []

    def apply(self, x: Any, /, *, initial_guess: Any | None = None) -> Any:
        self.calls.append((np.array(x, dtype=float, copy=True), initial_guess))
        return self.inner.solve(x)


@pytest.fixture
def parts():
    """``(M, gains, q, psi)`` — a strictly diagonally dominant ``M`` and two
    genuinely different gains, so no accidental symmetry can make a dropped
    or swapped gain invisible."""
    rng = np.random.default_rng(_SEED)
    m = rng.uniform(-1.0, 1.0, size=(_N, _N)) + 6.0 * np.eye(_N)
    n1 = rng.uniform(-0.4, 0.4, size=(_N, _N))
    n2 = rng.uniform(-0.3, 0.3, size=(_N, _N))
    assert not np.allclose(n1, n2), "the two gains must differ"
    return (
        _Matrix(m),
        (_Matrix(n1), _Matrix(n2)),
        rng.uniform(0.5, 1.5, size=_N),
        rng.uniform(0.5, 1.5, size=_N),
    )


# ══════════════════════════════════════════════════════════════════════
# lagged_source
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.foundation
def test_lagged_source_is_q_plus_the_gain_actions(parts):
    r"""``lagged_source(q, gains, ψ) == q + Σᵢ Nᵢ ψ``."""
    m, gains, q, psi = parts
    del m
    expected = q + gains[0].matrix @ psi + gains[1].matrix @ psi
    np.testing.assert_allclose(
        lagged_source(q, gains, psi), expected, rtol=1e-14, atol=0.0,
        err_msg="lagged_source is not q + Σ Nᵢψ",
    )


@pytest.mark.foundation
def test_lagged_source_with_no_gains_returns_the_source_ITSELF(parts):
    """An empty splitting has nothing to lag: the rhs **is** ``q_ext``.

    Two legs, and the second is the one with teeth.  *Equality* says the body
    added nothing — but equality alone is satisfied by any body that
    allocates a copy, so on its own this row is unreddenable by any realistic
    mutation (`[M]` 2026-09-06: none of arms A1–A7 moved it).  *Identity*
    says the no-gain path is a pure pass-through, which is what makes
    ``SourceIteration`` with no gains a genuine no-op wrapper around its
    inverse rather than a per-pass allocation.

    ⚠ A future body that copies defensively must DELETE the identity leg
    deliberately and say why — it must not be relaxed to make a red go away.
    """
    _, _, q, psi = parts
    out = lagged_source(q, (), psi)
    np.testing.assert_array_equal(out, q)
    assert out is q, (
        "lagged_source allocated on the no-gain path — the rhs should be "
        "q_ext itself.  If this is a deliberate defensive copy, retire this "
        "leg with a note rather than loosening it."
    )


@pytest.mark.foundation
def test_lagged_source_does_not_mutate_its_arguments(parts):
    """``q_ext`` and ``ψ`` survive the call unchanged.

    ``SourceIteration`` calls it once per pass on the SAME ``q_ext`` object,
    so an in-place accumulation would silently compound the source across
    iterations — a defect no end-to-end value gate distinguishes from slow
    convergence.
    """
    m, gains, q, psi = parts
    del m
    q_before, psi_before = q.copy(), psi.copy()
    lagged_source(q, gains, psi)
    np.testing.assert_array_equal(q, q_before)
    np.testing.assert_array_equal(psi, psi_before)


# ══════════════════════════════════════════════════════════════════════
# fixed_point_step
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.foundation
def test_fixed_point_step_is_the_inverse_of_the_lagged_source(parts):
    r"""``G(ψ) == M⁻¹ (q + Σ Nᵢψ)`` — the COMPOSITION.

    See the module docstring: the reference here is procedurally, not
    structurally, independent.  What it pins is that the inverse is applied
    to the LAGGED source — a step that applied it to ``q_ext`` alone, or that
    lagged after inverting, separates by O(1).
    """
    m, gains, q, psi = parts
    rhs = q + gains[0].matrix @ psi + gains[1].matrix @ psi
    expected = np.linalg.inv(m.matrix) @ rhs
    np.testing.assert_allclose(
        fixed_point_step(m.inverse(), gains, q, psi), expected,
        rtol=1e-12, atol=0.0,
        err_msg="fixed_point_step is not M⁻¹ applied to the lagged source",
    )


@pytest.mark.foundation
def test_a_fixed_point_of_the_split_equation_is_a_fixed_point_of_the_step(
    parts,
):
    r"""``G(ψ*) == ψ*`` where ``(M − Σ Nᵢ) ψ* = q``.

    **The load-bearing row.**  The reference assembles ``A = M − Σ Nᵢ`` and
    solves it ONCE; the SUT never forms ``A`` at all.  The two meet only if
    the splitting algebra is right, so a sign flip on the gains, a dropped
    gain, or ``A`` in place of ``M`` all redden it — and this is exactly the
    property the SN finalizes rely on when they evaluate the map once on a
    converged iterate (#448 / ERR-083).
    """
    m, gains, q, _ = parts
    a = m.matrix - gains[0].matrix - gains[1].matrix
    psi_star = np.linalg.solve(a, q)
    np.testing.assert_allclose(
        fixed_point_step(m.inverse(), gains, q, psi_star), psi_star,
        rtol=1e-11, atol=1e-13,
        err_msg=(
            "one step from the fixed point of (M − ΣN)ψ = q did not return "
            "it — the primitive's splitting is not the one the reference "
            "assembled."
        ),
    )


@pytest.mark.foundation
def test_the_step_MOVES_a_non_fixed_point(parts):
    """vv #19: the row above is green for a step that returns its argument.

    A ``fixed_point_step`` implemented as ``return psi`` satisfies
    ``G(ψ*) == ψ*`` perfectly.  This is the negative leg that makes the
    positive one informative: away from the fixed point the map must move
    the iterate by a margin far above the tolerance the row above uses.
    """
    m, gains, q, psi = parts
    a = m.matrix - gains[0].matrix - gains[1].matrix
    psi_star = np.linalg.solve(a, q)
    assert float(np.max(np.abs(psi - psi_star))) > 1e-2, (
        "the fixture's ψ happens to BE the fixed point — this row and the "
        "one above would then make the same assertion."
    )
    moved = fixed_point_step(m.inverse(), gains, q, psi)
    rel = float(np.max(np.abs(moved - psi)) / np.max(np.abs(psi)))
    assert rel > 1e-3, (
        f"the step moved a non-fixed point by only {rel:.3e} — it is not "
        f"acting, so the fixed-point row proves nothing."
    )


@pytest.mark.foundation
def test_the_iterate_is_threaded_as_the_initial_guess(parts):
    """The primitive seeds the apply with the point it is evaluated at.

    The contract is **threaded, may be ignored**: an exact
    ``InverseOperator`` ``del``\\ s the kwarg, while ``SweepOperator`` reads
    it into the curvilinear Carlson closure and the reflective partner
    trace.  So the pin is on the CALL, not on any effect — which is why it
    needs a recording double rather than a value comparison (vv Mode 26: a
    claim about the route cannot be gated by asserting the output).
    """
    m, gains, q, psi = parts
    spy = _RecordingInverse(m)
    fixed_point_step(spy, gains, q, psi)
    assert len(spy.calls) == 1, f"expected one apply, got {len(spy.calls)}"
    handed_rhs, guess = spy.calls[0]
    assert guess is psi, (
        "fixed_point_step did not thread the iterate as initial_guess — the "
        "curvilinear ψ½ seed and the reflective partner trace travel through "
        "that argument, so a dropped kwarg is a silent seed loss."
    )
    np.testing.assert_allclose(
        handed_rhs, lagged_source(q, gains, psi), rtol=1e-14, atol=0.0,
        err_msg="the rhs handed to the inverse is not the lagged source",
    )


# ══════════════════════════════════════════════════════════════════════
# The single-source claim
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.foundation
def test_source_iteration_drives_the_same_lagged_source(parts, monkeypatch):
    r"""``SourceIteration.solve``'s rhs IS ``lagged_source``'s.

    The #448 architecture rests on one sentence: *the source the
    reconstruction sees IS the source the iteration converged against*.  That
    is only true while both spell it through the same body — so this pins the
    call rather than re-deriving the value, and it reddens the day someone
    inlines the accumulation back into the loop "for speed".

    The spy wraps the module-global the loop resolves at call time, and
    asserts its own installation: a wrapper that binds nothing would report a
    clean zero (lessons L46e).
    """
    m, gains, q, psi = parts
    seen: list[np.ndarray] = []
    original = _iteration.lagged_source

    def recording(q_ext, gs, x):
        out = original(q_ext, gs, x)
        seen.append(np.array(out, dtype=float, copy=True))
        return out

    monkeypatch.setattr(_iteration, "lagged_source", recording)

    si = SourceIteration(
        m.inverse(), *gains, max_iter=200, tol=1e-13, budget_name="max_inner",
    )
    _, record = si.solve(q, initial_guess=psi)

    assert seen, (
        "SourceIteration.solve called lagged_source ZERO times — either the "
        "loop stopped sharing the body (the #448 single-source claim is "
        "false) or the spy bound nothing."
    )
    assert len(seen) == record.n_iterations, (
        f"lagged_source ran {len(seen)}× for {record.n_iterations} passes — "
        f"the loop no longer assembles its rhs once per pass."
    )
    np.testing.assert_allclose(
        seen[0], lagged_source(q, gains, psi), rtol=1e-14, atol=0.0,
        err_msg=(
            "the driver's FIRST rhs differs from lagged_source on its own "
            "initial guess — the loop and the finalize do not see the same "
            "source, which is the drift #448 was."
        ),
    )
