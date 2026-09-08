r"""The loss operator's kernel, in closed form, and the gauge that fixes it.

On an all-reflective Cartesian box closed by **diamond differencing**, the
within-group loss operator

.. math::

    A \;=\; L + C - S - B

is **exactly singular**, so the boundary trace it returns is a function of the
cold start rather than of the problem.  ``[M]`` (#344) three cold starts
differing only inside :math:`\ker A` converge to identical ``n_iter``, identical
residual :math:`8.54\times10^{-14}` and a bulk that is bit-stable at
:math:`7\times10^{-16}` — while the returned trace differs by up to **27.3 %**.
Both convergence functionals are blind to it (residual :math:`7.3\times10^{-16}`,
balance projection :math:`1.8\times10^{-16}`).

This module builds the kernel **in closed form** — no eigensolve, no SVD of
:math:`A` — and the :math:`G`-orthogonal projector onto it, so the solver can
return the one canonical member.

Why the gauge is CANONICAL, not conventional
--------------------------------------------

``[M]`` the exact solution is :math:`G`-orthogonal to :math:`\ker A`
(:math:`1.27\times10^{-15}`), and that is a **theorem**, not a fixture: every
kernel mode carries a non-trivial sign character on every axis of :math:`S`
(§3 below), so any *mirror-even* functional annihilates it.  The
minimum-:math:`\|\cdot\|_G` member of the solution manifold therefore **is** the
physical answer — the projection recovers it rather than picking a convention.

1. Why a kernel exists at all — DD's face involution
-----------------------------------------------------

``[M]`` the bulk share of the null projector is :math:`1.1\times10^{-28}`, so
take :math:`\psi_c \equiv 0` and read off what remains of diamond's closure and
the cell balance:

.. math::

    \psi_{\mathrm{out},a} = 2\psi_c - \psi_{\mathrm{in},a}
    \;\;\xrightarrow{\;\psi_c = 0\;}\;\;
    \psi_{\mathrm{out},a} = -\psi_{\mathrm{in},a}

The face-to-face transmission carries eigenvalue :math:`-1` — undamped — on the
cell-average-blind subspace, so the mode drives :math:`\psi_c = 0` and the
absorption term :math:`\Sigma_t V \psi_c` never sees it.  Along every mesh line
the face values alternate and the whole field collapses to one function per
axis, the **sawtooth**

.. math::

    \psi^n_a(k, i_\perp) \;=\; (-1)^k \, \varphi^n_a(i_\perp)

with :math:`k` the face index along axis :math:`a`.  This is why the kernel is a
*trace* object, and why
:class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`
— which carries the in-cell slope, so :math:`\psi_{\rm out} = -\psi_{\rm in}` is
no longer compatible with a zero cell moment — has ``dim ker A == 0`` on the
identical box.  The scheme is **asked**, never tabulated:
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.face_transmission_spectrum`.

2. One substitution empties the problem of physics
---------------------------------------------------

Insert the sawtooth into the cell balance and absorb the coefficients:

.. math::

    Y^n_a(i_\perp) \;:=\; |\mu^n_a| \; A_a(i_\perp) \;
                          (-1)^{\sum_{b \neq a} i_b} \; \varphi^n_a(i_\perp)

Every factor depends only on :math:`i_\perp` — the face area
:math:`A_a = \prod_{b \neq a} h_b` included, **which is why a graded mesh needs
no separate treatment**.  The balance becomes

.. math::

    \sum_{a \in S} s_a \, Y_a(s_{\neq a}; i_{\neq a}) \;=\; 0
    \qquad s_a = \operatorname{sign}\mu^n_a

*"a sum of functions, each blind to one coordinate and one sign, vanishing
identically."*  The cross-sections, mesh spacings, quadrature weights and
scattering ratio have all cancelled — which is the structural reason
``dim ker A`` is mesh-independent at :math:`d = 2`, independent of :math:`c`,
and exactly proportional to ``ng``.  ``[M]`` an absorber and a fissile mixture
on the same box give bit-identical residuals (:math:`2.799\times10^{-16}`), so
the basis is a **Stratum-1** (geometry-only) object.

:math:`S` is the set of axes that are simultaneously **reflective** and
**non-tangential** for that ordinate: a vacuum axis drops out because its
:math:`\varphi \equiv 0`, a tangential axis because its :math:`|\mu_a|` is zero.

3. Solving it — characters, then pair generators
-------------------------------------------------

Expanding each :math:`Y_a` in the sign characters
:math:`\chi_T(s) = \prod_{b \in T} s_b` splits the equation into one per subset
:math:`U \subseteq S`.  :math:`|U| \le 1` contributes nothing; :math:`|U| \ge 2`
is the classical additive-separable (ANOVA) problem, whose solution space is
spanned by **pair generators**: an axis pair :math:`\{a,b\} \subseteq S`, a
character :math:`U` with :math:`\{a,b\} \subseteq U \subseteq S`, and an index
tuple :math:`j` on the remaining axes.

    Every mode carries a non-trivial character on every axis of :math:`S` —
    that is the blindness theorem of the header, read off the construction.

Both counting laws follow as theorems:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - configuration
     - ``dim ker A``
   * - :math:`d = 2`, both axes reflective
     - :math:`n_g \cdot N/4` — **mesh-independent**
   * - :math:`d = 3`, all reflective
     - :math:`n_g \cdot (N/8) \cdot (2\sum_i n_i - 1)`
   * - :math:`d = 3`, two reflective pairs + a vacuum axis :math:`c`
     - :math:`n_g \cdot (N/4) \cdot n_c`
   * - :math:`d = 1`, or any orbit with :math:`|S| \le 1`
     - **0**

4. The object is BLOCKED, and the blocks are the algebra
---------------------------------------------------------

Every mode is supported on exactly one **(ordinate orbit, group)** pair, and
those supports are **disjoint**.  So the kernel is a direct sum

.. math::

    \ker A \;=\; \bigoplus_b V_b ,
    \qquad
    \Pi \;=\; \sum_b \gamma_b^{\mathsf T} \, \Pi_b \, \gamma_b

over :math:`n_{\rm orbits} \cdot n_g` blocks, with :math:`\gamma_b` the
:class:`~orpheus.numerics.operator.TraceRestrictionOperator` gather onto block
:math:`b`'s degrees of freedom and :math:`\gamma_b^{\mathsf T}` its scatter.
This is not a storage optimisation — it is the reason the Gram is block
diagonal at all.  ``[M]`` the blocked projector holds **150 MiB** at
``(12,12,12)`` S8 ``ng=4`` where a dense basis would be **17.6 GiB**, and spans
the same subspace (gap :math:`1.65\times10^{-14}` at 138 dimensions).

5. Each block is G-ORTHONORMAL, and that is what makes the frame route legal
-----------------------------------------------------------------------------

⚠ The pair generators are **not** :math:`G`-orthogonal at :math:`d \ge 3` —
``[M]`` :math:`\max|{\rm offdiag}| / \max|{\rm diag}|` of :math:`B^{\mathsf T}GB`
reads ``0.000e+00`` at :math:`d = 2` but **``1.72e-01``** at ``(2,2,2)`` and
**``4.05e-01``** at ``(3,4,5)``.  The :math:`d = 2` reading is *vacuous*:
:math:`\kappa(\{x,y\}) = 1`, so each orbit carries exactly ONE mode and a
:math:`1\times1` Gram is diagonal for free.  At :math:`d \ge 3` an orbit carries
:math:`2\sum n - 1` modes and the :math:`\{a,b\}` and :math:`\{a,c\}` generators
share the :math:`a` faces.

That matters because :attr:`~orpheus.numerics.frame.FrameBase.gram_inverse` computes its
diagonal by the row-sum probe ``analysis(reconstruction(ones))``, which equals
the true diagonal **only** if :math:`MR` is diagonal.  Declaring
:attr:`~orpheus.numerics.basis.GramStructure.DIAGONAL` on a 43 %-off-diagonal
Gram would normalise every coefficient by the wrong number, silently.

⟹ **a** :class:`LossKernelBasis` **IS the** :math:`G`-**orthonormal basis**; the
pair generators are its construction, not its content.  One
:math:`\sqrt{G}`-weighted SVD per block does the rank reduction and the
orthonormalisation together — replacing *both* factorisations the derivation
memo used (a coefficient-space SVD **and** a state-space QR) — after which
:math:`\Phi^{\mathsf T} G \Phi = I`, ``DIAGONAL`` is true by construction, and
:math:`G^{-1}` is the identity.

.. note::

    The rank that SVD finds **is** the counting law of §3, computed by a
    completely different route.  :func:`predicted_kernel_dimension` evaluates the
    law combinatorially without building a single vector, so the two can be — and
    are — checked against each other.

6. Scope: component R only
---------------------------

``ker A`` splits as :math:`T \oplus R`, separated exactly by the metric's
zero-set (``[M]`` :math:`\max|B_R|` on ``G == 0`` rows is ``0.000000e+00`` on
``level_symmetric``, ``product(4,4)`` AND ``lebedev(11)``):

* **R** — the genuine trace underdetermination, on current-carrying ordinates.
  Its modes carry non-zero trace metric, so the minimum-:math:`G`-norm member is
  unique and this module gauges it.
* **T** — the tangential slots (:math:`\Omega\cdot\hat n = 0`), whose rows AND
  columns of :math:`A` are identically zero.  ⛔ **They lie in** :math:`\ker G`,
  so *every* value has the same (zero) :math:`G`-norm and there is no
  minimum-norm representative to choose.  :math:`B^{\mathsf T} G B` is singular
  the moment :math:`T \neq 0`, so R and T must never be orthonormalised
  together.  T is out of scope here and is left **untouched** by the projection
  (which is the correct action, not an omission: :math:`Gt = 0` makes
  :math:`t \perp_G \operatorname{span} R`, so :math:`(I - \Pi)t = t`).
  On the default ``level_symmetric`` path ``T`` is empty.

The physics, the consequence a user observes, the parity rule, the evidence
and the remedy hierarchy are the theory-page half of this module:
:ref:`sn-loss-kernel-gauge` (in :doc:`/theory/methods/sn/cartesian_multid`,
one hop downstream of the local face-mode spectrum it closes), with the exit
behaviour at :ref:`sn-exit-gauge`.  GitHub #344.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from itertools import combinations, product as iter_product
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from orpheus.numerics.basis import Basis, GramStructure
from orpheus.numerics.face_layout import face_normal
from orpheus.numerics.frame import GalerkinFrame
from orpheus.numerics.manifold import IndexSet, Manifold
from orpheus.numerics.measure import DiscreteMeasure
from orpheus.numerics.operator import (
    LinearOperator,
    TraceRestrictionOperator,
)
from orpheus.numerics.space import FunctionSpace
from orpheus.transport.spatial.scheme import FaceModeDamping

if TYPE_CHECKING:  # pragma: no cover - typing only
    from orpheus.sn.mesh.augmented_mesh import SNMesh

__all__ = [
    "GAUGE_ESCALATION_FLAG",
    "GaugeFreedom",
    "GaugeFreedomWarning",
    "LossKernelBasis",
    "LossKernelGauge",
    "gauge_freedom",
    "predicted_kernel_dimension",
    "warn_if_gauge_freedom",
]

#: Below this relative correction the returned trace was ALREADY the canonical
#: member, so the gauge did nothing and there is nothing to tell the user.
#:
#: Not a tuning knob — the two populations are separated by **13 orders**.
#: ``[M]`` on `solve_sn`, all-reflective LS4 2-group fissile, ``gauss_seidel``:
#: an excited configuration reads ``4.1e-02 .. 7.8e-02`` (5 of 11 meshes) and an
#: unexcited one ``2.1e-15 .. 1.0e-14`` (6 of 11), with ``jacobi`` at ``~1e-15``
#: throughout. Any threshold in ``[1e-12, 1e-4]`` gives the same verdict.
_GAUGE_AUDIBLE_FLOOR = 1e-10

#: A direction cosine below this is TANGENTIAL to that axis, so the axis leaves
#: the ordinate's active set ``S``. Matches the trace layer's own threshold
#: vocabulary (``angular_trace_space.TANGENTIAL_EPS``) rather than minting a
#: second one; every shipped rule's tangential cosine is exactly ``0.0`` (#325).
_TANGENTIAL_MU = 4.0 * float(np.finfo(np.float64).eps)

#: Relative singular-value cut for the per-block rank. Justified by the MEASURED
#: gap, not by taste: ``[M]`` the retained singular values span only
#: ``9.326e+01 .. 4.227e+01`` at ``(3,4,5)`` S4 ``ng=2`` (a ratio of 2.2) while
#: the discarded ones are at round-off, so any cut in ``[1e-12, 1e-2]`` returns
#: the same rank — and the rank is independently checked against the counting
#: law by :func:`predicted_kernel_dimension`.
_RANK_RTOL = 1e-10


# ─────────────────────────────────────────────────────────────────────
# The warning — the freedom must be AUDIBLE, and the root fix named
# ─────────────────────────────────────────────────────────────────────
class GaugeFreedomWarning(RuntimeWarning):
    r"""The returned trace was repaired, or could not be classified.

    ⚠ **Deliberately NOT a**
    :class:`~orpheus.numerics.convergence.ConvergenceWarning`. That family
    means *"an iterative solve exhausted its budget; the answer is
    best-effort"*, and this is the opposite situation: the solve converged
    perfectly, and the ambiguity is in the **equation**, not the iteration.
    `[M]` the configuration where this fires hardest reports
    ``fully_converged = True`` and ``balance_defect = None``. Reusing that
    category would also make every user who escalates
    :data:`~orpheus.numerics.convergence.ESCALATION_FLAG` start failing on an
    unrelated condition.

    **Why this is worth a warning rather than a doc note.** On an
    all-reflective diamond-difference box the loss operator is exactly
    singular, so the returned trace is one member of a solution *manifold*.
    Every mirror-EVEN functional of it is blind to which member (a theorem —
    see the module docstring), so a user checking currents, leakage or
    :math:`k` sees nothing wrong. `[M]` what they would see, if they looked
    at the one class of functional that is not blind, is a **spurious ~7 %
    net current flowing sideways along a mirror face** — a quantity that
    cannot physically exist.

    And it is not predictable by inspection: `[M]` the excitation is a parity
    effect, present at ``n_x = 3`` and absent at ``n_x = 4``, so it appears
    and vanishes under a mesh change that alters nothing qualitative.

    Escalate to a hard failure with :data:`GAUGE_ESCALATION_FLAG` (the
    category must be DOTTED — ``-W`` resolves an undotted one against
    ``builtins``, so the short spelling is not a filter at all).
    """


#: The CI escalation recipe as a VALUE, derived from the class rather than
#: retyped — a module move or rename cannot leave it pointing at nothing.
#: Mirrors :data:`~orpheus.numerics.convergence.ESCALATION_FLAG`, including the
#: reason its short spelling silently gates nothing (#340, 2026-08-09).
GAUGE_ESCALATION_FLAG = (
    f"-W error::{GaugeFreedomWarning.__module__}."
    f"{GaugeFreedomWarning.__qualname__}"
)


# ─────────────────────────────────────────────────────────────────────
# The verdict — is there a gauge to fix?
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class GaugeFreedom:
    """Whether ``A`` has a null space on this configuration, and why.

    Three states, because "we could not classify the closure" is genuinely
    distinct from "there is no freedom" and must never be silently merged into
    it — a caller that read UNDETERMINED as ABSENT would skip the gauge on a
    scheme it never examined, which is exactly the blindness the derived
    predicate exists to remove.

    Attributes
    ----------
    present : bool
        ``True`` iff the closure leaves a face mode undamped AND the problem
        closes at least two reflective axis pairs. Both conjuncts are DERIVED —
        nothing here is tabulated per scheme or per geometry.
    undetermined : bool
        The closure could not be driven, so its damping is unknown. The caller
        must warn and must NOT gauge (user ruling, 2026-08-14).
    because : str
        A sentence naming the deciding conjunct, for the warning to quote.
    """

    present: bool
    undetermined: bool
    because: str

    def __post_init__(self) -> None:
        if self.present and self.undetermined:
            raise ValueError(
                "GaugeFreedom cannot be both present and undetermined — an "
                "unclassified closure yields no verdict about its face mode, "
                f"so `present` must be False. Got because={self.because!r}."
            )


def gauge_freedom(sn_mesh: "SNMesh") -> GaugeFreedom:
    r"""Does this configuration admit gauge freedom in :math:`\ker A`?

    .. math::

        \text{freedom} \iff
            \underbrace{\text{the closure leaves a face mode undamped}}
                       {\text{§1 — asked of the scheme}}
            \;\wedge\;
            \underbrace{\text{the problem closes} \ge 2
                        \text{ reflective axis pairs}}
                       {\text{§2 — asked of the mesh}}

    Both halves are **derived, never tabulated** — the closure answers from its
    own :meth:`cell_kernel_batch` coefficients and the geometry answers from the
    realized boundary laws.  A scheme added tomorrow answers for itself with no
    edit here, which is what makes "change the discretization" a real remedy
    rather than a coincidence.

    An undamped mode with only ONE closed axis pair still has nowhere to go:
    ``[M]`` at :math:`d = 2` a single vacuum face collapses ``dim ker A`` from
    12 to 0.
    """
    spectrum = sn_mesh.scheme.face_transmission_spectrum(sn_mesh.ndim)
    pairs = sn_mesh.reflective_axis_pairs

    if spectrum.damping is FaceModeDamping.UNDETERMINED:
        return GaugeFreedom(
            present=False,
            undetermined=True,
            because=(
                f"the {type(sn_mesh.scheme).__name__} closure could not be "
                f"classified at ndim={sn_mesh.ndim} "
                f"({spectrum.undetermined_because}), so whether it leaves a "
                f"face mode undamped is unknown; the problem closes {pairs} "
                f"reflective axis pair(s)"
            ),
        )

    if spectrum.damping is FaceModeDamping.DAMPED:
        return GaugeFreedom(
            present=False,
            undetermined=False,
            because=(
                f"the {type(sn_mesh.scheme).__name__} closure DAMPS every face "
                f"mode at ndim={sn_mesh.ndim} (spectral radius "
                f"{spectrum.spectral_radius:.6f} < 1), so no trace mode can "
                f"survive a round trip"
            ),
        )

    if pairs < 2:
        return GaugeFreedom(
            present=False,
            undetermined=False,
            because=(
                f"the closure leaves a face mode undamped (spectral radius "
                f"{spectrum.spectral_radius:.6f}) but the problem closes only "
                f"{pairs} reflective axis pair(s); a mode needs two to return "
                f"to itself"
            ),
        )

    return GaugeFreedom(
        present=True,
        undetermined=False,
        because=(
            f"the {type(sn_mesh.scheme).__name__} closure leaves a face mode "
            f"undamped (spectral radius {spectrum.spectral_radius:.6f}) and "
            f"the problem closes {pairs} reflective axis pairs"
        ),
    )


def _damping_alternatives(ndim: int) -> str:
    """The shipped closures that would remove the freedom at the root.

    ⭐ **ASKED of the registry, never tabulated** — the same ruling that shapes
    :func:`gauge_freedom`. A closure added tomorrow appears in this sentence
    with no edit here, and one that stops damping disappears from it; a hand-
    written list would rot in exactly the direction that matters (naming a
    remedy that is not one).
    """
    from orpheus.transport.spatial.scheme import DiscretizationSchemeBase

    damping = sorted(
        key
        for key, scheme_type in DiscretizationSchemeBase.registry.items()
        if scheme_type().face_transmission_spectrum(ndim).damping
        is FaceModeDamping.DAMPED
    )
    if not damping:
        return (
            "no closure registered in this build damps the face mode at "
            f"ndim={ndim}, so the only root fix here is to break a reflective "
            "axis pair"
        )
    return (
        f"switch to a spatial closure that damps it — {', '.join(damping)} "
        f"(`scheme=` on the entry) — or break one reflective axis pair"
    )


def warn_if_gauge_freedom(
    sn_mesh: "SNMesh", correction: float | None, *, where: str,
) -> None:
    r"""Say that the trace was repaired, or that the closure was unclassifiable.

    ⚠ **MUST be called DIRECTLY from a public entry, one frame deep.**
    ``stacklevel=3`` counts: frame 1 here, frame 2 the entry, frame 3 the user.
    Called from a private arm it blames ``orpheus/sn/solver.py`` — verbatim the
    defect #340 N4.7 measured at 2 of 8 emission sites and fixed by hoisting the
    call, and the reason a per-call ``stacklevel=`` argument is the wrong repair
    (a frame count asserted at the call site rots the moment a helper is
    interposed).

    **Three outcomes, and silence is one of them.**

    * **UNDETERMINED** — warn loudly and say the trace was NOT gauged. This is
      the user ruling: an unclassified closure is a third state, never merged
      into "no freedom".
    * **Repaired** — the gauge moved the trace by more than
      :data:`_GAUGE_AUDIBLE_FLOOR`. Say by how much, and name the root fix.
    * **Silent** — either there is no freedom, or there is and the returned
      trace was ALREADY the canonical member (``jacobi`` lands there;
      ``[M]`` ``~1e-15``). Nothing was done, so there is nothing to report — and
      the configuration's degeneracy is still legible in
      :attr:`~orpheus.sn.solution.IterationHistory.gauge_correction`, which
      carries the measured number either way.

    The warning reports an **action taken**, not a configuration property. That
    is what keeps it off the standard ``k_inf`` lattice, which is all-reflective
    by default and would otherwise warn on every solve.
    """
    verdict = gauge_freedom(sn_mesh)

    if verdict.undetermined:
        warnings.warn(
            f"{where}: the spatial closure could not be classified, so whether "
            f"the loss operator L+C-S-B is SINGULAR here is unknown and the "
            f"returned boundary trace was NOT gauge-fixed. {verdict.because}. "
            f"If it is singular, that trace is one arbitrary member of a "
            f"solution manifold: the bulk, k and every mirror-even functional "
            f"are still correct, but a mirror-odd one — a current tangential "
            f"to a reflective face — may be meaningless. To settle it, "
            f"{_damping_alternatives(int(sn_mesh.ndim))}. "
            f"Silence this per-call with warnings.catch_warnings(); make it "
            f"fatal everywhere with {GAUGE_ESCALATION_FLAG}.",
            GaugeFreedomWarning,
            stacklevel=3,
        )
        return

    if not verdict.present or correction is None:
        return
    if correction <= _GAUGE_AUDIBLE_FLOOR:
        return

    warnings.warn(
        f"{where}: the returned boundary trace was GAUGE-FIXED — "
        f"{correction:.2%} of it lay in ker(L+C-S-B), which is exactly "
        f"singular here, so the solve had converged to an arbitrary member of "
        f"a solution manifold rather than to a point. {verdict.because}. The "
        f"trace returned is now the canonical minimum-norm member (the one the "
        f"exact solution sits at); the bulk, k and every reaction rate are "
        f"unchanged, and no convergence certificate moved. "
        f"⚠ Nothing you could have checked would have shown this: every summed "
        f"functional of the trace is blind to the kernel by symmetry, so the "
        f"error surfaces only in a current TANGENTIAL to a reflective face. "
        f"To remove the freedom at the root instead of projecting it out, "
        f"{_damping_alternatives(int(sn_mesh.ndim))}. "
        f"Silence this per-call with warnings.catch_warnings(); make it fatal "
        f"everywhere with {GAUGE_ESCALATION_FLAG}.",
        GaugeFreedomWarning,
        stacklevel=3,
    )


# ─────────────────────────────────────────────────────────────────────
# The counting law — evaluated combinatorially, building nothing
# ─────────────────────────────────────────────────────────────────────
def _anova_dimension(cells: tuple[int, ...]) -> int:
    r"""The separable-equation dimension :math:`\kappa(U)` of §3.

    The solution space of "a sum of functions, the :math:`a`-th blind to
    coordinate :math:`i_a`, vanishing on the whole grid":

    .. math::

        \kappa(U) = \sum_{a \in U} \prod_{b \in U \setminus a} n_b
                    - \prod_{b \in U} n_b
                    + \prod_{b \in U} (n_b - 1)

    The middle two terms are :math:`-\dim(\text{image})`, from the ANOVA
    decomposition: such a sum reaches exactly the functions whose top
    interaction vanishes.

    >>> _anova_dimension((5, 7))            # two-term: both sides constant
    1
    >>> _anova_dimension((3, 4, 5))         # three-term: n_a + n_b + n_c - 1
    11
    """
    total = sum(
        int(np.prod([n for j, n in enumerate(cells) if j != i]))
        for i in range(len(cells))
    )
    return int(
        total
        - int(np.prod(cells))
        + int(np.prod([n - 1 for n in cells]))
    )


def predicted_kernel_dimension(sn_mesh: "SNMesh") -> int:
    r"""``dim ker A`` from the counting law — **without building any vector**.

    The structurally independent check on :class:`LossKernelGauge`: this walks
    the combinatorics of §3 and returns an integer, while the gauge walks the
    generators and takes an SVD.  Agreement between them is evidence that
    neither has a bookkeeping error; it is what
    :func:`~tests.sn.operators.test_loss_kernel_gauge` asserts in place of the
    dense SVD at sizes where the dense SVD is not viable.

    Counts the **R** component only, matching :class:`LossKernelGauge`'s scope —
    the tangential component ``T`` contributes one dimension per tangential
    trace DOF and is not gauged.
    """
    if not gauge_freedom(sn_mesh).present:
        return 0
    spatial = tuple(int(n) for n in sn_mesh.spatial_shape)
    total = 0
    for _orbit, active in _reflection_orbits(sn_mesh):
        if len(active) < 2:
            continue
        per_orbit = 0
        for size in range(2, len(active) + 1):
            for subset in combinations(active, size):
                spectators = int(np.prod(
                    [spatial[c] for c in range(sn_mesh.ndim)
                     if c not in subset] or [1]
                ))
                per_orbit += _anova_dimension(
                    tuple(spatial[c] for c in subset)
                ) * spectators
        total += per_orbit
    return total * int(sn_mesh.ng)


# ─────────────────────────────────────────────────────────────────────
# The ordinate orbits under the reflection group
# ─────────────────────────────────────────────────────────────────────
def _direction_cosines(sn_mesh: "SNMesh") -> NDArray:
    """``(N, 3)`` direction cosines, one row per ordinate."""
    quad = sn_mesh.quad
    return np.stack(
        [np.asarray(quad.mu_x, dtype=float),
         np.asarray(quad.mu_y, dtype=float),
         np.asarray(quad.mu_z, dtype=float)],
        axis=1,
    )


def _reflection_orbits(
    sn_mesh: "SNMesh",
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    r"""Orbits of ordinates under :math:`\langle R_a : a \text{ reflective}\rangle`.

    Returns ``(orbit_ordinates, active_axes)`` pairs, where ``active_axes`` is
    the set :math:`S` of §2 — reflective AND non-tangential for that orbit.

    Raises
    ------
    ValueError
        If the quadrature is not closed under one of the mirrors.  A reflective
        boundary on a rule that lacks the matching mirror partner is not
        realisable at all — the reflected ordinate has nowhere to land — so this
        is a genuine admission refusal, not a limitation of the construction.
    """
    mu = _direction_cosines(sn_mesh)
    reflective = sn_mesh.reflective_axes
    node_of = {tuple(np.round(row, 10)): n for n, row in enumerate(mu)}

    partners: dict[int, NDArray] = {}
    for axis in reflective:
        mirrored = mu.copy()
        mirrored[:, axis] *= -1.0
        found = np.array(
            [node_of.get(tuple(np.round(row, 10)), -1) for row in mirrored],
            dtype=np.intp,
        )
        if np.any(found < 0):
            missing = int(np.flatnonzero(found < 0)[0])
            raise ValueError(
                f"the quadrature is not closed under the axis-{axis} mirror "
                f"(ordinate {missing}, direction {mu[missing]}), so a "
                f"reflective boundary on that axis is not realisable: the "
                f"reflected ordinate is not in the rule."
            )
        partners[axis] = found

    seen = np.zeros(mu.shape[0], dtype=bool)
    orbits: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for start in range(mu.shape[0]):
        if seen[start]:
            continue
        stack, members = [start], []
        seen[start] = True
        while stack:
            node = stack.pop()
            members.append(node)
            for axis in reflective:
                partner = int(partners[axis][node])
                if not seen[partner]:
                    seen[partner] = True
                    stack.append(partner)
        active = tuple(
            axis for axis in reflective
            if abs(mu[members[0], axis]) > _TANGENTIAL_MU
        )
        orbits.append((tuple(sorted(members)), active))
    return tuple(orbits)


# ─────────────────────────────────────────────────────────────────────
# The basis of one block
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, eq=False)
class LossKernelBasis(Basis):
    r"""One :math:`(\text{ordinate orbit}, \text{group})` summand of ``ker A``.

    A precomputed, :math:`G`-**orthonormal** table over that block's trace
    degrees of freedom — see the module docstring §5 for why orthonormal rather
    than the raw pair generators, and §4 for why one block rather than the whole
    kernel.

    ``eq=False`` (identity equality) follows :class:`IndicatorBasis`: the fields
    are NumPy arrays, which have no value equality a dataclass can use.

    Parameters
    ----------
    table
        ``(n_block_dofs, n_modes)``, :math:`G`-orthonormal by construction.
    orbit, group
        The block's identity — carried so the coefficient space can be named
        distinguishably (two blocks' coefficient spaces must not compare equal,
        or an operator-composition guard would accept a mismatched pair).
    """

    table: NDArray
    orbit: tuple[int, ...]
    group: int

    # ── the projection-validity declaration ───────────────────────────
    @property
    def gram_structure(self) -> GramStructure:
        r"""``DIAGONAL`` — and here that is a THEOREM, not a measurement.

        The table is the left factor of a :math:`\sqrt{G}`-weighted SVD divided
        by :math:`\sqrt G`, so :math:`\Phi^{\mathsf T} G \Phi = U^{\mathsf T} U
        = I` exactly.  The frame's row-sum probe is therefore not merely *valid*
        (which is all ``DIAGONAL`` claims) but returns all ones, and
        :math:`G^{-1}` is the identity.

        ⚠ This would be **false** for the raw pair generators at
        :math:`d \ge 3` — ``[M]`` 43 % off-diagonal at ``(3,4,5)``.  The
        orthonormalisation is what earns the declaration; do not move it.
        """
        return GramStructure.DIAGONAL

    # ── tabulation ────────────────────────────────────────────────────
    def evaluate(self, points: NDArray, /) -> NDArray:
        """Return the precomputed table, validating the caller's node count.

        The modes are determined by the mesh, the quadrature and the boundary
        laws when the block is built — there is nothing to evaluate AT a point,
        so ``points`` is accepted for the :class:`Basis` contract and checked
        against the table's row count (the :class:`OverlapBasis` idiom).  That
        check is what stops a mesh/basis mismatch from becoming a silently
        mis-shaped einsum.
        """
        rows = int(np.asarray(points).shape[0])
        if rows != self.table.shape[0]:
            raise ValueError(
                f"LossKernelBasis was built for {self.table.shape[0]} block "
                f"DOFs but was evaluated at {rows} nodes — the measure and the "
                f"basis describe different blocks."
            )
        return self.table

    # ── table contractions ────────────────────────────────────────────
    # The canonical dual factor is the IDENTITY: for an orthonormal basis the
    # dual frame IS the frame, so `reconstruct` and `synthesize` coincide. (For
    # a non-orthonormal basis the dual factor would be measure-dependent and
    # therefore could not live on a Basis at all — the IndicatorBasis argument.)
    def synthesize(self, coefficients: NDArray, table: NDArray, /) -> NDArray:
        return np.einsum("nk,k...->n...", table, coefficients)

    def analyze(
        self, values: NDArray, table: NDArray, weights: NDArray, /,
    ) -> NDArray:
        return np.einsum("n,nk,n...->k...", weights, table, values)

    def analyze_transpose(
        self, coefficients: NDArray, table: NDArray, weights: NDArray, /,
    ) -> NDArray:
        return np.einsum("n,nk,k...->n...", weights, table, coefficients)

    def reconstruct(self, coefficients: NDArray, table: NDArray, /) -> NDArray:
        return np.einsum("nk,k...->n...", table, coefficients)

    def reconstruct_transpose(self, values: NDArray, table: NDArray, /) -> NDArray:
        return np.einsum("nk,n...->k...", table, values)

    def mass_matrix(self, measure: DiscreteMeasure, /) -> NDArray:
        r"""The discrete Gram :math:`\Phi^{\mathsf T} G \Phi` — should be ``I``.

        Not on any hot path; it exists so the orthonormality claim behind
        :attr:`gram_structure` is directly assertable rather than trusted.
        """
        table = self.evaluate(measure.nodes)
        return np.einsum("n,nj,nk->jk", measure.weights, table, table)

    @property
    def domain(self) -> Manifold:
        r"""The block's trace degrees of freedom, as an index set.

        There is nothing to evaluate AT a point here (:meth:`evaluate` returns
        a precomputed table), so the "points" are the block's own trace DOF
        indices — a finite set with no metric structure, which is exactly
        :class:`~orpheus.numerics.manifold.IndexSet`.

        ⭐ Built from the same ``f"sn_trace_orbit{orbit}_g{group}"`` label the
        block's :class:`~orpheus.numerics.measure.DiscreteMeasure` already
        carries as its ``support``, five lines from where that measure is
        constructed — so the basis and the measure of one frame name ONE
        manifold rather than two.  This class was already the tree's positive
        control on that discipline: it is the only basis that never fabricated
        a space name, because its author named the space by the block's own
        identity.
        """
        return IndexSet(label=f"sn_trace_orbit{self.orbit}_g{self.group}")

    @property
    def space(self) -> FunctionSpace:
        """The coefficient space, named by the block it belongs to."""
        return FunctionSpace(
            name=f"loss_kernel_coeff[orbit{self.orbit}_g{self.group}]",
            shape=(self.table.shape[1],),
        )


# ─────────────────────────────────────────────────────────────────────
# Building one block
# ─────────────────────────────────────────────────────────────────────
def _pair_generators(
    spatial: tuple[int, ...], active: tuple[int, ...], ndim: int,
) -> list[dict[tuple[int, frozenset], NDArray]]:
    r"""The §3 pair generators of :math:`\sum_{a \in S} s_a Y_a = 0`.

    One generator per (axis pair :math:`\{a,b\}`, character :math:`U` with
    :math:`\{a,b\} \subseteq U \subseteq S`, index tuple :math:`j` on the
    remaining axes).  Each is a dict mapping ``(axis, character)`` to the array
    that axis's :math:`Y` carries, over the axes ``!= axis``.

    Check that they solve the equation: :math:`s_a Y_a + s_b Y_b = \chi_U\delta -
    \chi_U\delta = 0`, and :math:`Y_a` is blind to :math:`s_a` (because
    :math:`a \notin U\setminus\{a\}`) and to :math:`i_a` (because :math:`a` is
    excluded from :math:`j`).  ∎

    The set is mildly over-complete — at :math:`d = 3` all-reflective there are
    :math:`2\sum n` generators spanning :math:`2\sum n - 1` dimensions, the one
    relation being :math:`M_{yz} - M_{xz} + M_{xy} = 0` among the full-character
    pair modes.  The rank cut in :func:`_build_block_table` removes it.
    """
    def indicator(kept: list[int], pinned: dict[int, int]) -> NDArray:
        out = np.ones(tuple(spatial[c] for c in kept), dtype=float)
        for position, axis in enumerate(kept):
            if axis not in pinned:
                continue
            column = np.zeros(spatial[axis])
            column[pinned[axis]] = 1.0
            out = out * column.reshape(
                [-1 if k == position else 1 for k in range(len(kept))]
            )
        return out

    generators: list[dict[tuple[int, frozenset], NDArray]] = []
    for a, b in combinations(sorted(active), 2):
        free_axes = [c for c in range(ndim) if c not in (a, b)]
        others_in_active = [c for c in sorted(active) if c not in (a, b)]
        for extra_count in range(len(others_in_active) + 1):
            for extra in combinations(others_in_active, extra_count):
                character = frozenset((a, b)) | frozenset(extra)
                for pinned_values in iter_product(
                    *[range(spatial[c]) for c in free_axes]
                ):
                    pinned = dict(zip(free_axes, pinned_values))
                    generators.append({
                        (a, character - {a}): indicator(
                            [c for c in range(ndim) if c != a], pinned),
                        (b, character - {b}): -indicator(
                            [c for c in range(ndim) if c != b], pinned),
                    })
    return generators


@dataclass(frozen=True, slots=True)
class _FacePlacement:
    """Where one ``(face, ordinate)``'s values land in a block's flat vector."""

    axis: int
    ordinate: int
    #: ``(-1)^k`` at this face: ``+1`` at the min face (``k = 0``), ``(-1)^{n_a}``
    #: at the max face (``k = n_a``) — the sawtooth evaluated at the two ends.
    sawtooth_sign: float
    #: Row of the block's index array for each flattened transverse position.
    rows: NDArray
    transverse_shape: tuple[int, ...]


def _block_support(
    sn_mesh: "SNMesh", orbit: tuple[int, ...],
    group: int, active: tuple[int, ...],
) -> tuple[NDArray, tuple[_FacePlacement, ...]]:
    r"""The block's trace DOFs, sorted, plus where each face writes into them.

    Indices are **trace-local** — offsets into the flat
    :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`
    buffer, which is the carrier
    :class:`~orpheus.numerics.operator.TraceRestrictionOperator` gathers from.
    (⚠ Not full-field offsets: ``FullFieldSpace``'s carrier is the *typed
    composite*, so a ``np.take`` operator declaring it as a domain would be a
    lie that the ``(name, shape)`` composability guard cannot catch.)

    Support is confined to faces whose axis lies in :math:`S`: a vacuum axis
    carries :math:`\varphi \equiv 0` and a tangential one carries a zero
    coefficient.  ⭐ That is also why the block never touches a ``G == 0`` row —
    every retained face has :math:`|\mu_a| > 0` for every ordinate of the orbit,
    so :math:`G = |\mu_a| w_n > 0` there **by construction**, and the
    :math:`\sqrt G` division of §5 is well-posed without a mask.  (The claim is
    re-asserted at build time rather than trusted; see
    :func:`_build_block_table`.)
    """
    layout = sn_mesh.angular_trace.layout
    spatial = tuple(int(n) for n in sn_mesh.spatial_shape)
    ndim = int(sn_mesh.ndim)

    raw: list[NDArray] = []
    stubs: list[tuple[int, int, float, NDArray, tuple[int, ...]]] = []
    for face_name, slot in layout.faces.items():
        axis, sign = face_normal(face_name)
        if axis not in active:
            continue
        transverse = tuple(spatial[c] for c in range(ndim) if c != axis)
        if tuple(slot.shape[2:]) != transverse:
            raise ValueError(
                f"face {face_name!r} has slot shape {slot.shape}, whose "
                f"trailing axes {tuple(slot.shape[2:])} are not the transverse "
                f"cell counts {transverse}. A multi-moment closure carries an "
                f"extra trailing axis — but such a closure DAMPS the face mode, "
                f"so this block should never have been built (see "
                f"`gauge_freedom`)."
            )
        per_ordinate = int(np.prod(slot.shape[1:]))
        per_group = int(np.prod(slot.shape[2:]))
        for ordinate in orbit:
            start = slot.offset + ordinate * per_ordinate + group * per_group
            absolute = start + np.arange(per_group, dtype=np.intp)
            raw.append(absolute)
            # k = 0 at the min face, k = n_axis at the max face.
            k = 0 if sign < 0 else spatial[axis]
            stubs.append((axis, ordinate, 1.0 if k % 2 == 0 else -1.0,
                          absolute, transverse))

    if not raw:
        return np.empty(0, dtype=np.intp), ()

    indices = np.sort(np.concatenate(raw))
    placements = tuple(
        _FacePlacement(
            axis=axis, ordinate=ordinate, sawtooth_sign=sawtooth_sign,
            rows=np.searchsorted(indices, absolute).astype(np.intp),
            transverse_shape=transverse,
        )
        for axis, ordinate, sawtooth_sign, absolute, transverse in stubs
    )
    return indices, placements


def _transverse_factors(
    sn_mesh: "SNMesh", axis: int,
) -> tuple[NDArray, NDArray]:
    r"""``((-1)^{sum_{c != a} i_c}, A_a(i_perp))`` on the transverse grid.

    The checkerboard and the face area of §2.  The area is
    :math:`\prod_{b \neq a} h_b(i_b)` — read from the mesh's own edges, which is
    what makes a **graded** mesh need no separate treatment.
    """
    ndim = int(sn_mesh.ndim)
    kept = [c for c in range(ndim) if c != axis]
    widths = [np.diff(np.asarray(sn_mesh.axes[c].edges, dtype=float))
              for c in kept]
    checker = np.ones(tuple(w.size for w in widths))
    area = np.ones(tuple(w.size for w in widths))
    for position, width in enumerate(widths):
        shape = [-1 if k == position else 1 for k in range(len(widths))]
        area = area * width.reshape(shape)
        checker = checker * ((-1.0) ** np.arange(width.size)).reshape(shape)
    return checker, area


def _build_block_table(
    sn_mesh: "SNMesh", orbit: tuple[int, ...],
    group: int, active: tuple[int, ...], trace_metric: NDArray,
) -> tuple[NDArray, NDArray]:
    r"""``(sorted DOF indices, G-orthonormal table)`` for one block.

    Three steps, of which the third is the one §5 exists to justify:

    1. the §3 pair generators, expanded over the orbit's ordinates;
    2. each generator mapped to the block's DOFs through
       :math:`\varphi = Y \cdot (-1)^{\sum i} / (|\mu_a| A_a)`;
    3. **one** :math:`\sqrt G`-weighted SVD, which reduces the over-complete
       generating set to its rank AND :math:`G`-orthonormalises the survivors —
       replacing both of the derivation memo's factorisations.
    """
    indices, placements = _block_support(sn_mesh, orbit, group, active)
    if indices.size == 0:
        return indices, np.zeros((0, 0))

    mu = _direction_cosines(sn_mesh)
    spatial = tuple(int(n) for n in sn_mesh.spatial_shape)
    factors = {axis: _transverse_factors(sn_mesh, axis) for axis in active}
    generators = _pair_generators(spatial, active, int(sn_mesh.ndim))

    columns = np.zeros((indices.size, len(generators)))
    for column, generator in enumerate(generators):
        # Expand the character-space generator over the orbit's ordinates.
        carried: dict[tuple[int, int], NDArray] = {}
        for ordinate in orbit:
            signs = {c: (1.0 if mu[ordinate, c] > 0.0 else -1.0)
                     for c in active}
            for (axis, character), array in generator.items():
                chi = 1.0
                for c in character:
                    chi *= signs[c]
                key = (ordinate, axis)
                carried[key] = carried.get(key, 0.0) + chi * array
        for place in placements:
            array = carried.get((place.ordinate, place.axis))
            if array is None:
                continue
            checker, area = factors[place.axis]
            phi = (place.sawtooth_sign * array * checker
                   / (abs(mu[place.ordinate, place.axis]) * area))
            columns[place.rows, column] = phi.ravel()

    metric = trace_metric[indices]
    if not np.all(metric > 0.0):
        raise ValueError(  # unreachable by construction — see _block_support
            f"block (orbit={orbit}, group={group}) touches "
            f"{int(np.sum(metric <= 0.0))} trace DOF(s) with a vanishing "
            f"metric. Component R is defined on current-carrying ordinates "
            f"only; a tangential row here means the active-axis set S was "
            f"mis-derived, and sqrt(G)-orthonormalisation would divide by zero."
        )

    root = np.sqrt(metric)
    left, singular, _ = np.linalg.svd(root[:, None] * columns,
                                      full_matrices=False)
    rank = int(np.sum(singular > _RANK_RTOL * singular[0])) if singular.size else 0
    table = left[:, :rank] / root[:, None]
    return indices, table


# ─────────────────────────────────────────────────────────────────────
# The gauge — a direct sum of block projectors
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class _GaugeBlock:
    r"""One summand :math:`\gamma_b^{\mathsf T} \Pi_b \gamma_b` of the gauge."""

    gather: TraceRestrictionOperator
    projector: LinearOperator
    basis: LossKernelBasis


class LossKernelGauge(LinearOperator):
    r"""The :math:`G`-orthogonal projector onto :math:`\ker A`.

    .. math::

        \Pi \;=\; \sum_b \gamma_b^{\mathsf T} \, \Pi_b \, \gamma_b ,
        \qquad
        \Pi_b \;=\; R_b \circ G_b^{-1} \circ M_b

    An endomorphism of the boundary-trace space — **not** of the full field,
    because the kernel is a trace object (module docstring §1) and the trace
    space's carrier is the flat boundary array a
    :class:`~orpheus.numerics.operator.TraceRestrictionOperator` can gather
    from.  Gauging a state is therefore ``boundary - gauge.apply(boundary)``,
    with the bulk untouched by construction rather than by a zero block.

    Each :math:`\Pi_b` is built the frame way —
    ``frame.conjugate(frame.gram_inverse)`` — so there is ONE
    spelling of "project onto a span" in the package and this is a consumer of
    it, not a second implementation.  Because each block's basis is
    :math:`G`-orthonormal (§5), :math:`G_b^{-1}` is the identity and the
    conjugation collapses to :math:`R_b M_b`; the machinery is kept rather than
    short-circuited so the orthonormality claim stays *checkable* by the
    frame's own Gram rather than assumed by a hand-rolled contraction.

    The summands act on **disjoint** index sets, so the sum IS the direct sum:
    no two blocks write the same row, and the result is a projector because
    each :math:`\Pi_b` is and their ranges are mutually :math:`G`-orthogonal.

    Not invertible — a projector onto a proper subspace never is — and that is
    spelled by :attr:`is_invertible` plus the **absence** of an ``inverse``
    method, the house convention (``TraceRestrictionOperator``).
    """

    def __init__(
        self, blocks: tuple[_GaugeBlock, ...], space: FunctionSpace,
    ) -> None:
        self._blocks = tuple(blocks)
        self._space = space
        seen = np.concatenate(
            [b.gather.indices for b in self._blocks]
        ) if self._blocks else np.empty(0, dtype=np.intp)
        if np.unique(seen).size != seen.size:
            raise ValueError(
                f"LossKernelGauge blocks must have DISJOINT supports — the "
                f"sum of block projectors is a projector only if no two "
                f"blocks claim the same trace DOF. Got {seen.size} DOFs "
                f"across {len(self._blocks)} blocks with only "
                f"{np.unique(seen).size} distinct."
            )

    # ── the arrow ─────────────────────────────────────────────────────
    @property
    def domain(self) -> Optional[FunctionSpace]:
        return self._space

    @property
    def codomain(self) -> Optional[FunctionSpace]:
        return self._space

    @property
    def blocks(self) -> tuple[_GaugeBlock, ...]:
        """The summands, one per ``(ordinate orbit, group)``."""
        return self._blocks

    @property
    def dimension(self) -> int:
        r"""``dim ker A`` restricted to component **R** — the rank of ``Pi``."""
        return sum(int(b.basis.table.shape[1]) for b in self._blocks)

    # ── the action ────────────────────────────────────────────────────
    def apply(self, x: NDArray) -> NDArray:
        r"""The component of ``x`` inside :math:`\ker A`."""
        values = np.asarray(x, dtype=float)
        out = np.zeros_like(values)
        for block in self._blocks:
            out[block.gather.indices] = block.projector.apply(
                block.gather.apply(values)
            )
        return out

    def apply_transpose(self, x: NDArray) -> NDArray:
        r""":math:`\Pi^{\mathsf T} = \Pi` — each :math:`\Pi_b` is self-adjoint.

        A :math:`G`-orthogonal projector onto a :math:`G`-orthonormal span is
        symmetric in the :math:`G` inner product, and the gather/scatter pair
        is a transpose pair, so the sum inherits it.
        """
        return self.apply(x)

    @property
    def is_adjointable(self) -> bool:
        return True

    @property
    def is_invertible(self) -> bool:
        """``False`` — a projector onto a proper subspace kills its complement."""
        return False

    def gauge(self, trace: NDArray) -> NDArray:
        r"""``(I - Pi) trace`` — the canonical, minimum-:math:`G`-norm member.

        The operation the solver actually performs.  Residual-neutral by
        construction: :math:`A(\psi - \Pi\psi) = A\psi` because
        :math:`\Pi\psi \in \ker A`, so **no convergence certificate can move**.
        """
        values = np.asarray(trace, dtype=float)
        return values - self.apply(values)

    def __repr__(self) -> str:
        return (
            f"LossKernelGauge(blocks={len(self._blocks)}, "
            f"dimension={self.dimension}, space={self._space!r})"
        )

    # ── construction ──────────────────────────────────────────────────
    @classmethod
    def for_mesh(cls, sn_mesh: "SNMesh") -> "LossKernelGauge":
        """Build the gauge for a mesh — **zero blocks when there is nothing to fix**.

        A zero-block gauge is the honest answer to a non-singular configuration,
        not a failure: the projection onto a trivial kernel is the zero map, so
        :meth:`gauge` is the identity and callers need no ``None`` branch.
        Whether there is anything to fix is decided by :func:`gauge_freedom`,
        whose two conjuncts are both derived.

        ⚠ An UNDETERMINED closure yields zero blocks too — *not gauging* is the
        ruled behaviour — but the caller owes the user a loud warning naming the
        obstruction, which :attr:`GaugeFreedom.because` supplies.
        """
        trace_space = sn_mesh.angular_trace
        if not gauge_freedom(sn_mesh).present:
            return cls((), trace_space)

        metric = np.asarray(trace_space.inner_product_weights, dtype=float)
        n_trace = int(np.prod(trace_space.shape))
        blocks: list[_GaugeBlock] = []
        for orbit, active in _reflection_orbits(sn_mesh):
            if len(active) < 2:
                continue
            for group in range(int(sn_mesh.ng)):
                indices, table = _build_block_table(
                    sn_mesh, orbit, group, active, metric)
                if table.size == 0 or table.shape[1] == 0:
                    continue
                basis = LossKernelBasis(
                    table=table, orbit=orbit, group=group)
                measure = DiscreteMeasure(
                    nodes=indices.astype(float),
                    weights=metric[indices],
                    # ⭐ Read from the basis, not re-spelled beside it. Tracker
                    # 2.1 left this pair as the ONE production frame whose two
                    # halves disagreed in spelling — the measure tagged the bare
                    # ``sn_trace_orbit…`` label while the basis wrapped it in
                    # ``index(…)`` — and pinned the divergence in ``test_d6``.
                    # Taking the manifold from its owner closes it by
                    # construction rather than by keeping two strings equal.
                    support=basis.domain,
                )
                frame = GalerkinFrame(basis, measure)
                blocks.append(_GaugeBlock(
                    gather=TraceRestrictionOperator(
                        indices, n_total=n_trace,
                        domain=trace_space, codomain=measure.space),
                    projector=frame.conjugate(frame.gram_inverse),
                    basis=basis,
                ))
        return cls(tuple(blocks), trace_space)
