.. _operator-inverse-family:

===========================
The Operator Inverse Family
===========================

.. contents:: Contents
   :local:
   :depth: 2


.. Machine header — the ``nexus-meta`` schema for this page (PROVISIONAL).
.. This page was extracted verbatim from ``operator_algebra.rst``; the schema
.. is provisional pending a full re-audit of the split corpus (#231).

.. dropdown:: Machine header — ``nexus-meta`` schema (PROVISIONAL)
   :color: muted

   .. code-block:: yaml

      module: transport
      concept: operator_inverse_family
      role: "how the operator algebra's inverse A⁻¹ is realized and materialized (#226 taxonomy)"
      covers: [driver-applied sweep inverse, Green preconditioned-splitting inverse, dense materialising inverse, sparse assembly axis]
      depends_on: [operator_algebra]
      status: "extracted from operator_algebra.rst; content verbatim, provisional header"


This page is the **inverse family** of the operator algebra: the four
realizations of "apply :math:`A^{-1}` / materialize :math:`A`" that the #226
operator taxonomy separates. The operators being inverted — the loss
composite :math:`A = L + C - S - N_{2n} - B`, and its invertible
sub-composite
:math:`L + C` whose inverse **is** the transport :term:`sweep` — are developed in
:doc:`/theory/foundations/operator_algebra`. This page picks up at the
inverse itself and follows it through four distinct realizations:

- the **solver builds / driver applies** split — the posing layer builds
  :math:`A^{-1}` once and the iteration primitive *applies* it
  (:ref:`inverse-application-driver`);
- the **Green operator**, the preconditioned-splitting sum inverse and the
  first *iterative* member of the family (:ref:`green-operator`);
- the **dense materialising inverse** — ``as_matrix`` out of the operator
  category, factored once into a
  :class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
  (:ref:`matrix-inverse-operator`);
- the **assembly axis** — structural sparse emission, the third per-axis
  three-layer surface (:ref:`operator-algebra-assembly-axis`).


.. _inverse-application-driver:

The solver builds the inverse; the driver applies it
====================================================

Steps 1–2 reified the inverse *operators* — the schedule-triangular
:class:`~orpheus.sn.operators.sweep_operator.SweepOperator` (step 1, the
WDD sweep as :math:`(L+C)^{-1}`), the reified splitting matrix
:math:`M=(L+C)-B_{\rm lower}` (step 2, :ref:`si-gauss-seidel-reification`
in :doc:`/theory/methods/sn/cartesian_multid`), and the windowed composition
:math:`P\circ A^{-1}` (step 2, :ref:`windowing-retyped` in
:doc:`/theory/methods/sn/cartesian_multid`). **Step 3
retypes the driver–inverse boundary.** The solver (posing) layer builds
the inverse *once*, and the iteration primitive
(:class:`~orpheus.numerics.iteration.SourceIteration`) *applies* it:

.. (vv-status rationale) The SI driver update
   ψ_{n+1} = A_inv.apply(q_ext + Σ_i g_i ψ_n, initial_guess=ψ_n), with
   A_inv = A⁻¹ built by the solver (#226 step 3). Governing iteration
   (structural); the SI fixed point is exercised by the source-iteration
   solver suites.
.. vv-status: inverse-driver-si-update documented

.. math::
   :label: inverse-driver-si-update

   \psi_{n+1} \;=\; A_{\rm inv}.\mathrm{apply}
   \Bigl(q_{\rm ext} + \textstyle\sum_i g_i\,\psi_n,\;
   \ \mathrm{initial\_guess}=\psi_n\Bigr),
   \qquad
   A_{\rm inv} \;=\; A^{-1}\ \text{(built by the solver)} .

Concretely :func:`_within_group_si <orpheus.sn.solver._within_group_si>`
does ``step, windowed = _maybe_window(base_resolvent.inverse(), S,
sn_mesh)`` and hands ``step`` to the ``SourceIteration`` constructor as
its **first argument** ``A_inv``. This closes the #226 steps-1–3 arc: the
duck-typed "resolvent" — the object whose ``apply`` and ``solve`` inverted
*different* operators — is **fully dissolved**. Nothing in the driver is a
resolvent any more; there is a forward operator (built by the solver,
invertible by contract) and its inverse (built by the solver, *applied* by
the driver).

This **refines the variadic-driver picture** of
:ref:`bc-extraction-variadic-driver`: the driver's first argument, there
described as "the resolvent it must invert", is — since step 3 — for
:class:`~orpheus.numerics.iteration.SourceIteration` the pre-built inverse
operator it *applies*, while
:class:`~orpheus.numerics.iteration.KrylovAcceleration` keeps the
*forward* :math:`A` for its GMRES matvec (and preconditions with
:math:`A^{-1}`). The homogeneous ``*gains`` bag is unchanged in both.


The seeded-apply family — who threads the start, who ignores it
---------------------------------------------------------------

The inverse family exposes ONE canonical apply signature — the static
contract :class:`~orpheus.numerics.iteration.SupportsSeededApply`:

.. (vv-status rationale) The canonical SupportsSeededApply signature
   apply(rhs, *, initial_guess=None) → codomain. A notation / type-contract
   definition (a static annotation target, deliberately not
   runtime_checkable).
.. vv-status: seeded-apply-signature documented

.. math::
   :label: seeded-apply-signature

   \mathrm{apply}(\text{rhs},\; *,\; \text{initial\_guess}=\text{None})
   \ \longrightarrow\ \text{codomain} .

The driver threads the previous iterate as ``initial_guess`` on **every**
step, uniformly — the plumbing the curvilinear sweep needs, where the
previous iterate *is* the Morel–Montry Carlson coupled-pole seed (the
sweep reads the level's :math:`\psi(\mu=-1)` from it — the curvilinear
seed obstruction, :ref:`sn-angular-windowing-geometry-restriction`). A
dropped / zeroed / stale seed is a **wrong-fixed-point** bug there, not a
rate change. Members with no use for a start accept the keyword and ignore
it, each documented per type:

.. list-table:: The seeded-apply contract across the inverse family
   :header-rows: 1
   :widths: 34 22 44

   * - Inverse operator
     - ``initial_guess``
     - Why
   * - :class:`~orpheus.sn.operators.sweep_operator.SweepOperator`
       (curvilinear 1-D)
     - **threaded**
     - The M–M Carlson closure seeds the :math:`\mu=-1`
       starting-direction flux from the previous iterate's :term:`per-ordinate <ordinate>`
       :math:`\psi`; ``apply`` delegates to ``inner.solve(rhs,
       initial_guess=...)`` which reads it.
   * - :class:`~orpheus.sn.operators.sweep_operator.SweepOperator`
       (2-D Cartesian DD)
     - accepted, ignored
     - The :term:`diamond-difference <diamond difference>` wavefront is a direct forward substitution
       down the upwind DAG — no interior-iterate seed. The kwarg threads
       through harmlessly.
   * - :class:`~orpheus.numerics.operator.InverseOperator`
       (value-bearing leaf)
     - accepted, ignored
     - An **exact** pointwise inverse (a division) has no iterative start
       to seed — ``del initial_guess``.
   * - :class:`~orpheus.sn.operators.windowing.WindowedSweep`
       (``P @ A.inverse()``)
     - accepted, ignored
     - The multi-D Cartesian walk has no bulk-seed consumer; a moment
       iterate could not seed the *angular* walk anyway (wrong
       representation). The reflective lag rides the driver's ``B`` /
       ``B_upper`` gain, never this kwarg.

The contract is **static only** (an annotation target), deliberately
**not** ``runtime_checkable`` — mirroring
:class:`~orpheus.numerics.operator.SupportsInverse`. The runtime truth is
the :attr:`~orpheus.numerics.operator.LinearOperator.is_invertible`
property; the Protocol pins the *shape* pyright checks. Threading the seed
**unconditionally** (no per-call decision) is what let the driver be
shape-agnostic — pinned at the time by the always-on Mode-11 spy
``test_seed_threading_spy.py`` (call *k*'s ``initial_guess`` must equal
call *(k−1)*'s return, by value; **retired in #280 2.5c** together with
the vestigial ``initial_guess`` threading it guarded). The live pins on
the seed contract are the strict no-forwarding spy in
``tests/sn/operators/test_inverse_operator_equivalence.py`` and the
curvilinear value catcher in ``tests/sn/eigenvalue/test_keff_curvilinear.py``.


Retiring the signature probe — why it existed, why it could go
--------------------------------------------------------------

Before step 3 the driver could not assume a uniform seed signature,
because it consumed **duck-typed resolvents** whose ``solve`` methods had
heterogeneous signatures (some took ``initial_guess``, some did not). It
therefore probed each one at runtime:

.. code-block:: python

   # RETIRED at #226 step 3
   if _solve_accepts_seed(resolvent):        # inspect.signature(L.solve)
       psi = resolvent.solve(rhs, initial_guess=psi_prev)   # _solve_with_seed
   else:
       psi = resolvent.solve(rhs)

The ``inspect.signature`` probe (``_solve_accepts_seed`` /
``_solve_with_seed``) was a **stringly-typed dispatch on shape** — exactly
the anti-pattern the reification exists to remove. It could retire **only
once the family signature was canonical**: with every inverse advertising
the *same* ``apply(rhs, *, initial_guess=None)``
(:class:`~orpheus.numerics.iteration.SupportsSeededApply`), the driver
threads the keyword unconditionally and the probe has nothing to decide.
Its post-rewire falsifier is the **M-PROBE** mutation (sever the
``initial_guess`` thread at
:meth:`SweepOperator.apply <orpheus.sn.operators.sweep_operator.SweepOperator.apply>`),
which reddens the seed-threading spy in under a second.

Two further gates dissolved with it:

* **The constructor invertibility gate is gone.** The driver's whole
  contract is now a callable ``apply`` — the step operator arrives
  *pre-inverted*, so it never needs a ``solve``. The **invertibility
  obligation moved to the inverse builder**: on a non-invertible leaf
  the inverse simply cannot be constructed (a *structural* leaf declares
  no ``inverse()`` — a static error; a *value-dependent* one raises
  :class:`~orpheus.numerics.operator.NotInvertible`), so "is this a
  faithful inverse of the intended forward?" is discharged where the
  operator is *built*, not re-checked where it is *applied*.
* **An apply-only step operator is legitimate by design.** The windowed
  product ``P @ A.inverse()`` reports ``is_invertible = False`` (its
  :math:`P` factor is a coisometry, :ref:`windowing-retyped`), yet it is
  a valid step operator because a step operator only needs to *apply*.
  Gating on a callable ``apply`` alone — not on invertibility — is what
  makes the apply-only windowed product a first-class step operator
  rather than an illegal state needing an adapter.


The narrowing boundary and the structural resolution
----------------------------------------------------

``SupportsSeededApply`` and ``SupportsInverse`` are not
``runtime_checkable`` (an ``isinstance`` against a Protocol carrying only
a method member is a false-positive machine). The narrow from "an
invertible operator" to "a seeded-apply operator" therefore lives in
**one** place,
:func:`~orpheus.numerics.iteration.seeded_inverse` (spelled
``_seeded_inverse`` until #276 A4 promoted it — a third consumer, the
daggered adjoint :class:`~orpheus.numerics.iteration.SourceIteration`,
had joined the two below and cross-module callers were already importing
the underscore name):

.. code-block:: python

   def seeded_inverse(A):                        # numerics/iteration.py
       if not invertible(A):                     # the guard lives HERE
           raise NotInvertible(...)
       inv = A.inverse()
       if _wrap_delegate_member(inv):            # TypeGuard on the family
           return inv                            # canonical seeded apply
       return _SeededExactApply(inv)             # accept-and-drop wrapper

Two things about that body are load-bearing. First, **the runtime
precondition is internal, not the caller's**: ``seeded_inverse`` runs its
own :func:`~orpheus.numerics.operator.invertible` check and raises
:class:`~orpheus.numerics.operator.NotInvertible`, so a caller's
``is_invertible`` guard is a *domain-message* courtesy, never the thing
that makes the call sound. Second, **there is no** ``cast``: both
narrowings are checked bridges (Design C) — ``invertible`` is a
``TypeGuard`` whose runtime predicate *is* the static permission for the
``.inverse()`` call, and ``_wrap_delegate_member`` is a ``TypeGuard``
``isinstance`` on :class:`~orpheus.numerics.operator.InverseWrapMixin`
membership, i.e. type-as-structure dispatch on the two-kinds split rather
than a signature probe. The algebra-closed branch wraps in
``_SeededExactApply``, which accepts and drops the seed — an exact
inverse has nothing to seed.

:class:`~orpheus.numerics.iteration.KEigenvalue` (inner
:class:`~orpheus.numerics.iteration.SourceIteration`),
:class:`~orpheus.numerics.iteration.KrylovAcceleration` (default
preconditioner), the
:class:`~orpheus.numerics.green_operator.GreenOperator` builder and the
adjoint fixed-source entry all route through it, so the narrowing — and
its guard — is single-sourced.

When step 3 shipped, ``_seeded_inverse`` narrowed to the seeded-apply
signature only for the inverses the posing layers here actually reach —
the SN :class:`~orpheus.sn.operators.sweep_operator.SweepOperator` and the
leaf :class:`~orpheus.numerics.operator.InverseOperator`, both of which
carried ``apply(rhs, *, initial_guess=None)`` by per-leaf convention. It
was **not** a claim that *every* ``inverse()`` in the family took the
keyword, and whether that conformance should become **structural** (a
shared mixin, so the signature is *guaranteed* rather than
asserted-per-leaf) or stay per-leaf convention was the open decision
`#285 <https://github.com/deOliveira-R/ORPHEUS/issues/285>`_.

**Step 4 resolved #285 STRUCTURAL.** The wrap-delegate back-half was
extracted into :class:`~orpheus.numerics.operator.InverseWrapMixin`, whose
**abstract** ``apply(x, /, *, initial_guess=None)`` every sibling
inherits — so a new inverse *cannot* forget the keyword: pyright rejects a
kwarg-less override (``reportIncompatibleMethodOverride``,
mutation-verified) and ``ABCMeta`` blocks a sibling that omits ``apply``
entirely. The narrowing ``seeded_inverse`` performs is therefore now a
checked bridge over an *already-guaranteed* shape rather than a
hoped-for one, and every wrap-delegate ``.inverse()`` a posing layer
reaches through it carries the seeded signature by construction.

**Step 5 closed the product residue.** One ``.inverse()`` stayed outside the
family after step 4: a composed
:meth:`OperatorProduct.inverse <orpheus.numerics.operator.OperatorProduct.inverse>`
is a *composition* (:math:`(AB)^{-1}=B^{-1}A^{-1}`), not a wrap-delegate
sibling, and it returned a **raw reversed product** whose positional-only
``apply(x, /)`` raised ``TypeError`` when a driver seeded it. It now returns
:class:`~orpheus.numerics.operator.InverseOperator`\ ``(self)`` — the generic
family member wrapping the product. The action is **bit-identical**
(``apply`` delegates to the product's own ``solve`` = ``b.solve(a.solve(q))``,
exactly the two solves the reversed product composed), but the wrapper adds
the canonical seeded ``apply`` (the #285 ``TypeError`` repro flips to
*accepted* — the solve path never threaded seeds either, so behavior is
unchanged) and **strengthens the involution to object identity**
(``(A@B).inverse().inverse() is (A@B)``, where the reversed-product spelling
rebuilt fresh objects). The factors stay reachable as ``.inner.a`` /
``.inner.b``.

**Two kinds of inverse.** The closure exposes a clean split in what
``.inverse()`` returns across the whole algebra:

.. list-table:: Wrap-delegate vs algebra-closed inverses
   :header-rows: 1
   :widths: 26 34 40

   * - Kind
     - Members
     - What ``.inverse()`` returns
   * - **Wrap-delegate** (the #226 family)
     - :class:`~orpheus.numerics.operator.InverseOperator`,
       :class:`~orpheus.sn.operators.sweep_operator.SweepOperator`,
       :class:`~orpheus.numerics.green_operator.GreenOperator`,
       :class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`,
       and now the composed :class:`~orpheus.numerics.operator.OperatorProduct`
     - a thin typed wrapper around the *forward* :math:`A`, carrying the
       canonical seeded ``apply`` — the object *represents* :math:`A^{-1}`
       by inverting :math:`A` on demand.
   * - **Algebra-closed** (first-class forwards)
     - :class:`~orpheus.numerics.operator.PermutationOperator` (inverse *is*
       a permutation), :class:`~orpheus.numerics.operator.IdentityOperator`
       (self-inverse), :class:`~orpheus.numerics.operator.ScaledOperator`
       (:math:`(\alpha L)^{-1}=\alpha^{-1}L^{-1}`)
     - a first-class **forward** operator in the same closed structure — the
       *other* kind of inverse, deliberately left **unwrapped** (a
       permutation's inverse is nothing more exotic than a permutation, so
       wrapping it would only hide its structure).

The wrap-delegate mixin's ``_ForwardT`` bound was also relaxed from
``_InvertibleForward`` to the new minimal ``_WrappedForward`` Protocol
(``domain`` / ``codomain`` / ``apply`` — exactly what the back-half
consumes), which the 4th sibling forced:
:class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
inverts the *materialization* and never touches ``inner.solve`` or
``inner.is_invertible``, exposing the true minimum bound. The mixin and its
enforcement are documented at :ref:`green-operator`; the materialising
sibling, the ``as_matrix`` functor, and the two-error-class boundary at
:ref:`matrix-inverse-operator`.

The two eigenvalue-outer consumers pose the inverse explicitly:

* :class:`~orpheus.numerics.iteration.KEigenvalue` guards
  ``A.is_invertible`` **at construction** (a non-invertible :math:`A`
  raises :class:`~orpheus.numerics.operator.NotInvertible` with a
  domain message, not an ``AttributeError`` mid-iteration) and builds its
  inner :class:`~orpheus.numerics.iteration.SourceIteration` step via
  :func:`~orpheus.numerics.iteration.seeded_inverse`.
* :class:`~orpheus.numerics.iteration.KrylovAcceleration` keeps the
  **forward** :math:`A` (its GMRES matvec is
  :math:`A\cdot - \sum_i g_i\cdot`) and rewired its default-preconditioner
  fallback from the old ``CAP_SOLVE`` probe (with a ``# type: ignore``) to
  the honest ``A.is_invertible`` test + ``seeded_inverse(A).apply`` — the
  transport-corrected sweep preconditioner (Adams & Larsen 2002 §III).


Verification — the seed spy and the windowed×G-S corner
-------------------------------------------------------

The rewire was pinned by three structural gates (all ``-O``-proof —
``pytest.fail`` / ``np.testing.assert_*``, never a bare ``assert``;
``test_seed_threading_spy.py`` was **retired in #280 2.5c** with the
vestigial ``initial_guess`` threading it guarded — the live seed-contract
pins are ``tests/sn/operators/test_inverse_operator_equivalence.py`` and
``tests/sn/eigenvalue/test_keff_curvilinear.py``):

.. list-table:: Step-3 regression gates
   :header-rows: 1
   :widths: 42 58

   * - Gate
     - What it pins
   * - ``test_seed_threading_spy.py``
       (foundation / sentinel; vv Mode 11)
     - Every inner solve's ``initial_guess`` equals the previous iterate's
       return, by value. **Route-invariant across the rewire**: it wraps
       :meth:`StreamingCollisionOperator.solve <orpheus.sn.operators.streaming.StreamingCollisionOperator.solve>`
       — the surface *both* driver generations route through (pre-step-3
       ``resolvent.solve(...)``, post-step-3
       ``SweepOperator.apply`` → ``inner.solve``) — so it was green before
       the rewire and stayed green through it. Teeth: M-SEED-DROP / ZERO /
       STALE (pre-rewire) and **M-PROBE** (post-rewire) all redden it.
   * - ``test_2d_windowed_product_over_gauss_seidel_M_equals_post_projection``
     - The **windowed×G-S corner**: production's 2-D Cartesian default
       ``inner_schedule`` is ``gauss_seidel``, so the driver's *actual*
       step operator is ``P @ M.inverse()`` with
       :math:`M=(L+C)-B_{\rm lower}`. The fused scheduled-walk moment emit
       ≡ the deforested scheduled solve + projection, within scale-relative
       :math:`4N\varepsilon`, with a ``B_lower`` non-degeneracy guard.
       Closes the gap where this corner was pinned only at the
       :math:`\ell=0` / integration level.
   * - ``test_si_single_primitive_contract.py``
     - Both within-group SI paths (eigenvalue inner + fixed-source) build
       the SAME step operator: a
       :class:`~orpheus.sn.operators.sweep_operator.SweepOperator` whose
       ``.inner`` is the
       :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` forward
       — the forward identity moved *one level in*, onto ``.inner`` (the
       solver builds the inverse; the driver applies it).

The seed the spy guards is load-bearing **only on curvilinear** meshes: a
simulated seed-drop on the het-2G sphere moves the eigenvalue by
:math:`|\Delta k|\approx3.46\times10^{-2}` (the ``@slow`` value catcher
``test_si_krylov_eigenvalue_equivalence_sphere``, marked
``@catches("ERR-026", "M-SEED-DROP")``). On 2-D Cartesian the direct
wavefront ignores the seed, so the fast net is the *path* spy, not a value
gate. The full tier (SN + numerics + transport) is **2981 passing / 0
real regressions**; the deleted probe also cleared a laundered typed union
from the pyright ratchet (152 → 148).


.. _green-operator:

The Green operator — the preconditioned-splitting sum inverse
=============================================================

Steps 1–3 delivered the *exact/direct* inverse family: the leaf
:class:`~orpheus.numerics.operator.InverseOperator` (a pointwise division)
and the schedule-triangular
:class:`~orpheus.sn.operators.sweep_operator.SweepOperator` (the WDD sweep
as :math:`(L+C)^{-1}`), both invertible in a single pass. **Step 4 adds the
first *iterative* member** — the
:class:`~orpheus.numerics.green_operator.GreenOperator`, returned by
:meth:`OperatorSum.inverse <orpheus.numerics.operator.OperatorSum.inverse>`
— and with it the ``OperatorSum`` invertibility contract that routes to it.
It is the discrete Green's function of a general operator *sum*: the object
whose columns are the flux responses to unit point sources.


The convergent :math:`A`-preconditioned splitting
-------------------------------------------------

A general operator sum has **no operand-wise inverse**: :math:`(A+B)^{-1}`
is not a function of :math:`A^{-1}` and :math:`B^{-1}`
(Sherman–Morrison–Woodbury applies only under low-rank structure — absent
from the bulk blocks :math:`L`, :math:`C`, but exactly what the boundary
block :math:`B` has: :ref:`the scoped statement <smw-low-rank-exception>`,
Issue #300), and —
unlike the schedule-triangular ``(L+C)`` — there is no substitution order
that solves it in one pass. What a sum *does* have, when its **leading**
term :math:`A` is invertible, is a convergent **splitting**. Write the sum
as :math:`A - B` (gains carried with their signs) and precondition by
:math:`A^{-1}`:

.. math::
   :label: green-neumann-series

   (A - B)^{-1}\,q
   \;=\; \bigl(I - A^{-1}B\bigr)^{-1} A^{-1}\,q
   \;=\; \sum_{k=0}^{\infty} \bigl(A^{-1}B\bigr)^{k}\,A^{-1}\,q .

.. vv-status: green-neumann-series documented

The right-hand side is the **Neumann series** of the :math:`A`-preconditioned
splitting. In transport it *is* the **multiple-scattering expansion**:
:math:`A^{-1}q` is the uncollided flux, :math:`(A^{-1}S)\,A^{-1}q` the
once-rescattered contribution, :math:`(A^{-1}S)^{k}A^{-1}q` the
:math:`k`-times-rescattered contribution (:cite:`LewisMiller1984` §3.2;
:cite:`AdamsLarsen2002` §II). Its partial sums are **exactly** the Richardson /
source-iteration iterates started from zero,

.. math::
   :label: green-splitting-iteration

   x_{n+1} \;=\; A^{-1}\bigl(q + B\,x_n\bigr),
   \qquad x_0 = 0
   \;\Longrightarrow\;
   x_{n} = \sum_{k=0}^{n-1}\bigl(A^{-1}B\bigr)^{k}A^{-1}q ,

.. vv-status: green-splitting-iteration documented

which converge iff the iteration matrix is a contraction,
:math:`\rho(A^{-1}B) < 1`. For the within-group transport loss
:math:`A_{\rm loss} = (L+C) - S` this is the physical **scattering-ratio**
bound

.. (vv-status rationale) The scattering-ratio contraction bound
   ρ((L+C)⁻¹S) ≤ max Σ_s/Σ_t = c < 1 (Adams & Larsen 2002 §II).
   Literature-transcribed mathematical identity, matching the sentineled
   green-neumann-series / green-splitting-iteration siblings.
.. vv-status: green-scattering-ratio-bound documented

.. math::
   :label: green-scattering-ratio-bound

   \rho\bigl((L+C)^{-1}S\bigr)
   \;\le\; \max_{\rm cell}\ \frac{\Sigma_s}{\Sigma_t} \;=\; c \;<\; 1 ,

guaranteed below unity for any absorbing (physical) medium
(:cite:`AdamsLarsen2002` §II: :math:`\rho = c`). The convergence is thus a
material property, not a numerical accident — and it is precisely why the
sum has an inverse *operator* at all.


The name is earned — G-Neumann, G-reciprocity, G-kernel
-------------------------------------------------------

A subclass name in this family is a **promise backed by a test** (taxonomy
§13): a distinguishing invariant a *bare*
:class:`~orpheus.numerics.operator.InverseOperator` does not automatically
have. Round-trip alone (:math:`A^{-1}(Ax)=x`) earns only the generic name;
``GreenOperator`` must be *tested* to carry the Green's-function structure.

.. list-table:: The three name-earning invariants of ``GreenOperator``
   :header-rows: 1
   :widths: 22 78

   * - Invariant
     - What it asserts (and why a generic :math:`A^{-1}` fails it)
   * - **G-Neumann**
     - The partial sums of :eq:`green-neumann-series` equal
       ``green.apply(q)`` (the multiple-scattering expansion). A generic
       :math:`A^{-1}` has **no splitting** to satisfy — this is the
       *distinguishing* invariant. Pinned with the exact geometric decay of
       the split (see :ref:`green-verification`).
   * - **G-reciprocity**
     - The Green's-function reciprocity theorem in the **Euclidean** inner
       product, :math:`\langle\phi_2, G\phi_1\rangle = \langle
       G^{\mathsf T}\phi_2, \phi_1\rangle`, where :math:`G^{\mathsf T}` is
       the Green built over the **transposed operands** (:math:`A^{\mathsf T}
       = A^{\mathsf T}_{\rm lead} - B^{\mathsf T}`). It is the *cheap* proof
       (no second dense oracle) that the split derivation is correct for a
       *different* operand configuration.
   * - **G-kernel**
     - ``green.apply(δ_j)`` is column :math:`j` of :math:`(A-B)^{-1}` — the
       flux response to a unit point source at :math:`j`. Folded into the
       anchor's input set (the :math:`\delta_j` basis) rather than gated
       separately, since it *is* the forward anchor evaluated on unit
       vectors.

.. important::

   **G-reciprocity uses the Euclidean transpose, not the metric adjoint.**
   :math:`G^{\mathsf T}` here is the plain transpose under the :math:`L^2`
   pairing, built manually from the transposed operands — **not** the
   ``.H`` Hilbert adjoint :math:`G^{\dagger} = \mathcal G^{-1}
   G^{\mathsf T}\mathcal G` carrying the angular Gram metric. The
   adjoint-inverse axis (``G.H == (A-B).H.inverse()`` — free at the object
   level on the iterative branch, but blocked on the SN preconditioner's
   as-yet-unbuilt multi-D transpose sweep) is the separate
   `#280 <https://github.com/deOliveira-R/ORPHEUS/issues/280>`_ family and
   is deliberately *not* promised here.

The family is keyed by **which mathematical object the inverse is, not by
the algorithm that realizes it**. A Richardson-realized Green and a
GMRES-realized Green are the *same* Green operator — which is why
``KrylovInverseOperator`` was **rejected** as a sibling name: "Krylov"
names the orthogonal *realization* axis, not the object.


Green wraps the driver — it re-implements nothing
-------------------------------------------------

``GreenOperator`` is a thin wrapper over a driver, not a new solver
(taxonomy §11.2, Pattern 5). Its application engine is
:class:`~orpheus.numerics.iteration.SourceIteration` — the **same**
Richardson driver the SN solver and
:class:`~orpheus.numerics.iteration.KEigenvalue` consume directly. At
construction it derives the splitting **once, from the sum's structure**:

* the **left-spine head** becomes the preconditioner :math:`A^{-1}`,
  through *its own* structure-keyed ``.inverse()`` — the WDD
  :class:`~orpheus.sn.operators.sweep_operator.SweepOperator` for an
  ``(L+C)`` head, a leaf
  :class:`~orpheus.numerics.operator.InverseOperator` for a value-bearing
  head;
* every remaining term becomes a **negated gain** of the driver (the
  ``A - S`` spelling arrives as ``ScaledOperator(-1, S)``; the gain is
  :math:`S` itself, un-wrapped so the driver holds the *named* operator).

It then hands these to ``SourceIteration`` and re-implements no iteration
math of its own. The left spine is flattened by walking **exact**
``OperatorSum`` nodes only: the fused
:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` is a structural
*leaf* (its sum-ness is an MRO fact, its identity a fused operator with a
direct inverse), so ``((L+C) - S)`` flattens to preconditioner ``(L+C)`` +
gain ``[S]`` — never dissolving the ``(L+C)`` into its own summands. A
GMRES-realized Green (with
:class:`~orpheus.numerics.iteration.KrylovAcceleration` as the engine) is a
future realization *strategy* of this same object, not a sibling type.


The ordering ruling — four edges of one canonical order
-------------------------------------------------------

Operand **spelling** selects the algorithm — the #261 canonical-ordering
rule, extended to the sum inverse. The four edges:

.. list-table:: How a sum's spelling routes its inverse
   :header-rows: 1
   :widths: 20 26 54

   * - Spelling
     - ``.inverse()`` builds
     - Behaviour
   * - ``L + C``
     - :class:`~orpheus.sn.operators.sweep_operator.SweepOperator`
     - The fusion dispatch on
       :meth:`StreamingOperator.__add__ <orpheus.sn.operators.streaming.StreamingOperator.__add__>`
       (streaming.py:510, "the canonical and only ordering") returns the
       :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator`
       specialisation, whose ``.inverse()`` override (→ the direct sweep)
       **shadows the generic Green by MRO** (type-as-structure, §11.1).
   * - ``A_loss = (L+C) - S``
     - :class:`~orpheus.numerics.green_operator.GreenOperator`
     - Leading ``(L+C)`` invertible → the physical splitting: preconditioner
       the sweep, gain :math:`S`. Converges (:math:`c<1`); does **not**
       raise.
   * - ``C + L``
     - :class:`~orpheus.numerics.green_operator.GreenOperator`
     - A *legal* spelling whose leading term ``C`` happens to be invertible,
       so a Green **constructs** — the algebra cannot read
       :math:`\rho(C^{-1}L) > 1`. Its collision-preconditioned Richardson
       **diverges**, and ``apply`` raises
       :class:`~orpheus.numerics.green_operator.ConvergenceFailure`
       **loudly**. Same math as ``L + C``, different algorithm by spelling —
       never a silent wrong answer (Cardinal Rule 1).
   * - ``(-S) + A``
     - *(refused)*
     - The left-spine head :math:`-S` is not invertible; the factory raises
       :class:`~orpheus.numerics.operator.NotInvertible` at
       **construction**, naming the canonical ordering (spell the invertible
       operator first, ``A - S``).

The keying predicate is honest about its scope.
:attr:`OperatorSum.is_invertible <orpheus.numerics.operator.OperatorSum.is_invertible>`
returns ``self.a.is_invertible`` — the *leading* term — and therefore reads
"**leading-term-preconditionable at this operand order**", **not**
spelling-independent mathematical invertibility (verification spec §18.B).
For ``(-S) + A`` the operator :math:`A - S` *is* mathematically invertible,
yet the predicate reports ``False`` because :math:`-S` is spelled first.
This is acceptable precisely because the #261 rule already makes operand
order semantically load-bearing, no production consumer relies on the
spelling-independent meaning, and the refusal is **loud**. Since carve
P4 there is no parallel ``CAP_SOLVE`` tag to keep in lockstep:
``is_invertible`` is the *single* advertisement, and the keystone v2
faithfulness contract (:ref:`capability-set-semantics`) pins that
``.inverse()`` returns exactly when ``is_invertible`` is ``True``.


The promise — the TRUE residual, driven not merely checked
----------------------------------------------------------

``GreenOperator`` promises the **converged** :math:`A^{-1}q` or a loud
:class:`~orpheus.numerics.green_operator.ConvergenceFailure` — never a
silent partial iterate. The subtlety is *which residual* defines
"converged". :class:`~orpheus.numerics.iteration.SourceIteration` stops on
the iterate **increment** :math:`\lVert\Delta\psi\rVert / \lVert\psi\rVert
< {\rm tol}`, which **understates** the true equation error by the factor
:math:`\rho/(1-\rho)` (numerical-bug-signatures Signature 9, ρ-blind
stopping). As :math:`\rho \to 1` an increment-converged iterate can sit
orders of magnitude off the equation. The promise is therefore read on the
**true relative residual**:

.. math::
   :label: green-true-residual

   \frac{\bigl\lVert (A - B)\,\psi - q \bigr\rVert}{\lVert q \rVert}
   \;<\; {\rm tol} .

.. vv-status: green-true-residual documented

Crucially the promise is **driven, not merely checked**. A check-only design
(raise whenever :eq:`green-true-residual` fails at increment-stop) would
falsely raise for **every** split with :math:`\rho > 1/2` — i.e. most
physical :term:`scattering ratios <scattering ratio>` — because increment-stop delivers only
:math:`\rho/(1-\rho)\cdot{\rm tol}` there.
:meth:`GreenOperator.apply <orpheus.numerics.green_operator.GreenOperator.apply>`
instead runs a **refinement loop**: after each increment-stopped driver
call it measures :eq:`green-true-residual` and, if unmet with budget
remaining, re-seeds the driver with its own iterate (steps accumulate
against one total ``max_iter``). The driver stays the sole iteration engine;
the loop is tolerance bookkeeping only, at one extra forward matvec per
check.

The refinement's terminal ``raise`` also has to be **NaN-safe**, because a
hard-divergent split (the ``C + L`` trap) produces two distinct
floating-point failure shapes — both found by the divergence tooth
(2026-07-02):

#. the iterate itself overflows, so the increment is ``nan``; and
#. — sharper — the driver's stopping **denominator** :math:`\lVert\psi\rVert`
   overflows to ``inf`` one step *before* the numerator, so the driver
   "converges" at ``increment = finite/inf = 0.0`` onto a ~\ :math:`10^{154}`
   garbage iterate.

Both are caught because the promise test reads the true residual of the
returned iterate (huge or non-finite in either shape), never the driver's
increment. This is exactly the "what was tried and failed" material a naive
increment-check would have shipped as a silent wrong answer.


Green versus the k-eigenvalue inner — normal form versus inexact relaxation
---------------------------------------------------------------------------

``K = A_loss.inverse() @ F`` is the **normal form** of the k-eigenvalue
problem: the eigen-operator whose dominant eigenvalue is :math:`k`. But
production power iteration **deliberately does not** build this exact
``GreenOperator`` — it keeps consuming
:class:`~orpheus.numerics.iteration.SourceIteration` directly, as a
**warm-started, budget-bounded inner relaxation**. That partial convergence
is *by design* in nested iteration (classic inexact power iteration): the
inner need only be converged enough that the *outer* dominant-eigenvector
direction is accurate, and warm-starting from the previous outer iterate
amortises the cost. This is a **different contract** from Green's
converged-or-raise promise, and conflating them would either over-solve the
inner (wasted matvecs) or import Green's hard raise into a loop that
legitimately runs partial solves. So
:class:`~orpheus.numerics.iteration.KEigenvalue` was examined and
**deliberately not rewired** to ``A_loss.inverse() @ F``; the normal form is
the *concept*, the inexact inner is its production *realization*.

Green's consumers today are the invariant gates (see
:ref:`green-verification`). Production consumers arrive later: the #200
preconditioner algebra, a diffusion ``.inverse()`` realized as a
CG-preconditioned Green (the taxonomy's *negative control* — an iterative
inverse with no sweep), and future explicit normal-form spellings of the
k-problem.


The wrap-delegate mixin — extracted at the third sibling
--------------------------------------------------------

Every inverse in this family is a thin typed wrapper around its own forward
operator :math:`A`: ``apply`` realizes :math:`A^{-1}` by the sibling's
algorithm, and *everything else is delegation*. That back-half —
``is_invertible = True`` with ``solve`` = the forward matvec
``inner.apply`` (solving :math:`A^{-1}y = b` *is* applying :math:`A`),
the domain↔codomain swap
(an inverse maps the forward's codomain back to its domain), and the
object-identity involution ``inverse() → inner``
(:math:`(A^{-1})^{-1} = A`) — was carried byte-identically by
:class:`~orpheus.numerics.operator.InverseOperator` and
:class:`~orpheus.sn.operators.sweep_operator.SweepOperator` as documented
twins. ``GreenOperator`` is the **third** sibling, and it fired the
extraction trigger both twins recorded (defer-until-≥2, extract at 3): the
shared mechanism lifted into
:class:`~orpheus.numerics.operator.InverseWrapMixin`. The forward-operand
Protocol was renamed ``_SolveBackedLeaf`` → ``_InvertibleForward`` to match
its now-general role as the mixin's ``_ForwardT`` bound. Each sibling keeps
exactly three things of its own: the constructor **guard** (what makes its
``inner`` invertible — a value check, a type, a derivable splitting), the
``apply`` **body** (the inversion algorithm), and ``__repr__``.

The mixin also carries the **abstract** seeded-apply signature ``apply(x,
/, *, initial_guess=None)`` — the structural resolution of #285 discussed at
:ref:`inverse-application-driver`: a new sibling *cannot* forget the keyword
(pyright rejects a kwarg-less override; ``ABCMeta`` blocks a missing one).
The adjoint axis is **not** part of the back-half — ``is_adjointable`` /
``.H`` stay at the base defaults, deferred to the #280 family.


.. _green-verification:

Verification
------------

``GreenOperator`` is the **first** inverse in the family with **no legacy
``.solve`` to inherit from** — the sum was not invertible before step 4, so
there is no bit-identity twin to ride. Its correctness therefore rests
**entirely on structurally-independent anchors**, never on inheritance.
Every Green value gate is a foundation / flux-shape claim against such an
anchor: **no eigenvalue claim, no MMS reference** (an iterative sum inverse
is not an eigenvalue solver, and MMS is source-driven — neither pillar
applies here). The ``inverse().apply ≡ solve`` equivalence that anchored
steps 1–3 is a **tautology** for the sum (since carve P4 the generic sum
carries **no** ``solve`` at all — ``inverse().apply`` *is* its only
inverse action, so no second spelling exists to compare) and is
deliberately excluded as evidence.

.. list-table:: Step-4 verification gates (all ``@pytest.mark.foundation``)
   :header-rows: 1
   :widths: 42 58

   * - Gate
     - What it pins
   * - ``test_green_operator.py`` (L0, dense split ``A = D − αP``)
     - **G-I1** round-trip both ways at driver tol + the **dense-LU anchor**
       (:func:`numpy.linalg.solve` of the materialized sum), including the
       :math:`\delta_j` basis (the G-kernel fold). **G-Neumann** partial
       sums converge to ``green.apply(q)`` with the **exact** 4-cycle decay
       ratio :math:`\rho = \alpha/(\prod d_i)^{1/4}` (the permutation's
       spectrum makes the pin exact, not a fuzzy band). **G-reciprocity**
       via the transposed-operand Green (non-symmetric ``A``, so not
       vacuous). Divergence + near-critical (:math:`\rho=0.99`) raises, each
       with a convergent control.
   * - ``test_green_operator_sn.py`` (L1, het 2G **vacuum** slab)
     - **G-Neumann-L1** on the real operators: :math:`\sum_k ((L+C)^{-1}S)^k
       (L+C)^{-1}q` → ``green.apply(q)`` with the geometric tail of the
       physical scattering ratio. The anchor is a **trace-consistent
       manufactured** pair (``x_tc = (L+C)⁻¹(random)``, ``q = A_loss·x_tc``),
       which resolves the #284 source-subspace caveat — sweep inverses are
       exact on the source subspace, so the exact solution *is* ``x_tc`` with
       no dense-LU trace mismatch. Plus **driver bit-identity** vs the
       hand-built ``SourceIteration(sweep, S)``, and the four ordering-ruling
       edges end-to-end.

The config discipline is Mode-9: the L1 slab is **heterogeneous, ≥2-group,
vacuum** — a reflective isotropic box would null the
streaming↔scattering redistribution the Neumann series expands, and 1-group
is blind to a scattering-matrix transpose (Mode 6). The teeth are
**14 mutations verified** (2026-07-02): 12 bite their named gates
(sign/swap/flatten/tol/increment/seed/order/…), and 2 are designed-green
controls — the always-wrap ``_negated`` no-op (proving the gain-unwrap is
pure deforestation) and the **M-GRN-SEED blindness proof** (a dropped
driver-start reddens Green's own Mode-11 seed spy while the landed step-3
spy and every value gate stay green, proving the step-3 spy is structurally
blind to Green's driver-start threading). The pyright ratchet held exactly
at baseline; the Sphinx build stayed ``-W`` clean.


.. _matrix-inverse-operator:

The materialising functor and the dense direct inverse
======================================================

Steps 1–4 built the inverse *operators* — the exact leaf
:class:`~orpheus.numerics.operator.InverseOperator`, the
schedule-triangular
:class:`~orpheus.sn.operators.sweep_operator.SweepOperator`, and the
iterative :class:`~orpheus.numerics.green_operator.GreenOperator` — each a
matrix-free wrapper that realizes :math:`A^{-1}` by *applying* it. **Step 5
adds the** *materialising* **family**: the serialization boundary
:meth:`~orpheus.numerics.operator.LinearOperator.as_matrix` promoted to a
universal ``LinearOperator`` base method, and the fourth
:class:`~orpheus.numerics.operator.InverseWrapMixin` sibling
:class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator` —
the inverse of a **structureless small** operator, obtained by
materializing :math:`[A]`, factoring it once (LU), and back-solving every
:meth:`apply`. The step also **closes #285** for composed inverses (the product residue
retired in the #285 structural-resolution section above) and retires the
homogeneous solver's ``_as_dense`` prototype into the new base method.


``as_matrix`` — the functor out of the operator category
--------------------------------------------------------

The inverse, adjoint, and composition maps of :ref:`operator-algebra` are
all **endofunctors** :math:`\mathrm{Op}\to\mathrm{Op}`: each takes an
operator to another operator. ``as_matrix`` is the **fourth arrow** of the
taxonomy (§2) — the functor *out* of the operator category,
:math:`\mathrm{Op}\to\mathrm{Mat}`, the serialization boundary that leaves
the matrix-free world entirely. It is realized as the **apply-to-basis**
loop: column :math:`j` is the operator applied to the :math:`j`-th basis
element,

.. math::
   :label: matrix-functor-out

   [A]_{:,j} \;=\; \operatorname{ravel}\bigl(A\,e_j\bigr),
   \qquad
   e_j \;=\; \operatorname{unravel}(\delta_j),
   \qquad
   j = 0,\dots,\textstyle\prod_k b_k - 1,

.. vv-status: matrix-functor-out documented

with basis elements enumerated in **C-order** over the carrier shape
:math:`(b_0,\dots,b_{d-1})` and each output raveled the same way, so
:math:`[A]\,x_{\rm flat} = (A\,x)_{\rm flat}` exactly and — for a
group-leading :math:`(n_g,1)` carrier — column :math:`j` is the response to
a unit source in group :math:`j`. The matrix is
:math:`(\prod\text{out shape})\times(\prod_k b_k)`: **rectangular whenever
the operator is not endomorphic**, because the output dimension emerges
from :meth:`apply` itself, never from declared metadata (a coisometry
materializes as a genuinely wider-than-tall block, :ref:`windowing-retyped`).

This default body is the **promoted** ``_as_dense`` apply-to-basis loop
that lived privately in ``homogeneous/solver.py`` until step 5. Its
promotion to the base class is the Cardinal-Rule-2 move: the one place that
turned an operator into a dense matrix becomes the method *every* operator
inherits, and the homogeneous solver drops its private copy
(:ref:`matrix-inverse-consumers`).


The resolution rule and the two error classes
---------------------------------------------

Two questions must be answered before the loop runs: *what shape* are the
basis elements, and *is the materialization affordable*. They raise **two
deliberately distinct exception classes** (spec §27.C), because they are
different kinds of failure.

**Basis-shape resolution** is single-sourced in ``_resolve_basis_shape``
(shared with the eager
:class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
constructor, which must know the resolved shape to reshape solutions back
into carriers — one source, no reciprocal drift). The rule has three arms:

#. an explicit ``basis_shape`` wins;
#. else the operator's own
   :attr:`~orpheus.numerics.operator.LinearOperator.domain` supplies
   ``domain.shape``;
#. else — **no domain and no explicit shape** — the request is *ill-posed*:
   the basis cannot be derived at all, and the method raises ``ValueError``
   naming both remedies (construct the operator with a space, or pass an
   explicit ``basis_shape=(n_g, 1)``).

**The size gate** is orthogonal. When the resolved dimension
:math:`n=\prod_k b_k` exceeds ``max_dimension``, the method raises
:class:`~orpheus.numerics.operator.MatrixTooLarge` — a *well-posed* request
that this environment *refuses on resource grounds*.

The two must never be caught with one loose ``except (ValueError,
RuntimeError)``: a bug that collapsed the two would pass such a test. The
boundary gates pin each separately, and their ``pytest.raises`` matches are
class-discriminating — a Pattern-4 illegal-states check, "un-materializable
*as posed*" :math:`\neq` "too big to materialize *here*".


The size gate and the dense-vs-sparse ruling
--------------------------------------------

:class:`~orpheus.numerics.operator.MatrixTooLarge` is a **RuntimeError**,
not a ``ValueError`` — and there is deliberately **no** ``is_materializable``
predicate beside
:attr:`~orpheus.numerics.operator.LinearOperator.is_invertible`. The reason
is the taxonomy §17 A2 distinction: ``as_matrix`` is a **total functor**.
*Every* finite-dimensional linear operator *has* a matrix; nothing about
its structure or values can make it "un-materializable". What the gate
guards is a **resource effect** — materializing commits :math:`O(n^2)`
memory and :math:`n` applications — so the refusal is a runtime resource
precheck (it reads nothing but a dimension), *not* a structural restriction
(``is_invertible`` / ``is_adjointable`` read the operator's structure *and*
values). The default budget is ``max_dimension = 4096`` — a :math:`4096^2`
float64 is 134 MB and 4096 applications, generous for every
dense-by-construction consumer (0-D energy spectra, CP :math:`[P]`) and
prohibitive for a meshed SN full-field operator by design. It is a per-call
knob: ``try: A.as_matrix() except MatrixTooLarge: <iterative path>``, or
raise the budget for one call.

**The return is a dense** :class:`numpy.ndarray`. The dense-vs-sparse
return keying was the one open thread (W4) that taxonomy §11.4 explicitly
**deferred to step 5**, and the ruling is: *keyed by the operator's
structural override, with dense the only realization built until a sparse
consumer exists*. A structured operator MAY override ``as_matrix`` with a
direct assembly — the future per-octant sparse-triangular streaming
assembly noted at ``sweep_graph.py:66`` is exactly such an override, and it
is **deferred with its 3-D consumer** (there is no sparse consumer today,
so building the sparse path now would be a primitive with no product —
defer-until-consumer, Cardinal Rule 2).
:class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator` is
the one override that ships in step 5, and it collapses to a single batched
LU back-solve (below), not a sparse form.

.. note:: **Landed (stencil-assembly 2b, 2026-07-04).** The deferred
   sparse-assembly override arrived. Not yet the *octant-batched 3-D*
   form — that vectorization is still deferred with its 3-D consumer —
   but the assembly axis itself: structured leaves now emit a
   :class:`~orpheus.numerics.assembled_operator.SparseAssembledOperator`,
   and ``as_matrix`` **delegates** to the densified emission when the
   operator is :func:`~orpheus.numerics.operator.assemblable` (the
   probing loop retained as the fallback and fuller-view oracle). The
   dense-vs-sparse keying above is now *realised* rather than deferred —
   the key is exactly the ``is_assemblable`` predicate. See
   :ref:`operator-algebra-assembly-axis`.


``MatrixInverseOperator`` — the dense direct inverse
----------------------------------------------------

``A.inverse()`` is a **structure-keyed factory** (taxonomy §13): the
concrete subclass names the mathematical *object* the inverse is, never the
algorithm. A schedule-triangular forward returns the direct-substitution
:class:`~orpheus.sn.operators.sweep_operator.SweepOperator`; a general sum
with an invertible leading term returns the preconditioned-splitting
:class:`~orpheus.numerics.green_operator.GreenOperator`; a value-bearing
leaf returns the pointwise
:class:`~orpheus.numerics.operator.InverseOperator`.
:class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator` is
the inverse of a **structureless small** operator — the 0-D energy
spectrum, a CP collision-probability block, any composition whose only
exploitable property is that it *fits*. Its algorithm is the honest
consequence of "no structure to exploit, but small enough to hold":
materialize :math:`[A]` once, ``lu_factor`` it once, and every
:meth:`apply` is a direct ``lu_solve`` back-solve; its
:meth:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator.as_matrix`
override collapses the base loop to one **batched** ``lu_solve(lu, I)``
against the same factors.

It is the fourth
:class:`~orpheus.numerics.operator.InverseWrapMixin` sibling, so its entire
back-half — ``is_invertible = True``, the domain↔codomain swap,
``solve`` = the forward matvec ``inner.apply``, and the object-identity
involution ``inverse() → inner`` — is inherited. It keeps only the family's
three per-sibling pieces: its constructor guard, its ``apply`` body, and
``__repr__``. Its home is a leaf module (the ``green_operator.py``
placement precedent), and — *unlike* Green — nothing in ``operator.py``
routes back to it: **no automatic** ``.inverse()`` **returns this type**,
so there is no late-import seam at all, and (step 5 scope) it is **direct
construction only**, never a factory-dispatch target. The
``A.inverse() → MatrixInverseOperator`` routing and the normal-form
spelling ``K = MatrixInverseOperator(loss) @ F`` land later (task #138 / CP).


The name is earned — M-materialise and M-direct at the precision grain
----------------------------------------------------------------------

A subclass name in this family is a **promise backed by a test**
(taxonomy §13): a distinguishing invariant a bare
:class:`~orpheus.numerics.operator.InverseOperator` does not automatically
have. For ``MatrixInverseOperator`` the promise is the **Matrix** invariant
in two faces — M-materialise (the explicit inverse, both ways) and M-direct
(the true residual):

.. math::
   :label: matrix-inverse-materialise

   \bigl\lVert\,[A^{-1}]\,[A] - I\,\bigr\rVert
   \;=\;
   \bigl\lVert\,[A]\,[A^{-1}] - I\,\bigr\rVert
   \;\le\; K\,\varepsilon_{\rm mach}\,\kappa(A)
   \qquad\text{(M-materialise, both ways)} ,

.. vv-status: matrix-inverse-materialise documented

.. math::
   :label: matrix-inverse-direct-residual

   \frac{\bigl\lVert A\,(A^{-1}q) - q\bigr\rVert}{\lVert q\rVert}
   \;\le\; K\,\varepsilon_{\rm mach}\,\kappa(A)
   \qquad\text{(M-direct, seed-independent)} .

.. vv-status: matrix-inverse-direct-residual documented

**What earns the name is the precision GRAIN, not the existence of a
matrix** — and this **supersedes the taxonomy §13 M-row parenthetical**
("a matrix-free inverse *cannot* satisfy M-materialise, having no
``as_matrix``"). That parenthetical is now **false**: step 5 made
``as_matrix`` universal, so an *iterative*
:class:`~orpheus.numerics.green_operator.GreenOperator` **also** satisfies
:math:`[\,\text{green}\,]\,[A]\approx I` — each column ``green.apply(e_j)``
:math:`\approx A^{-1}e_j`. "No ``as_matrix``" no longer distinguishes the
direct inverse; the **grain of the approximation** does:

.. list-table:: Why the grain is the name-earner (spec §27.A)
   :header-rows: 1
   :widths: 24 38 38

   * - Face
     - ``MatrixInverseOperator``
     - an iterative Green over the same :math:`A`
   * - :math:`[A^{-1}]\,[A]-I`
     - one batched ``lu_solve`` on the identity against the SAME
       factorization — **machine·cond**, no second realization, no
       iteration floor
     - :math:`n` iterative solves, each to **driver tolerance**
       (:math:`\sim 10^{-8}`) — floors at its stopping tol
   * - :math:`\lVert A(A^{-1}q)-q\rVert`
     - a direct solve — **machine·cond** true residual
     - floors at the driver tolerance, never machine grain

So M-materialise and M-direct are gated at **machine·cond** grain
(:math:`\text{atol}=K\,\varepsilon_{\rm mach}\,\kappa`), and the gate carries
an explicit **contrast**: the *same* :math:`A`, wrapped as a
:class:`~orpheus.numerics.green_operator.GreenOperator`, meets only
driver-tol — proving the invariant *distinguishing*, not merely satisfied
(the §21 discipline "a bare invariant a generic inverse also satisfies is
not name-earning"). M-direct is also the **seed-independence** face: an
exact direct inverse has nothing to seed, so the canonical
``initial_guess`` keyword is accepted and *ignored*, and the result is
bit-identical under any seed — contrast the sweep's Carlson closure and
Green's splitting start, which *consume* it.


Values, not structure — the guard difference and the witness
------------------------------------------------------------

The constructor deliberately **does not consult** ``inner.is_invertible``.
That predicate is *structural* — it reads the operand tree, and for a sum it
reports "leading-term-preconditionable at this spelling"
(:ref:`green-operator`), a property of the *spelling*, not of the matrix.
``MatrixInverseOperator`` reads **values**: it materializes :math:`[A]` and
factors it, and a matrix is either numerically invertible or it is not,
regardless of how the operator that produced it was spelled. The guards
that remain are the honest value-level ones, all raised at **construction**
(this module family's composition-time-not-call-time principle), never at
apply time:

.. list-table:: The three construction-time guards
   :header-rows: 1
   :widths: 22 78

   * - Guard
     - Behaviour
   * - **Size**
     - :class:`~orpheus.numerics.operator.MatrixTooLarge` propagates from
       the eager materialization (the size gate above).
   * - **Squareness**
     - a rectangular materialization has no two-sided inverse — a
       ``ValueError`` in domain language ("M-materialise is unsatisfiable").
   * - **Exact singularity**
     - a zero LU pivot raises :class:`numpy.linalg.LinAlgError`. scipy's
       *own* singularity signal is only a ``LinAlgWarning`` (``getrf`` info
       :math:`>0`), which would let a zero pivot flow into ``inf`` / ``nan``
       back-solves — the constructor **silences that warning and raises the
       loud error** from the U-diagonal instead (Cardinal Rule 1: fail at
       construction, never return a non-inverse). **Near**-singularity is
       *not* refused — it is priced into the M-direct :math:`\kappa(A)`
       bound.

**The witness.** The values-vs-structure difference is provable, and it is
the taxonomy §3 ``strategy=`` override seam realized *honestly* — not a flag
on ``.inverse()`` but an explicit construction by a consumer who knows the
problem is small (**the type IS the strategy choice**). The witness is a sum
whose **leading term is not invertible**: the canonical motivating form is
the SN :math:`(-S)+(L+C)`, which
:class:`~orpheus.numerics.green_operator.GreenOperator` **refuses at
construction** (left-spine head :math:`-S` non-invertible, canonical-ordering
:class:`~orpheus.numerics.operator.NotInvertible`) yet which materializes
to a perfectly invertible matrix. Because the real :math:`(-S)+(L+C)` is a
``FullField`` carrier — **out of ``as_matrix``'s ndarray scope** (the
honest-scope note below) — the gate proves the identical refusal/inversion
asymmetry on the realizable ndarray analog :math:`(-S_{\rm ao})+D` (an
apply-only leaf :math:`S_{\rm ao}`, ``is_invertible = False``, plus a
diagonal :math:`D`): ``.inverse()`` refuses it, while
``MatrixInverseOperator`` materializes :math:`D-S_{\rm ao}` and inverts it,
anchored against :math:`\mathrm{np.linalg.solve}(D-S_{\rm ao},\,q)`. The
*structure* — a leading-non-invertible sum that is nonetheless an invertible
matrix — is what the witness proves; the physics is incidental.

.. note::

   **Honest scope — ``as_matrix`` serves ndarray carriers only.** The
   apply-to-basis loop builds bare-ndarray :math:`e_j`; a **typed-carrier**
   (``FullField`` SN composite) operator's ``apply`` needs a ``FullField``
   basis vector and sits far above any sane size gate, so it stays
   matrix-free. Step 5 gates ndarray-carrier operators only (diagonal /
   permutation / the model-shared energy leaves / small compositions /
   dense test operators); the ``FullField`` :math:`(L+C)` and
   :math:`(-S)+(L+C)` materialization is a future carve (its 3-D
   sparse-triangular sibling is the deferred ``sweep_graph.py:66``
   override). This is exactly why the witness above uses the ndarray analog.


.. _matrix-inverse-consumers:

The consumer ruling — the latent normal form
--------------------------------------------

Like :class:`~orpheus.numerics.green_operator.GreenOperator` at step 4,
``MatrixInverseOperator`` **shipped** at step 5 verified but not yet wired as
a production spelling — its value gates constructed it directly, and the
factory routing waited. Taxonomy **step 5b closed that loop**: the homogeneous
solver is now the **first production consumer** (below). What still waits is
the *factory routing* — an automatic ``.inverse()`` that returns this type —
which homogeneous deliberately **bypasses** (see the ``direct_eigenvalue``
bullet). The consumers, from latent to landed:

* **The retired ``_as_dense``.** The homogeneous solver's private
  apply-to-basis loop was retired into
  :meth:`~orpheus.numerics.operator.LinearOperator.as_matrix` at step 5, so
  the materialization of the loss operator :math:`\mathbf A = C - K_{\rm iso}`
  and the fission dyad :math:`\mathbf F = \chi\otimes\nu\Sigma_f` both flow
  through the one promoted base method rather than a bespoke loop. The output
  stayed **byte-identical** through that retirement (same basis columns, same
  C-order, same eigen call), and the landed SymPy :math:`k_\infty` pins stayed
  green untouched. (Step 5b then re-spelled the whole eigensolve as
  ``K = MatrixInverseOperator(loss) @ production``: the loss materialization
  now happens *inside* the inverse operator's constructor, and the product's
  own ``as_matrix`` forms :math:`[\mathbf K] = \mathbf A^{-1}\mathbf F`.) See
  :ref:`theory-homogeneous`.
* **``direct_eigenvalue`` — the latent consumer, now realized.**
  :func:`~orpheus.numerics.eigenvalue.direct_eigenvalue`'s dense
  ``np.linalg.solve(A, F)`` **is** this operator's action written as free
  functions — the latent consumer that made the promotion finishable
  (taxonomy §5: "no current consumer" often means "consumer not yet
  wired"). The engine itself stays **ndarray-pure** — its closed-form
  verification is the point — so it was never going to be the *production*
  spelling. That spelling landed at **task #138 (step 5b)**: the homogeneous
  solver now composes ``K = MatrixInverseOperator(loss) @ production`` and
  eigendecomposes ``K.as_matrix()``, making it the **first production
  consumer** of the class. It constructs the matrix inverse *explicitly*
  rather than via the structure-keyed ``loss.inverse()`` — which, reading the
  sum :math:`C - K_{\rm iso}` (invertible leading term), would return the
  iterative :class:`~orpheus.numerics.green_operator.GreenOperator` splitting
  — the direct-realization strategy encoded as a type. With the operator
  spelling live, ``direct_eigenvalue`` has **zero production consumers**,
  retained as the ``(A, F)``-posed sibling engine and the RQI test oracle.
  See :ref:`direct-eigensolve-solve`.
* **CP :math:`[P]` — the next production method in line.** The
  collision-probability matrix is dense **by construction**
  (:doc:`/theory/methods/collision_probability`, §14b), so a CP ``.inverse()`` realized as a
  ``MatrixInverseOperator`` is the next production consumer after homogeneous.

This is the **same consumer ruling** ``GreenOperator`` shipped under at step
4: the type is correct and pinned, and its consumers arrive incrementally —
homogeneous first (step 5b), CP next. The *factory dispatch* that would route
``.inverse()`` to this type automatically is the remaining seam; explicit
construction by a size-aware consumer is the honest interim (and, for
homogeneous, the deliberate permanent choice).


Verification
------------

``MatrixInverseOperator`` is a **direct** inverse, so — unlike the iterative
:class:`~orpheus.numerics.green_operator.GreenOperator` — its correctness is
asserted at **machine·cond** grain against a **closed-form** dense reference
(hand-built ``Diagonal`` / ``Permutation`` matrices; the
structurally-independent
:meth:`~orpheus.transport.operators.isotropic_transfer.IsotropicScattering.dense_per_material`
*storage* transpose for the energy leaves), never at a driver tolerance. It
shares Green's honest framing in one respect: it is not an eigenvalue solver
and is not source-driven, so **no gate here makes an eigenvalue claim on an
MMS reference**; the homogeneous :math:`k_\infty` that rides through the
retired-loop relocation is anchored on its landed **closed-form** SymPy
value (``test_kinf_exact``, 1e-12), the analytical pillar.

.. list-table:: Step-5 verification gates (all ``@pytest.mark.foundation``)
   :header-rows: 1
   :widths: 42 58

   * - Gate file
     - What it pins
   * - ``tests/numerics/test_matrix_inverse_operator.py``
     - The base ``as_matrix`` L0 (exact vs hand-built + the storage-oracle
       cross-check; the **C-order column convention on a non-symmetric op**;
       rectangular-honesty; :math:`\equiv` the retired ``_as_dense`` loop),
       the ``ValueError`` / ``MatrixTooLarge`` boundary (class-discriminated,
       with an **at-threshold** designed-green control), and the
       ``MatrixInverseOperator`` invariants (M-materialise + M-direct at
       machine·cond with the **Green driver-tol contrast**, seed
       bit-identity, the back-half anchor, the non-square / singular /
       too-large guards with a positive constructs-cleanly control, and the
       :math:`(-S_{\rm ao})+D` **witness**).
   * - ``test_inverse_universal.py`` /
       ``test_operator_capability_predicates.py`` /
       ``test_operators_apply_typed.py``
     - The 4th sibling's participation in the universal family: the
       direct-construction registry row, object faithfulness, and the static
       ``assert_type`` / ``SupportsSeededApply`` conformance (a kwarg-less
       ``apply`` override is a CLI-pyright
       ``reportIncompatibleMethodOverride``).

The config discipline follows the family: the column convention is pinned on
a **non-symmetric** operator (a symmetric one is blind to a transpose,
Mode 6), and every value gate anchors on the **hand-built** reference, not on
``np.linalg.solve(A.as_matrix(), ·)`` alone (which would be self-referential).
The teeth are a **14-mutation bank** verified under ``-O`` (2026-07-02): each
mutation reddens a *named* gate — the ``as_matrix`` transpose / ravel /
size-gate / resolve family, the ``MatrixInverseOperator`` LU-transpose /
seed-consume / forward-``as_matrix`` / non-square-guard / kwarg-drop /
structural-guard-added family, the two ``OperatorProduct``-closure mutations,
and the homogeneous-relocation divergence — beside the designed-green
controls (the at-threshold materialization, the positive constructor, the
ignored seed). The pyright ratchet held exactly at 148; the homogeneous
:math:`k_\infty` / flux stayed byte-identical; the Sphinx build stayed
``-W`` clean.


.. _operator-algebra-assembly-axis:

The assembly axis — structural sparse emission (stencil-assembly 2b)
====================================================================

Step 5 established ``as_matrix`` as the functor **out** of the operator
category, :math:`\mathrm{Op}\to\mathrm{Mat}`, realised by apply-to-basis
probing (:eq:`matrix-functor-out`), and deferred a *sparse* override with
its consumer. Stencil-assembly 2b lands that override — as a full
**assembly axis** parallel to the inverse and adjoint axes. The
per-method cell mathematics (the SN closure walk, the diffusion FD
stencil) is developed in :doc:`/theory/methods/sn/loss_representation`
(:ref:`loss-rep-three-modes`) and :doc:`/theory/methods/diffusion_1d`; this section is
the *operator-algebra* view: how emission threads through the composers,
how ``as_matrix`` delegates, and why the retained probing pathway is
kept as a permanent oracle.

Two realizations of the Mat-functor
-----------------------------------

The Mat-functor now has **two** realizations, and the split is the same
total-versus-partial distinction that separates ``as_matrix`` (a total
functor) from ``inverse()`` (partial — only where an inverse exists;
:ref:`capability-set-semantics`):

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Realization
     - Domain
     - Cost
   * - **apply-to-basis probing** (``_as_matrix_by_probing``)
     - **total** — every finite-dimensional operator has a matrix
     - dense, :math:`n` operator applications
   * - **structural emission** (:meth:`~orpheus.numerics.operator.SupportsAssembly.assemble`)
     - **partial** — only where a leaf declares a stencil
     - sparse, :math:`O(\mathrm{nnz})` scatter

Probing is the honest fallback for any operator (it reads only
``apply``); emission is the efficient path for a structured operator
that knows its own :math:`(\text{row},\text{col},\text{value})`
footprint. The emitted matrix lands in
:class:`~orpheus.numerics.assembled_operator.SparseAssembledOperator` — a
thin :mod:`scipy.sparse` wrapper conforming to
:class:`~orpheus.numerics.operator.LinearOperator`, **not** a new
COO-builder type with its own algebra (that would twin the operator
algebra one layer down — every law restated on triplet buffers). scipy's
own ``COO → CSR`` conversion performs the FEM duplicate-summing, so an
emitter may scatter per-cell / per-face contributions freely and the
carrier assembles them. The functor is **closed** on that carrier: an
assembled operator's ``assemble()`` is itself.

The three-layer assembly surface
--------------------------------

Assembly is the third **per-axis three-layer surface**, minted exactly
like the inverse and adjoint axes (:ref:`capability-set-semantics`):

#. a **predicate** —
   :attr:`~orpheus.numerics.operator.LinearOperator.is_assemblable`
   (base default ``False``; the runtime, instance-accurate truth,
   recursive on composites) — the successor idea to the retired
   capability tags, reading structure not a registry;
#. a **narrowing target** —
   :class:`~orpheus.numerics.operator.SupportsAssembly` (a Protocol
   *extending* ``LinearOperator``, declaring ``assemble() ->
   SparseAssembledOperator``), deliberately **not** ``runtime_checkable``
   (an ``isinstance`` would match every ``OperatorSum``, which defines
   ``assemble`` class-uniformly even when a summand cannot emit);
#. a **checked bridge** —
   :func:`~orpheus.numerics.operator.assemblable` (a PEP-647
   ``TypeGuard``): the one construct that turns the runtime predicate
   into the static permission to call ``assemble()``.

The refusal is the assembly-axis sibling of
:class:`~orpheus.numerics.operator.NotInvertible` (inverse axis) and
:class:`~orpheus.numerics.operator.MissingAdjoint` (adjoint axis):
:class:`~orpheus.numerics.operator.MissingAssembly`, a ``TypeError``
raised **eagerly** by the composer ``assemble()`` bodies when an operand
cannot emit. An operator without a stencil simply does **not** declare
``assemble`` (misuse is a *static* error, method absence over an
advertising flag), and the probing ``as_matrix`` remains its total
Mat-functor. A space-anonymous leaf (a bare-ndarray multiplier / iso
operator with no composite layout) reports ``is_assemblable = False``
honestly — there is no global DOF numbering to emit into.

The composer homomorphism laws
------------------------------

Emission is an **additive-monoidal functor**, so a composite assembles
by recursion through its operands' emissions — no re-walk of the
stencils. Each composer carries one homomorphism law in its
``assemble()`` body, and the matching ``is_assemblable`` predicate is the
conjunction over its legs:

.. math::
   :label: matrix-functor-homomorphism

   [A+B] = [A] + [B], \qquad
   [A\,B] = [A]\,[B], \qquad
   [\alpha L] = \alpha\,[L],

realised respectively by the carrier's own CSR **addition**
(:class:`~orpheus.numerics.operator.OperatorSum`), CSR **matmul**
(:class:`~orpheus.numerics.operator.OperatorProduct`), and scalar
**multiply** (:class:`~orpheus.numerics.operator.ScaledOperator`). A sum
or product is assemblable iff **both** legs are; a scaled operator iff
its operand is; and the eager :class:`~orpheus.numerics.operator.MissingAssembly`
guard-narrows the legs (Design C) before any operand call. The
:class:`~orpheus.numerics.operator.TensorProductOperator` law would be
:math:`[A\otimes B] = [A]\otimes[B]` (a :func:`scipy.sparse.kron`), but
it is **deferred with no consumer** — the diffusion loss and the SN
:math:`L(+C)` trees contain no tensor-product leaf, so building it now
would be a primitive with no product (Cardinal Rule 2).

R2 — ``as_matrix`` delegates to densified assembly
--------------------------------------------------

The ruling **R2** wires the two realizations together without changing
``as_matrix``'s contract. ``as_matrix`` keeps its dense semantics — same
basis-shape resolution, same :class:`~orpheus.numerics.operator.MatrixTooLarge`
size gate — and then, when the operator is
:func:`~orpheus.numerics.operator.assemblable`, **delegates** the
densification to ``self.assemble().as_matrix(...)`` (the assembled
matrix's own ``.toarray()``, with the column dimension checked against
the resolved basis shape) instead of running :math:`n` probing
applications; otherwise it falls back to the probing loop. Because **no
operator was assemblable until an emitter landed**, the delegation is a
no-op for every pre-2b call site — bit-safety by construction. The
:class:`~orpheus.numerics.flat_operator.FlattenedOperator` is transparent
to the axis: its ``is_assemblable`` / ``assemble()`` are its inner
operator's, since the typed operator's emission is already in the flat
layout.

The anti-tautology discipline
-----------------------------

Once ``as_matrix`` delegates to assembly, a naive
"``as_matrix`` :math:`\equiv` ``assemble().to_dense()``" gate compares
assembly **with itself** — vacuous. The defence is a retained,
**separately named** pathway: the probing loop lives on as
``_as_matrix_by_probing`` (not inlined in ``as_matrix``), precisely so a
gate can *force* the probing realization on an assemblable operator and
compare it against the structural emission. This is the
fuller-view-oracle discipline (the same reason the rolling-window sweep
keeps its full-field oracle): the retained relinquished view is the
verification pathway that pins the optimized path, and **an assembly bug
must never be able to hide inside its own densification**.

The G3 gate is exactly **one permanent pin per family** — the diffusion
loss (``test_g3_probed_equals_assembled_pin``) and one DD slab block
(``test_g3_dd_slab_probed_column_pin``) — forcing
``_as_matrix_by_probing`` against ``assemble().as_matrix()``. One pin per
family suffices: the equivalence is a structural property of the
delegation, not a per-instance numerical coincidence.

The first production consumer — the diffusion resolvent
-------------------------------------------------------

The assembly axis is not a primitive-without-a-product: its first
production consumer is the **diffusion resolvent**. The exact inner
solve is ``MatrixInverseOperator(FlattenedOperator(A, template))`` — one
eager LU at construction, one back-substitution per outer iteration
(:doc:`/theory/methods/diffusion_1d`). Because every diffusion loss leaf now emits,
``A.as_matrix()`` routes through the R2 delegation, so
:class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
LU-factors the **assembled** matrix automatically — no consumer-side
change, and the whole existing diffusion suite (its k / trace / balance
gates) becomes the assembled path's regression net.

That "automatically" is the exact place a coverage gate can go vacuous
(vv-principles **Mode 11** — a green twin that never executes the
rewired line), so it is pinned by a **sentinel** rather than trusted:
``test_resolvent_materializes_through_assembly`` monkeypatches
``_as_matrix_by_probing`` to a counter and asserts that constructing the
resolvent fires **no probe at the composite flat dimension** — the
delegation to assembly genuinely executed. (Probes at the tiny law
dimension :math:`n_g` **are** expected and allowed: the boundary emitter
extracts each realized law's :math:`n_g\times n_g` block *through* the
law's own ``apply`` — that in-emitter probing is the one-source
discipline, not a delegation escape.) A green equivalence gate proves the
values; only the sentinel proves the consumer is on the new path. The
diffusion probed-versus-assembled densification measured a max
:math:`|\Delta| = 0.0` (bit-identical) on the heterogeneous fixture.

