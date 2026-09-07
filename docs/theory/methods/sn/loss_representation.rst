.. _loss-representations:

==================================================================
Selectable Representations of the S\ :sub:`N` Loss Operator
==================================================================

This is the **capstone architecture page** for the within-group
S\ :sub:`N` loss operator :math:`(L+C)` — the streaming-plus-collision
operator whose inverse *is* the transport :term:`sweep`. It documents the
final state of issue #222 (the *sweep-strategy carve* + the S6
operator/representation re-layering, complete 2026-06-11 at the
Fork-B2 default flip): one lower-triangular operator, several
*algorithms* that realise it, a single source of truth for which
algorithm a mesh gets, and the L21 theorem that makes "matvec is the
same operator as the sweep" a type fact rather than a coincidence.

It deliberately does **not** re-derive the cell-level mathematics.
Those live in the S\ :sub:`N` book chapters:

* the WDD cell balance and closure — :eq:`dd-cartesian-1d`,
  :eq:`dd-cartesian-2d`, :eq:`dd-2d-balance-form`;
* the 1-D cumprod recurrence — :ref:`sweep-cumprod`;
* the 2-D anti-diagonal wavefront — :ref:`sweep-wavefront`;
* the three-layer **walk / level-op / kernel** stack — the
  :ref:`sweep-dispatch-relayering` section (S6.4(e)).

…and the operator algebra lives in :doc:`/theory/foundations/operator_algebra`:

* the typed forward/solve/adjoint actions —
  :eq:`operator-apply`, :eq:`operator-solve`,
  :eq:`operator-apply-transpose`;
* the interior face-flux cochain :math:`C^1_{\rm int}` and its
  succession after the typed ``WavefrontFlux`` retirement —
  :ref:`wavefront-flux-cochain`.

This page is the layer *above* both: the **representation** layer that
selects and unifies them.

.. contents:: On this page
   :local:
   :depth: 2


Key Facts
=========

.. admonition:: Key Facts — the loss-representation architecture
   :class: important

   * **One operator, several schedules.** :math:`(L+C)` is
     **lower-triangular** under the upwind (sweep) cell ordering. Its
     two actions are one object: :math:`SOLVE = (L+C)^{-1}q` is forward
     substitution (the *transport sweep*); :math:`APPLY = (L+C)\psi`
     is the row action (the *Krylov matvec*). A
     :class:`~orpheus.sn.loss_representation.LossRepresentation` is a
     **schedule** for traversing that triangular structure, not a
     different operator.

   * **A third consumption mode — ASSEMBLE** (stencil-assembly 2b,
     :ref:`loss-rep-three-modes`). Beside *solve* (sweep) and *apply*
     (matvec), the same per-cell closure coefficients can be emitted as
     ``(row, col, value)`` triplets and scattered into a global
     :mod:`scipy.sparse` matrix —
     :func:`~orpheus.sn.loss_representation.assembly.assemble_ordinate_blocks`.
     The assembler carries **no stencil spelling of its own**: it reads
     every coefficient by unit probes of the production matvec kernel
     (:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.residual_kernel_batch`,
     exact because the closure is ``is_linear``), so *solve*, *apply*
     and *assemble* share ONE coefficient source by construction. The
     assembled block is **lower-triangular in walk order** — the
     object-level statement that *the sweep IS forward substitution*
     (the sweep-inverse-contract discharge, #284). The bulk assembler
     is Cartesian-only; the
     curvilinear pole-seed back edge is gone (#282 landed) and the
     augmented walk order is certified triangular by matvec probe.

   * **The four representations**
     (:mod:`orpheus.sn.loss_representation`), each a stateless frozen
     ``@dataclass`` carrying only the mesh:

     - :class:`~orpheus.sn.loss_representation.CumprodScan` — the 1-D
       Blelloch parallel-prefix scan (slab + sphere + cylinder via one
       body); the **1-D production default**.
     - :class:`~orpheus.sn.loss_representation.ScanMarch` —
       :math:`\mathrm{scan}(x)\circ\mathrm{march}(y)`; the **multi-D
       Cartesian production default since the S6.9 Fork-B2 flip**.
     - :class:`~orpheus.sn.loss_representation.MovingFrontierWindow` —
       the anti-diagonal wavefront carrying a rolling
       :math:`(d{-}1)`-frontier; a **selectable peer** (the d=2 default
       through S6.9).
     - :class:`~orpheus.sn.loss_representation.FullFieldWavefront` —
       the same DAG schedule retaining the **whole** interior cochain;
       the **verification oracle**, explicit-select only.

   * **Selection is a single source of truth.**
     :meth:`~orpheus.sn.loss_representation.LossRepresentation.supports`
     returns :class:`~orpheus.sn.loss_representation.Compatibility`
     ``(ok, reason)``; :func:`~orpheus.sn.loss_representation.default_for`
     returns the first match in the ordered registry
     :data:`~orpheus.sn.loss_representation.LOSS_REPRESENTATIONS`.
     Illegal ``(representation, mesh)`` pairings are unrepresentable —
     ``__post_init__`` re-checks ``supports`` and raises
     :class:`~orpheus.sn.loss_representation.IncompatibleRepresentation`.

   * **L21 — one walk, one instance.** A single d-generic
     ``_OctantWalk._interior_walk`` frame serves sweep AND matvec for
     every multi-D representation, forked only by a **kernel object** +
     **emit policy** (never a boolean). The operator holds ONE
     representation instance consumed by ``apply``, ``solve``, and the
     Gauss–Seidel resolvent. Both are *type facts*, pinned by spy tests.

   * **The orientation axis — two frames complete the 2×2** (#280,
     :ref:`loss-rep-orientation-two-frames`). The four faces
     ``{forward, transpose} × {solve, apply}`` unify into TWO frames,
     each shared across **orientation** (the free axis): the
     *apply-loop* frame ``{loss_action, loss_action_transpose}`` and the
     *solve-scan* frame ``{sweep, sweep_transpose}`` (the reverse-scan
     :math:`(L+C)^{-\mathsf T}`, the previously-empty cell). Execution
     strategy ``{scan, cell-loop, wavefront-graph}`` is a **non-free
     third axis** pinned by ``(kernel, dim)`` — the coherence axis is
     ORIENTATION, not kernel. The transpose is the **discrete Euclidean
     transpose** (reversed DAG + face in↔out swap), NOT
     :math:`\mu`-reversal and NOT the continuous adjoint; the physical
     ``.H`` rides on top via the metrics. The **swap law**
     ``A.H.inverse() ≡ A.inverse().H`` (:eq:`loss-rep-adjoint-inverse-swap`)
     then holds by object identity, so the metric adjoint-solve is free
     — no metric code in the sweep.

   * **The Fork-B2 decision (the WHY of the default).** ScanMarch is
     **0.55–0.84×** the window's per-sweep time at **identical** peak
     memory; the window's memory claim only ever held against the
     full-field oracle (1.3–1.4× both). The window was **kept**, not
     retired — a genuinely different schedule over the same operator is
     the point of selectability.

   * **Bit-identity vs principled-equivalence.** Different *schedules*
     are NOT bit-comparable (FP-association differs by construction) →
     ``nulp`` / solver-tol Mode-9 gates. Different *storage policies* of
     the same schedule ARE bit-identical → ``array_equal`` oracles. The
     cell kernel's explicit left fold ``((Σ_t + s_0) + s_1)`` is the FP
     reduction tree of record (sha256-source-pinned).

   * **Gotcha —** ``loss_action`` **returns the full loss, and** :math:`L`
     **is it at** :math:`\sigma = 0`. ``loss_action`` MUST return the
     **full** :math:`(L+C)\psi` for the :math:`\sigma` it is handed, NOT
     bare :math:`L\psi`. The two operator doors are two readings of that
     one action: the composite passes its own :math:`\sigma` and returns
     the result directly, while :math:`L` passes :math:`\sigma = 0` (via
     ``streaming_action``). Resolution A :math:`L = (L+C) - C` is the
     identity behind that, but since #257 S8b **nothing evaluates its
     subtractive form** — :math:`L` reads no :math:`\sigma` at all
     (:eq:`loss-rep-resolution-a`). A leaf that ignored :math:`\sigma`
     and returned bare :math:`L\psi` would drop the collision diagonal
     out of the composite's matvec while ``solve`` kept it.


.. _loss-rep-native-frame:

The native mathematical frame: a lower-triangular operator
==========================================================

The within-group discrete-ordinates balance, for one :term:`ordinate`
:math:`\Omega_n` and one cell, is the WDD relation derived in
:doc:`/theory/methods/sn/cartesian_multid` (:eq:`dd-cartesian-2d`). Collect every cell
and every ordinate into the within-group operator

.. math::
   :label: loss-rep-LpC

   (L+C)\,\psi \;=\; q,
   \qquad
   L \;=\; \Omega\cdot\nabla\big|_{\rm WDD},
   \quad
   C \;=\; \sigma_t\,\odot,

where :math:`L` is the discretised streaming operator (the upwind
WDD difference relations) and :math:`C` is the collision diagonal
(multiply-by-:math:`\sigma_t`). The decisive structural fact:

.. admonition:: :math:`(L+C)` is lower-triangular under the sweep ordering
   :class: note

   Order the unknowns cell-by-cell in **upwind (inflow-to-outflow)
   order** for a given ordinate. Then cell :math:`i`'s balance depends
   only on faces that are *already known* — the domain inflow, or the
   outflow of strictly-upstream cells. In that ordering the matrix of
   :math:`(L+C)` is **lower-triangular** (block-lower-triangular over
   groups). A lower-triangular system has **three** canonical
   consumptions, and they are the same operator viewed three ways
   (:ref:`loss-rep-three-modes` develops the third):

   .. list-table::
      :header-rows: 1
      :widths: 14 34 26 26

      * - Consumption
        - Linear-algebra name
        - S\ :sub:`N` name
        - Code entry
      * - :math:`SOLVE`
        - forward substitution :math:`(L+C)^{-1}q`
        - the **transport sweep**
        - ``LossRepresentation.sweep``
      * - :math:`APPLY`
        - the row action :math:`(L+C)\psi`
        - the **Krylov matvec**
        - ``LossRepresentation.loss_action``
      * - :math:`ASSEMBLE`
        - materialize :math:`[L+C]` as ``(row, col, value)`` triplets
        - the **structural sparse emission**
        - ``assemble_ordinate_blocks``

This is the **L21 invariant** referenced throughout the SN code: *the
sweep and the matvec are two actions of the SAME operator* — they are
not independent code paths that happen to agree, and they must never be
allowed to drift. Forward substitution and the row action share the
triangular factor; they differ only in which is the unknown.

.. implements:: loss-rep-LpC
   :by: orpheus.sn.operators.streaming.StreamingCollisionOperator

   The object that **is** :math:`(L+C)`. It is an
   :class:`~orpheus.numerics.operator.OperatorSum` of the
   :class:`~orpheus.sn.operators.streaming.StreamingOperator` leaf and the
   collision multiplier :math:`C = M[\sigma]`
   (a :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`),
   so the equation's :math:`L + C` decomposition is this class's
   *construction signature*, not a description bolted onto it. Its
   :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply`
   is the equation's left-hand side and its
   :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve`
   is the system's solution — the same triangular factor read the two ways
   the table above names.

.. implements:: loss-rep-LpC
   :by: orpheus.sn.operators.streaming.StreamingOperator

   The discretised streaming half :math:`L = \Omega\cdot\nabla|_{\rm WDD}`
   (plus, in curvilinear geometry, the angular redistribution) as its own
   leaf. It is declared separately rather than treated as an internal
   detail of the composite because the algebra builds :math:`(L+C)`
   **from** it — ``L + C`` dispatches one-directionally on the streaming
   operator — and because :math:`L` is the operand every other
   within-group composite (:math:`L + C - S`,
   :math:`L + C - S - N_{2n} - B`) is
   assembled from. Declaring only the composite would leave the
   equation's own left-hand factor unlinked.

Resolution A — one action, two readings of σ
--------------------------------------------

The representation returns the **full** within-group loss action
:math:`(L+C)\psi` for whatever collision diagonal :math:`\sigma` its
caller hands it. The streaming leaf :math:`L` and the composite
:math:`(L+C)` are then two *readings* of that one action, related by the
algebra glue

.. math::
   :label: loss-rep-resolution-a

   L\,\psi \;=\; (L+C)\,\psi \;-\; \sigma_t \odot \psi.\mathrm{bulk}
   \qquad\text{(Resolution A: } L = (L+C) - C\text{)}.

.. note::

   **How the glue is realised today — the** :math:`\sigma = 0` **reading
   (#257 S8b).** Resolution A is an *identity*: it is
   :eq:`loss-rep-affine` read at two values of :math:`\sigma`, and it is
   true. But since #257 S8b **no shipped code evaluates its right-hand
   side.**
   :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply` does
   not compute :math:`(L+C)\psi` and subtract :math:`\sigma_t\odot\psi`;
   it calls the representation's
   :meth:`~orpheus.sn.loss_representation.LossRepresentation.streaming_action`,
   whose whole body is ``loss_action(0, ψ)`` — the SAME walk evaluated at
   :math:`\sigma = 0`, so :math:`L` reads no :math:`\sigma` at all
   (``coding-elegance`` Pattern 4: a :math:`\sigma` parameter on
   :math:`L` would be one the leaf never reads).

   The subtraction is genuine history. Until #257 S8b :math:`L`'s matvec
   *was* a literal :math:`-\sigma_t\psi`, and before the S6.3
   re-layering that subtraction was duplicated five times across five
   private ``_apply_*`` bodies. Collapsing it first to one site and then
   to :math:`\sigma = 0` is the same single-source-of-truth move made
   twice — the second time hard enough that the twin disappeared instead
   of being deduplicated.

Either way the identity imposes the same **load-bearing contract** on
every representation:

.. warning::

   ``loss_action`` MUST return the FULL :math:`(L+C)\psi` **for the**
   :math:`\sigma` **it is given**, not bare :math:`L\psi`.

   The *consequence* of breaking it moved with the realisation, so state
   it in current terms:
   :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply`
   passes the composite's own diagonal and returns the result
   **directly**, so a leaf that ignored :math:`\sigma` and returned bare
   :math:`L\psi` would drop the collision diagonal out of the Krylov
   matvec entirely while
   :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve`
   kept it — a matvec and a sweep that are no longer the same operator,
   i.e. the L21 invariant broken (vv-principles failure Mode 3,
   *missing/duplicated factor*). Under the pre-#257 subtractive
   realisation the same mistake produced the mirror-image defect, a
   :math:`C` counted *twice*.

   The convention is pinned by ``tests/sn/operators/test_loss_action_convention.py``: the
   non-tautological anchor checks that for a flat reflective field
   :math:`L\psi_{\rm flat} = 0`, so :math:`(L+C)\psi = \sigma_t\psi` —
   proving the action is the *full* :math:`(L+C)` loss, not bare
   :math:`L` — and its sibling cross-checks the :math:`+C` composition
   against an independent collision multiplier :math:`C = M[\sigma_t]`
   (a :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`),
   asserting ``(L + C).apply(ψ)`` is byte-identical to
   ``loss_action(σ_t, ψ)`` and that ``L.apply(ψ)`` recovers
   ``loss_action(σ_t, ψ) − C·ψ`` to FP ULP — which is the affine relation
   :eq:`loss-rep-affine`, not a check that ``apply`` performs a
   subtraction.

.. implements:: loss-rep-resolution-a
   :by: orpheus.sn.loss_representation._LossRepresentation.streaming_action

   The one symbol whose *value* is :math:`L\psi` derived from the full
   loss. Its entire body is ``self.loss_action(self._zero_sigma_for(psi),
   psi)`` — the identity realised as the :math:`\sigma = 0` reading rather
   than as a subtraction, which is why it lives on the base class and is
   inherited by every representation instead of being re-spelled per
   schedule. It is the single source of the :math:`\sigma`-free streaming
   action for the whole package.

.. implements:: loss-rep-resolution-a
   :by: orpheus.sn.operators.streaming.StreamingOperator.apply

   The operator-level door returning :math:`L\psi`. It is the *public*
   face of the identity — a caller who asks the algebra for :math:`L`
   gets exactly this — and it delegates in one line to
   :meth:`~orpheus.sn.loss_representation.LossRepresentation.streaming_action`
   above, so the equation is realised once and exposed once.

.. implements:: loss-rep-resolution-a
   :by: orpheus.sn.operators.streaming.StreamingCollisionOperator.apply

   This label is **overloaded**, deliberately: since #240 Phase 2 Step B
   it also carries the *composite-owns-its-matvec* claim
   (:ref:`loss-rep-removal-form-matvec`), which is the same Resolution-A
   glue read from the other end — one ``loss_action``, one source of
   :math:`\sigma`. This override is the forward half: it passes the
   composite's own ``self.sigma`` — the SAME array
   :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve`
   threads into the WDD sweep — and returns
   ``loss_action(self.sigma, psi)`` directly, in place of the inherited
   :class:`~orpheus.numerics.operator.OperatorSum` leaf sum. It is what
   ``tests/sn/operators/test_removal_form_matvec_sweep.py`` verifies under
   this label.

.. implements:: loss-rep-resolution-a
   :by: orpheus.sn.operators.streaming.StreamingCollisionOperator.apply_transpose

   The transpose half of the same overloaded claim: ``loss_action_transpose(self.sigma, phi)``,
   one :math:`\sigma` source, both orientations. Declared alongside its
   forward sibling because a single-source claim that held in only one
   direction would not be one.

.. _loss-rep-removal-form-matvec:

The composite owns its matvec — the removal form :math:`C(\sigma_r)`
------------------------------------------------------------------------

So far the matvec :math:`(L+C)\psi` has been described as the matvec twin of
the sweep, applied by :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply`
(:eq:`loss-rep-resolution-a`). But the *composite* operator
:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` — the :math:`A = (L+C)`
object whose :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve` *is* the
sweep — has its own matvec too, and getting that matvec to single-source its
diagonal :math:`\sigma` from the *same* place :meth:`solve` does is the
substance of issue #240 Phase 2 Step B. This subsection records that carve: a
**principled-equivalence** refactor (not a bug fix — no wrong value ever
shipped) that turns a value-correct-by-coincidence leaf sum into a
correct-by-construction action, and in doing so foreclosed a latent trap that
the **removal form** :math:`\mathrm{StreamingCollisionOperator}(L(\sigma_t),
C(\sigma_r))` would have made observable.

The :math:`\sigma` parameter is now explicit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The protocol signature of the two matvec actions changed: both
:meth:`~orpheus.sn.loss_representation.LossRepresentation.loss_action` and
:meth:`~orpheus.sn.loss_representation.LossRepresentation.loss_action_transpose`
now take the collision diagonal :math:`\sigma` **explicitly** as their first
argument (``loss_action(sigma, psi)``), exactly symmetric with the sweep door
``sweep(Q, sig_t, ...)``. Before the carve the representation read
``operator.sigma_t`` *off an operator handle* it was passed — so the leaf, not
its caller, decided which :math:`\sigma` the matvec realised. After the carve
the **caller single-sources** :math:`\sigma`:

* :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply` passes
  :math:`\sigma = 0` and so receives bare :math:`L\psi` — the
  :math:`\sigma`-free reading of Resolution A
  (:eq:`loss-rep-resolution-a`). ⛔ At the #240 Step B carve described here it
  still passed its own ``sigma_t`` and subtracted it back; #257 S8b then
  removed the subtraction entirely, and :math:`L` reads no :math:`\sigma`
  today;
* the **new** :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply` /
  :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply_transpose` overrides pass
  the composite's *own* diagonal ``self.sigma`` — the SAME array
  :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve` threads into the WDD
  sweep — and realise :math:`M(\sigma)\psi` *directly*, via
  ``self.loss_representation.loss_action(self.sigma, psi)``, instead of inheriting
  the :meth:`~orpheus.numerics.operator.OperatorSum.apply` leaf sum.

This is the **one-instance theorem** (:ref:`loss-rep-one-walk-one-instance`)
extended to the diagonal: the operator already held ONE representation instance
shared by :meth:`apply` and :meth:`solve`; now it also holds ONE
:math:`\sigma`, supplied by the same operand whichever door is exercised. The
removal form is what makes that single-sourcing *matter*.

The affine-in-:math:`\sigma` structure of the forward matvec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The inherited :meth:`~orpheus.numerics.operator.OperatorSum.apply` is the
literal leaf sum :math:`L.\mathrm{apply}(\psi) + C.\mathrm{apply}(\psi)`. The
crux of the carve is that this leaf sum is value-correct **only by
coincidence** — a coincidence that follows from a structural property of the
*forward* (apply) direction that does **not** hold in the sweep direction.

Start from the WDD cell balance derived in :doc:`/theory/methods/sn/index`. In the
forward (apply) direction the cell average :math:`\bar\psi` is **known**: it is
the input. The matvec is the *row action* — it reconstructs the residual that
the balance would have to absorb,

.. math::
   :label: loss-rep-affine-cell

   M(\sigma)\,\psi\big|_{\rm cell}
   \;=\;
   \mathrm{denom}\cdot\bar\psi \;-\; \mathrm{numer}_{\rm upstream},
   \qquad
   \mathrm{denom} \;=\; \underbrace{\textstyle\sum_a c_a}_{\text{streaming}}
                        \;+\; \sigma,

.. (vv-status rationale) Derivation step in the affine-in-sigma argument.
   Its terminal removal-form matvec is foundation-gated
   (test_removal_form_matvec_sweep, verifies loss-rep-resolution-a).
.. vv-status: loss-rep-affine-cell documented

.. implements:: loss-rep-affine-cell
   :by: orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch

   The Cartesian cell residual, literally: ``residual = denom * psi_bar -
   numer``, with ``denom`` and the per-axis couplings :math:`c_a`
   returned by the shared
   :meth:`~orpheus.transport.spatial.diamond.DiamondDifference._cartesian_streaming_diagonal`
   as :math:`\Sigma_t + \sum_a 2 g_a`. The equation's split of
   ``denom`` into a streaming sum plus :math:`\sigma` is the code's own
   explicit left fold, in that order.

.. implements:: loss-rep-affine-cell
   :by: orpheus.sn.loss_representation._OneDimScanWalk._apply_walk

   The curvilinear arm of the same cell relation, which does **not** ride
   the batched kernel: it computes ``m_full = (denom * psi_cell -
   numer_upstream) / V[i]`` through the ``cell_balance_for_streaming``
   density path, because the Morel–Montry angular redistribution threads
   in-sweep and is not expressible as a per-axis coefficient triple.
   Curvilinear S\ :sub:`N` is DD-only, so this is a single-occupant
   geometry rather than a polymorphism gap — and it is the reason
   :eq:`loss-rep-affine-cell` needs two implementers, not one.

with :math:`c_a = 2|\mu_a|/\Delta a` the per-axis scheme-scaled streaming
coupling (:eq:`loss-rep-scanmarch-solve`; the diamond :math:`2` is the
scheme's). Because :math:`\bar\psi` is known,
:math:`\sigma` enters :eq:`loss-rep-affine-cell` only through the *additive*
term :math:`\mathrm{denom}\cdot\bar\psi = (\sum_a c_a)\bar\psi +
\sigma\bar\psi`. Collecting the :math:`\sigma`-independent part into a
``streaming_action``,

.. math::
   :label: loss-rep-affine

   M(\sigma)\,\psi
   \;=\;
   \mathrm{streaming\_action}(\psi) \;+\; \sigma\cdot\psi
   \qquad\text{(forward matvec is AFFINE in }\sigma\text{).}

.. (vv-status rationale) Derivation identity (the forward matvec is affine
   in sigma). Foundation-gated via loss-rep-resolution-a
   (test_removal_form_matvec_sweep).
.. vv-status: loss-rep-affine documented

.. implements:: loss-rep-affine
   :by: orpheus.sn.loss_representation._LossRepresentation.streaming_action

   The equation *names* ``streaming_action``, and the method exists
   **because** of this identity. Affinity in :math:`\sigma` is what makes
   ``loss_action(0, ψ)`` a legitimate spelling of the
   :math:`\sigma`-independent part: if the forward matvec were rational
   in :math:`\sigma` — as the sweep direction is — evaluating the walk at
   zero would give something other than :math:`L\psi` and this method
   could not exist.

.. implements:: loss-rep-affine
   :by: orpheus.sn.loss_representation._LossRepresentation.streaming_action_transpose

   The transpose sibling, ``loss_action_transpose(0, φ)``, resting on the
   same identity at :math:`\sigma = 0`: since :math:`C = \sigma\odot` is a
   self-adjoint diagonal, :math:`M(\sigma)^{\mathsf T} = L^{\mathsf T} +
   \sigma\odot` is affine in :math:`\sigma` too.

This is the decisive fact. **The matvec is affine in** :math:`\sigma`: a clean
additive :math:`+\,\sigma\cdot\psi`, never a :math:`1/\mathrm{denom}`.

Contrast the **sweep** direction, where :math:`\bar\psi` is the *unknown*. The
cell-average solve is :math:`\bar\psi = \mathrm{numer}/\mathrm{denom}` — and
now :math:`\sigma` sits in the **denominator** (:math:`\mathrm{denom} = \sum_a
c_a + \sigma`), so :math:`\bar\psi` is a *rational* function of :math:`\sigma`,
**non-affine**. The whole non-affinity of the loss operator in :math:`\sigma`
lives in the sweep direction; the apply direction is affine because it never
inverts the denominator. (This asymmetry is why the round-trip
:math:`\mathrm{apply}\circ\mathrm{solve}` cannot detect a :math:`\sigma`-routing
error in ``apply`` alone — see the verification subsection below.)

The affine structure :eq:`loss-rep-affine` is precisely what made the
inherited leaf sum value-correct. Each leaf read its **own** diagonal: as
:math:`L.\mathrm{apply}` was then realised it returned
:math:`M(\sigma_t)\psi - \sigma_t\cdot\psi = \mathrm{streaming\_action}(\psi)`
— Resolution A subtracting :math:`L`'s own :math:`\sigma_t` — and
:math:`C.\mathrm{apply}` returned :math:`\sigma_r\cdot\psi`
(the collision leaf's own :math:`\sigma_r`). Summing,

.. math::
   :label: loss-rep-leaf-sum

   \underbrace{L.\mathrm{apply}(\psi)}_{M(\sigma_t)\psi - \sigma_t\psi
                                       \;=\; \mathrm{streaming\_action}(\psi)}
   \;+\;
   \underbrace{C.\mathrm{apply}(\psi)}_{\sigma_r\cdot\psi}
   \;=\;
   \mathrm{streaming\_action}(\psi) + \sigma_r\cdot\psi
   \;=\;
   M(\sigma_r)\,\psi.


.. no-implementation:: loss-rep-leaf-sum
   :kind: identity

   **Nothing implements this**, and nothing can: it is a derivation STEP —
   the algebraic collapse of the leaf sum to :math:`M(\sigma_r)\psi` — not a
   computation any symbol performs. It is also, since the fusion, a route the
   tree does NOT take: :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator`
   overrides ``apply`` and walks once, so no production path ever evaluates
   ``L.apply(psi) + C.apply(psi)``. The identity is what made the historical
   leaf sum value-correct; declaring an implementer for it would name a
   sweep that no longer runs.

.. (vv-status rationale) Derivation step (the leaf sum collapses to
   M(sigma_r)psi). Foundation-gated via loss-rep-resolution-a.
.. vv-status: loss-rep-leaf-sum documented

So the leaf sum **did** compute :math:`M(\sigma_r)\psi`, the right value for
the removal form — but it got there by sourcing :math:`\sigma_t` from
``L.sigma_t`` (the streaming leaf), cancelling it, and then *re-adding*
:math:`\sigma_r` from ``C``. The value was right; the **source was wrong** —
:math:`\sigma` came from a *different operand* than the one
:meth:`solve` uses, and only the affine structure :eq:`loss-rep-affine` kept
that from showing up as a wrong number.

.. note::

   **Two independent reasons this route is now unreachable**, which is
   why :eq:`loss-rep-leaf-sum` is classified a *superseded path* and
   declares no implementer (:ref:`loss-rep-unimplemented-labels`). First,
   the #240 Step B override below means the composite never reaches
   :meth:`~orpheus.numerics.operator.OperatorSum.apply` at all. Second,
   #257 S8b took :math:`\sigma_t` off :math:`L` entirely, so the
   equation's left underbrace — :math:`L.\mathrm{apply}(\psi) =
   M(\sigma_t)\psi - \sigma_t\psi` — no longer describes what
   :meth:`StreamingOperator.apply <orpheus.sn.operators.streaming.StreamingOperator.apply>`
   computes either. The equation remains a true statement of the algebra;
   what is gone is any code that walks it.

The two-source hazard
~~~~~~~~~~~~~~~~~~~~~~~

The latent defect is a **twin-source**. The composite's two doors source the
diagonal :math:`\sigma` from two *different* operands:

.. list-table:: Where the two doors source :math:`\sigma` (pre-carve)
   :header-rows: 1
   :widths: 24 38 38

   * - Door
     - Sources :math:`\sigma` from
     - Realises
   * - :meth:`StreamingCollisionOperator.solve`
     - the collision leaf ``C`` (``self.sigma``)
     - :math:`(L+C(\sigma_C))^{-1}q` (the sweep)
   * - inherited ``OperatorSum.apply``
     - the streaming leaf ``L`` (``L.sigma_t``)
     - :math:`\mathrm{streaming\_action} + \sigma_C\cdot\psi`
       via :eq:`loss-rep-leaf-sum`

They **agree** today only because production always builds :math:`L` and
:math:`C` from the *same* :math:`\sigma_t` (``L.sigma_t == C.sigma``). The
moment they differ — the removal form — the two doors realise actions sourced
from different diagonals, held in agreement purely by the affine coincidence
:eq:`loss-rep-affine`. This is the ``coding-elegance`` Smell of a *value
correct by coincidence rather than by construction*: the inherited
``OperatorSum.apply`` is a twin-path realisation of the composite's own
action, re-deriving the streaming part through :math:`\sigma_t` only to cancel
it.

The override (#240 Step B) collapses the twin-source. Both
:meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply` and
:meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply_transpose` now read the
composite's **own** ``self.sigma`` — the SAME array
:meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve` threads into the sweep —
and call ``loss_action(self.sigma, psi)`` directly:

.. code-block:: python

   class StreamingCollisionOperator(OperatorSum):
       def apply(self, psi):                         # (L+C)ψ = M(σ)ψ
           _require_typed_composite("StreamingCollisionOperator.apply", self.sn_mesh, psi)
           return self.loss_representation.loss_action(self.sigma, psi)

       def apply_transpose(self, phi):               # (L+C)ᵀφ = M(σ)ᵀφ
           _require_typed_composite(..., self.sn_mesh, phi)
           return self.loss_representation.loss_action_transpose(self.sigma, phi)

This is ``coding-elegance`` Pattern 2 (single source of truth): **one**
``loss_action``, **one** source of :math:`\sigma`, consumed identically by
both directions. The composite never asks a leaf for a :math:`\sigma`-bearing
action it must then undo. The input contract (typed composite + the
**space-content** invariant — mesh-object identity until campaign 1
CS4b S3) is itself single-sourced through the module-level
``_require_typed_composite`` helper that :meth:`StreamingOperator.apply` now
shares. The multi-D Cartesian *adjoint* rides the same discipline since
#310 C4 (``ScanMarch.loss_action_transpose`` is the row-march reverse — one
``loss_action_transpose``, one :math:`\sigma` source, both directions), so
:meth:`apply_transpose` is correct-by-construction on every registered
representation — and since #310 C5 on every registered *scheme* too (the
moment-tailed LD-2D reverse rides the same wavefront frame); a scheme
that registers no transpose kernel pair still refuses typed and loud,
never a silent wrong answer.

The removal form makes the distinction observable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **removal form** is the within-group regime that makes the two-source
distinction visible. It folds the within-group self-scatter into the
collision diagonal,

.. math::
   :label: loss-rep-removal-sigma

   \sigma_r \;=\; \sigma_t \;-\; \Sigma_{s,0}^{\,g\to g},


.. no-implementation:: loss-rep-removal-sigma
   :kind: definition

   **Nothing implements this** — it is notation. The page's own vv-status
   rationale says so verbatim: *"Notation/definition (the removal
   cross-section sigma_r = sigma_t - Sigma_s0). Definitional, no isolated
   claim."* Code that computes :math:`\sigma_r` obeys the definition; it does
   not implement it, any more than an addition implements the definition of
   ``+``.

.. (vv-status rationale) Notation/definition (the removal cross-section
   sigma_r = sigma_t - Sigma_s0). Definitional, no isolated claim.
.. vv-status: loss-rep-removal-sigma documented

so the composite becomes :math:`\mathrm{StreamingCollisionOperator}(L(\sigma_t),
C(\sigma_r))` with :math:`\sigma_C = \sigma_r \neq \sigma_t` (Adams & Larsen
2002 §III — the within-group iteration framing in which the sweep
:math:`(L+C)^{-1}` *is* the transport operator; #200). On this form the
sweep solves :math:`(L+C(\sigma_r))^{-1}`, and the matvec MUST realise the
matching :math:`M(\sigma_r)\psi` — its inverse twin (L21). With
:math:`\sigma_t \neq \sigma_r` the question "**which** :math:`\sigma` does the
matvec realise?" finally has two different answers, and the override is the one
that single-sources :math:`\sigma_r` from ``C`` exactly as :meth:`solve` does.

There is **no production caller of the removal form yet**: the consumer that
would build :math:`\sigma_r` is the within-group self-scatter fold of issue
#200, which is not wired. The collision multiplier :math:`C = M[\sigma]`
(a :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`)
already accepts either :math:`\sigma_t` or :math:`\sigma_r` (it carries no
interpretation flag — both are :math:`(\mathrm{ng}, \ldots)` arrays applied as
:math:`\sigma\cdot\psi`), and a :math:`\sigma_r`-*sweep* is **not** a correct
within-group accelerator for anisotropic flux (it inverts the
diagonal-in-angle removal :math:`\sigma_r\,I`, not the isotropic-projection
self-scatter :math:`\Sigma_{s,0}P_{\rm iso}` — issue #215, 46–56 % errors on
vacuum/heterogeneous). The carve is therefore *prophylactic*: the override
forecloses the latent twin-source **before** any consumer can trip it. It
would have become a genuine numerical trap only if a future refactor made
``L.apply`` *non-affine* in :math:`\sigma` (e.g. an :math:`L`-leaf that itself
inverts a denominator) — at which point :eq:`loss-rep-leaf-sum` would no
longer collapse to :math:`M(\sigma_r)\psi`. The override removes that coupling
by construction.

Verification — value-preserving re-association, structurally cross-checked
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the carve preserves the matvec *value* and only re-associates the
floating-point reduction, it is verified as a #240-Phase-2-Step-A-shape
change (a value-preserving re-association), against the
:ref:`vv-principles three criteria <loss-rep-bit-vs-principled>`. The gate is
``tests/sn/operators/test_removal_form_matvec_sweep.py``, ``foundation``-tagged
and ``verifies("loss-rep-resolution-a")`` (the carve *sharpens* the existing
Resolution-A equation rather than minting a new label — the composite-owns-its-matvec
convention is the same :eq:`loss-rep-resolution-a` glue, now single-sourcing
:math:`\sigma`). It has four parts:

(a) **The structural teeth** —
    ``test_invertible_apply_is_M_of_C_sigma_bit_identical`` (and its adjoint
    sibling, slab/sphere/cyl) demand
    :math:`(L+C).\mathrm{apply}(\psi) = M(\sigma_r)\psi`
    **bit-identically** (``array_equal``, 0 ULP) against a structurally
    *independent* reference: a SEPARATE :class:`~orpheus.sn.operators.streaming.StreamingOperator`
    whose **own** :math:`\sigma_t` *is* :math:`\sigma_r`, so its
    ``loss_action(σ_r, ψ)`` is unambiguously :math:`M(\sigma_r)\psi` (no
    removal/leak ambiguity — the leaf reads its own diagonal). Under the
    override, ``op.apply`` IS that same ``loss_action`` call on the same walk →
    0 ULP. Under the pre-fix leak it is the leaf sum :eq:`loss-rep-leaf-sum`:
    value-equal but a *different* reduction tree → :math:`\le 2` ULP off
    ``array_equal`` → it fails. **These are the only gates that discriminate
    the carve.** They were ``xfail(strict=True)`` until the override landed;
    the marker is removed and they now pass plainly. The 2-D adjoint row is
    *included* since #310 C4 (the row-march reverse landed — the former
    exclusion's deferral raise is gone).

(b) **The value ground** — the matvec≡sweep round-trip
    (``test_removal_form_matvec_sweep_roundtrip``: slab/cyl/2-D, sphere
    excluded) and the affine-in-:math:`\sigma` value characterisation
    (``test_removal_form_apply_value_equals_M_of_sigma_r``). These PASS under
    BOTH the leak and the override — the leak is value-correct
    (:eq:`loss-rep-affine`), so the round-trip
    :math:`\mathrm{apply}\circ\mathrm{solve}\approx\mathrm{id}` cannot tell
    leaky from override (a vv Mode-9 degeneracy *in the round-trip gate's
    design* — the teeth therefore live in (a), not here). Sphere is excluded
    from the bare round-trip because the curvilinear Morel–Montry sweep reads
    the previous iterate for the Carlson coupled-pole closure (ERR-058
    family), so a single :meth:`solve` is not the one-shot inverse of
    :meth:`apply`; the sphere matvec≡sweep claim rides the teeth gate (a) plus
    the production-:math:`\sigma` fixed-point bridge.

(c) **The production-:math:`\sigma` invariant** —
    ``test_production_sigma_apply_value_preserved`` builds the *production*
    composite (:math:`\sigma_C = \sigma_t`) and an explicit
    ``L.apply + C.apply`` leaf-sum reference, and asserts they agree to
    ``nulp=6``. The override **drops** the redundant
    :math:`(x - \sigma\psi) + \sigma\psi` round-trip the leaf sum carries
    (:math:`L.\mathrm{apply}` returns :math:`\mathrm{loss\_action}(\sigma_t) -
    \sigma_t\psi`, then :math:`C.\mathrm{apply}` adds :math:`\sigma_t\psi`
    back), so the override is the **more accurate** path. The drift is
    FP-non-associativity (reduction-depth :math:`\times` ULP), measured
    :math:`\le 2` ULP on slab/sphere (the 1-D ``_OneDimScanWalk``),
    :math:`\le 4` on cylinder, :math:`\le 5` on 2-D Cartesian (the
    ``_OctantWalk``). The gate's tolerance was itself re-baselined off a
    ``bitexact=True`` spec: that spec came from a probe comparing two
    *override-style* paths (0 ULP), but the real gate compares
    override-vs-leaf-sum, where bit-identity is *not* a property of the pair
    (the override removes the round-trip). All three vv criteria hold — named
    intermediate (``loss_action``), verified against :math:`M(\sigma_r)` by the
    teeth (a), drift bounded by reduction depth.

(d) **The negative control** —
    ``test_removal_form_nonpositive_sigma_r_rejected`` confirms
    :math:`\mathrm{StreamingCollisionOperator}(L, C(\sigma_r\le 0))` raises at
    construction (vv L11: a contract-validation path needs BOTH a positive
    must-not-raise case — the removal cases — AND a negative must-raise case).

The eigenvalue cross-check (the closed-form :math:`\kinf = \nu\Sigma_f /
\Sigma_a` for a homogeneous reflective medium with the within-group
self-scatter folded into :math:`\sigma_r`,
``test_removal_form_kinf_independent_reference_2g``) is the *structurally
independent* reference required by vv criterion 2 for the removal regime; it is
``xfail`` until the #200 solver entry exists. Since 2026-08-10 (issue #340,
step R5) that deferral is a **declarative** ``@pytest.mark.xfail(strict=True)``
over a capability probe rather than an imperative ``pytest.xfail()`` call: the
strict marker turns the day #200's solver entry appears into an ``XPASS``, i.e.
a FAILURE, so the stub cannot outlive its blocker unnoticed. The blast-radius
split is itself
the structural-independence evidence that the apply re-baseline is principled:
APPLY-path snapshots (the cyl/2-D matvec golden) re-associate by :math:`\le 5`
ULP, but SWEEP/SOLVE snapshots (slab/sphere) stay **bit-identical** — they ride
:meth:`solve`, which never touched the override. And the 2-D Krylov converged
flux (which *does* drive the override through GMRES) agrees with the 2-D SI
converged flux (which does **not** use the override) to :math:`3.9\times10^{-12}`
relative — two structurally distinct iteration paths landing on the same fixed
point, the cross-check that the apply-direction re-baseline did not move the
physics.

The two actions bottom out, for every Cartesian representation, in **one
pure kernel pair** on
:class:`~orpheus.transport.spatial.diamond.DiamondDifference`:

* :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch`
  (solve) — :math:`\bar\psi = \mathrm{numer}/\mathrm{denom}`;
* :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch`
  (apply) — :math:`r = \mathrm{denom}\cdot\bar\psi - \mathrm{numer}`,

with the explicit **left fold**
:math:`((\Sigma_t + s_0) + s_1) + \cdots` as the IEEE-754 reduction
tree of record (see :ref:`loss-rep-bit-vs-principled`).


.. _loss-rep-three-modes:

The three consumption modes — solve, apply, assemble
====================================================

The native frame gives one triangular operator :math:`(L+C)` with three
canonical consumptions. *Solve* and *apply* are the two developed by the
four representations below; **assemble** is the third — realised in the
stencil-assembly campaign (Phase 2b) — and it is the sharpest expression
of the L21 invariant, because it emits the *same per-cell coefficients*
in a third layout and lets a structurally-independent linear-algebra
routine consume them:

.. list-table::
   :header-rows: 1
   :widths: 12 46 42

   * - Mode
     - Consumer
     - Walk
   * - :math:`SOLVE`
     - the sweep — forward / triangular substitution
     - cells in walk order
   * - :math:`APPLY`
     - the Krylov matvec — the row action
     - cells in **any** order
   * - :math:`ASSEMBLE`
     - emit the SAME coefficients as ``(row, col, value)`` and scatter
       into a global :mod:`scipy.sparse` matrix
     - cells in walk order, threading the face rows

The assembler is
:func:`~orpheus.sn.loss_representation.assembly.assemble_ordinate_blocks`;
it lands its output in a numerics carrier,
:class:`~orpheus.numerics.assembled_operator.SparseAssembledOperator`,
which is a *serialization* of the operator, **not** a fifth
:class:`~orpheus.sn.loss_representation.LossRepresentation` (the
operator-algebra view of the assemble functor — the three-layer
``assemble()`` surface, the composer homomorphism laws, and the
``as_matrix`` delegation — is documented at
:ref:`operator-algebra-assembly-axis`). The mode is **Cartesian only**
in 2b; the curvilinear obstruction is a walk-order back edge (#282,
below).

The one-source guardrail — assemble reads the coefficients solve/apply read
---------------------------------------------------------------------------

The load-bearing design constraint is the **Phase-F twin-path lesson**:
a second spelling of a discretisation is a latent divergence bug, so
``assemble()`` MUST NOT carry its own copy of the diamond fold, the UBLD
cell system, or the trace reconstruction. It reads **the same
coefficient source** ``apply`` reads — for the diffusion emitters, the
precomputed conductance / closure attributes and the production einsum
kernels; for the SN block, the scheme's own production matvec kernel
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.residual_kernel_batch`
(the identical body ``apply`` runs, fed by the same
:meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.streaming` raw per-axis
data). The consequence is that a sign flip *anywhere* in the shared
kernel algebra moves solve, apply, AND assemble **together** — the
one-source teeth (:ref:`loss-rep-assembly-verification`) bite by
construction, with zero parallel spelling to drift. A leaf whose
``apply`` is monkeypatched to *raise* still assembles: emission and
action share coefficients, not call paths.

Coefficient extraction by kernel probing
----------------------------------------

The SN emitter owns no stencil spelling because it recovers every
coefficient by **unit probes of the production kernel**, exploiting the
scheme's declared linearity (``is_linear``, which DD and LD both
assert). The within-cell residual and the outgoing-face reconstruction
are **affine** in the cell moments :math:`\bar\psi`, the inflow face
traces :math:`\psi^{\rm in}_b`, and the source :math:`\bar Q`:

.. math::
   :label: loss-rep-affine-kernel-maps

   r \;=\; \bar A\,\vec\psi \;-\; \sum_b C_b\,\psi^{\rm in}_b \;-\; \bar Q,
   \qquad
   \psi^{\rm out}_a \;=\; T_a\,\vec\psi \;+\; \sum_b \alpha_{ab}\,\psi^{\rm in}_b .

.. implements:: loss-rep-affine-kernel-maps
   :by: orpheus.sn.loss_representation.assembly._probe_coefficient_blocks

   Returns exactly the equation's four coefficient blocks —
   ``(diag, inflow, trace, memory)`` :math:`= (\bar A, C_b, T_a,
   \alpha_{ab})` — by driving the production matvec kernel with unit
   inputs. Its reading is an algebraic identity rather than a fit
   *only because* the maps are affine, which is what this equation
   asserts; the precondition is enforced at the assembler's entry, which
   raises :class:`~orpheus.numerics.operator.MissingAssembly` for a
   closure that does not declare ``is_linear``.

.. implements:: loss-rep-affine-kernel-maps
   :by: orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch

   The DD instance of the affine maps: scalar faces, so
   :math:`1\times1` blocks. This is the kernel the probes call, so it is
   the object whose affinity the equation is about — declaring only the
   prober would name the reader of the coefficients and not their source.

.. implements:: loss-rep-affine-kernel-maps
   :by: orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous.residual_kernel_batch

   The LD instance: the bilinear UBLD system, so :math:`2^d \times 2^d`
   cell blocks and :math:`2^{d-1}`-moment faces, with the mass-diagonal
   normalisation that puts the residual in raw per-moment units. Both
   closures are declared because :eq:`loss-rep-affine-kernel-maps` is the
   claim that **one** extraction serves every batched-kernel closure —
   a claim with a single implementer would be vacuous.

Because these maps are affine, a **unit** input reads a coefficient
block *exactly* — there is no summation error to accumulate, so the
extraction is not a fit but an algebraic identity:

* a cell probe :math:`\bar\psi = e_m \otimes \mathbf{1}_{\rm cells}` (one
  per cell-moment slot) returns column :math:`m` of the per-cell
  diagonal block :math:`\bar A` (in :math:`r`) and column :math:`m` of
  the outgoing-trace map :math:`T_a` (in :math:`\psi^{\rm out}_a`);
* a face probe :math:`\psi^{\rm in}_b = e_f \otimes \mathbf{1}_{\rm cells}`
  (one per face-moment slot) returns :math:`-` column :math:`f` of the
  inflow coupling :math:`C_b` and column :math:`f` of the face-memory
  map :math:`\alpha_{ab}`.

The **same** extraction serves every batched-kernel closure. DD's scalar
faces give :math:`1\times1` blocks — :math:`\bar A` is the diamond
denominator, :math:`C` the :math:`2|\mu|/\Delta x` coupling,
:math:`T = 1/w = 2`, and :math:`\alpha = -1`, the face **memory** that
makes DD's assembled block carry dense upstream chains. LD's bilinear
faces give the :math:`2^d\times 2^d` UBLD cell system with an **upwind**
outgoing trace, so :math:`\alpha = 0` — no memory, a block tri-diagonal
sparsity. The moment stride is the #253 single-sourced
``cell_moment_count`` / ``face_moment_count`` (DD's ``cm = 1``
degenerates to the plain cell ravel).

The walk-order triangularity theorem
------------------------------------

With the blocks in hand, the emitter walks the octant's
:class:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph`
in the **same topological order** the sweep and matvec use, carrying per
in-flight face its trace **row block** — the face trace's linear map
from the bulk DOFs (a dense :math:`(f_m, n_{\rm dof})` matrix; a boundary
inflow face is the zero block, the zero-inflow posing below). Per cell
:math:`c`:

.. math::
   :label: loss-rep-walk-order-rows

   \operatorname{rows}(c) \;=\; \bar A_c\,E_c \;-\; \sum_b C_{b,c}\,R^{\rm in}_b,
   \qquad
   R^{\rm out}_a \;=\; T_{a,c}\,E_c \;+\; \sum_b \alpha_{ab,c}\,R^{\rm in}_b,

where :math:`E_c` selects cell :math:`c`'s own DOF columns. Each cell's
rows depend only on **already-emitted** upstream face rows, so the
assembled block is **(block-)lower-triangular in walk order** — which is
the object-level content of the theorem *the sweep IS forward
substitution*. This is precisely #284's sweep-inverse question, answered
as a matrix fact: on the source subspace (bulk source, zero inflow trace
— every production right-hand side today) the production sweep and
LAPACK's triangular back-solve are the *same* linear map. The all-zero
degenerate label (a pure-:math:`z` ordinate over a lower-dimensional
mesh — the :math:`Q/\Sigma_t` short-circuit) has no graph and no face
threading; its block is the cell-probe diagonal alone.

.. implements:: loss-rep-walk-order-rows
   :by: orpheus.sn.loss_representation.assembly.assemble_ordinate_blocks

   The walk body *is* the equation. Per cell it forms
   ``cell_rows -= inflow[b] @ row_in`` for each incoming face and
   ``row_out = trace[a] @ E_c + Σ_b memory[a][b] @ row_in`` for each
   outgoing one, where :math:`E_c` is realised as the column slice
   ``[:, base:base + cm]`` into the cell's own DOF block. The in-flight
   ``face_blocks`` dict carries :math:`R^{\rm in}_b`; a face absent from
   it is a boundary inflow, which is the zero block — the zero-inflow
   posing, spelled as an absence rather than as a stored zero.

The two face closures print two triangular shapes: DD's :math:`\alpha=-1`
memory propagates each face row down the whole upstream chain (dense
lower-triangular in 1-D — the honest matrix of the marching
reconstruction; per-axis chains, :math:`O(n_x+n_y)` per row, in 2-D);
LD's :math:`\alpha=0` upwind trace forgets, giving a block tri-diagonal
matrix.

The sweep ⇄ global moment-frame conjugation
-------------------------------------------

A multi-moment closure (LD) produces and consumes its :math:`2^d` moment
vector in the per-ordinate **sweep frame** (each axis oriented so the
downstream face is at local :math:`+1`), but the assembled block must be
the operator over the **global**-frame bulk field. The walks convert
with the single-sourced sign involution
:func:`~orpheus.transport.spatial._ubld.octant_moment_frame_signs`
(:eq:`ld-ubld-octant-moment-frame-signs`), and the assembler applies the
*same* involution as a similarity conjugation of the whole block,

.. math::
   :label: loss-rep-sweep-global-conjugation

   M_{\rm global} \;=\; \Phi\,M_{\rm sweep}\,\Phi,
   \qquad
   \Phi \;=\; \operatorname{diag}\bigl(\text{signs per cell}\bigr),

with :math:`\Phi` the per-DOF sign vector tiled over cells (``None`` for
a single-moment closure, where the sweep and global frames coincide and
the conjugation is the identity). This is the assembly-mode analogue of
the ERR-061 root cause: the slope moment must be lifted to the global
frame *before* it enters a frame-agnostic consumer, whether that
consumer is the scattering source (:eq:`ld-ubld-slope-angular-reduction`)
or, here, the global sparse matrix.

.. implements:: loss-rep-sweep-global-conjugation
   :by: orpheus.sn.loss_representation.assembly.assemble_ordinate_blocks

   The conjugation is applied by ``_to_global_frame``, whose body is
   ``vals * dof_signs[rows] * dof_signs[cols]`` — the row and column
   factors of :math:`\Phi M \Phi`, applied to the COO triplets rather
   than to a materialised matrix. ``dof_signs`` is
   :func:`~orpheus.transport.spatial._ubld.octant_moment_frame_signs`
   tiled over cells, and is ``None`` for a single-moment closure, where
   the function short-circuits to the identity.

   ⚠ ``_to_global_frame`` is a **nested** function and therefore not an
   addressable node in the knowledge graph, so the enclosing
   :func:`~orpheus.sn.loss_representation.assembly.assemble_ordinate_blocks`
   is the implementer this equation can name. That is the right granularity
   in any case: the conjugation is applied at two sites inside that
   function — once on the degenerate cell-diagonal branch and once on the
   walked block — and a declaration on the inner helper would hide the
   fact that both branches are conjugated by the same rule.

Zero-inflow posing — the trace coupling is a separate block
-----------------------------------------------------------

The per-ordinate block emits the bulk-to-bulk action :math:`A_{bb}` at
**zero boundary inflow**: the Cartesian :math:`L+C` is per-ordinate,
per-group block-diagonal on the bulk (no angular or energy coupling
lives in it), and the trace coupling :math:`A_{bs}` — the inflow seed —
is a *separate composite block* deliberately NOT emitted here. Every
consumer of this block (the triangularity gates, the #284
forward-substitution discharge, DSA's :math:`R\,A\,P` moment reduction
for #2 / #200) poses the problem at zero inflow, so the bulk block is
exactly what they need; the boundary law :math:`B` is the sibling
operator that supplies the trace coupling in the full
:math:`(L+C-S-N_{2n}-F-B)` algebra
(:doc:`/theory/foundations/operator_algebra`; the within-group member
list without the fission term is :eq:`sn-within-group-with-n2n`).

.. _loss-rep-assembly-cartesian-scope:

Cartesian-only scope and the pole-seed back edge
------------------------------------------------

The per-ordinate block factorisation exists only in Cartesian geometry.
A curvilinear ordinate couples its angular neighbour through the
Morel–Montry half-angle closure, and the lagged :math:`\psi_{1/2}` pole
seed is precisely a **walk-order back edge**: the seed row reads
*later*-ordinate columns, so no cell ordering makes the block triangular.
The mesh enforces the scope honestly —
:meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.streaming` is the Cartesian
gate the assembler consumes.

Rather than hide the obstruction, 2b **characterised it positively**: a
gate probed the spherical within-group operator from the production
matvec (well-defined on a sphere; only the sweep's inversion lagged) and
asserted the back edge was present (``np.any(triu ≠ 0)``), never an
``xfail`` (L16 discipline — a silent xfail cannot alert on repair). The
cylinder control shows exact triangularity because its first-ordinate
seed is a rank-duplicate of :math:`\psi_0` (:math:`\tau_{{\rm raw},0} =
0`; see :ref:`sn-direct-seed-circle-vs-interval`), so there is no independent
seed row to order. The closed :term:`starting-direction <starting direction>` solve of Hébert
§3.432–3.435 has since **landed** (#282 route (a), #280 Phase 2.5d): the
:math:`\psi_{1/2}` pole seed is now a first-class STATE block whose rows
the augmented :math:`(L+C)` emits, and the gate **flipped** — it now
certifies the augmented one-group matrix is *exactly* block-lower-
triangular in the augmented sweep order
(``test_282_augmented_walk_order_is_triangular``, the transpose analog of
the #284 discharge; L16 loud-flip — it replaces, not relaxes, the RED
characterisation). The bulk per-ordinate assembler itself stays
Cartesian-only; the augmented triangularity is a matvec-probe
certificate, not an assembler capability.

.. _loss-rep-assembly-verification:

Verification — the gates and the sweep-inverse-contract discharge
-----------------------------------------------------------------

The assembly mode is pinned by three gate families (in
``tests/sn/sweep/test_assembly_mode.py`` and the diffusion
``TestAssemblyMode`` class in ``tests/diffusion/test_operators.py``):

* **G1 — object-level matvec equivalence.** ``assembled @ x ≡ apply(x)``
  per ordinate × group × geometry, on heterogeneous, non-uniform-mesh,
  :math:`\ge 2`-group fixtures with a non-flat seeded random :math:`x`.
  It is compared at ``rtol ≈ 1e-11`` (never ``array_equal`` — the CSR
  duplicate-summing order differs from ``apply``'s grouping — and never a
  scalar functional, whose invariance group could hide a block error,
  vv Mode 12); the measured agreement is :math:`\sim 6\times10^{-16}`.
* **G2 — the sweep-inverse-contract discharge (#284).**
  ``scipy.linalg.solve_triangular(PᵀMP, Pᵀq) ≡ (L+C).solve(q)``. LAPACK's
  ``dtrtrs`` is a **structurally independent** forward substitution — it
  is not the ORPHEUS sweep recurrence — so its agreement (again
  :math:`\sim 6\times10^{-16}`) *earns* the L2 cross-check status and
  discharges #284 at the object level: on the source subspace the sweep
  IS forward substitution on the walk-order-triangular matrix. The
  triangularity leg ``triu(PᵀMP, 1) == 0`` is the one **exact**
  (0-tolerance) assertion in the family — triangularity is a structural
  zero, not a numerical near-zero.
* **One-source teeth.** A sign flip monkeypatched into the *shared*
  coefficient source (``_cartesian_streaming_diagonal`` for SN,
  ``_interior_conductance`` / ``_boundary_closure`` for diffusion) must
  move the sweep, the matvec, AND the assembly together: both absolute
  values leave their hand-posed baselines :math:`O(1)` while the
  cross-mode equivalence *persists*. Divergence under a shared-source
  mutation would be the twin-path signature — a STOP-and-catalogue
  condition.

For diffusion the same discipline holds bit-for-bit where the reduction
order matches: the probed-versus-assembled densification (the retained
fuller-view pathway, :ref:`operator-algebra-assembly-axis`) measured a
max :math:`|\Delta| = 0.0` on the heterogeneous fixture, and the
production resolvent
(``MatrixInverseOperator(FlattenedOperator(A, template))``) now
LU-factors the assembled matrix automatically through the ``as_matrix``
delegation — the entire existing diffusion suite becomes the assembled
path's regression net.


.. _loss-rep-four:

The four representations — four schedules of one operator
=========================================================

A :class:`~orpheus.sn.loss_representation.LossRepresentation` is a
stateless frozen dataclass (its only field is the
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` it was selected for). Each is a
distinct **schedule** over the same lower-triangular
:math:`(L+C)` — a different topological linearisation of the identical
cell dependencies. They are *algorithms*, not operators.

.. list-table:: The four loss representations
   :header-rows: 1
   :widths: 22 16 22 20 20

   * - Representation
     - ``supports``
     - Schedule
     - Storage
     - Role
   * - :class:`~orpheus.sn.loss_representation.CumprodScan`
     - 1-D, any geometry
     - Blelloch prefix scan
     - chain recurrence
     - 1-D **production default**
   * - :class:`~orpheus.sn.loss_representation.ScanMarch`
     - 1-D OR 2-D Cartesian
     - ``scan(x) ∘ march(y)``
     - :math:`(d{-}1)`-slab per line
     - 2-D **production default**
   * - :class:`~orpheus.sn.loss_representation.MovingFrontierWindow`
     - Cartesian, d = 2
     - anti-diagonal wavefront
     - rolling :math:`(d{-}1)`-frontier
     - selectable **peer**
   * - :class:`~orpheus.sn.loss_representation.FullFieldWavefront`
     - Cartesian, any d
     - anti-diagonal wavefront
     - full interior cochain
     - verification **oracle**


CumprodScan — the 1-D chain
---------------------------

A 1-D mesh is a total order, so the sweep is a **chain prefix scan**.
:class:`~orpheus.sn.loss_representation.CumprodScan` is intrinsically
1-D — a prefix scan needs a total order, and there is no "2-D chain
scan" (this is *legitimate* d-specificity by the algorithm's nature,
not a narrow crutch). The geometry difference (slab vs sphere vs
cylinder) is absorbed by the two-stratum sweep cache, so all three
geometries share **one body**
(:meth:`~orpheus.sn.loss_representation._OneDimScanWalk.sweep` →
:func:`~orpheus.sn.sweep.scan.ordinate_scan`); the curvilinear
Morel–Montry angular redistribution folds into the scan's affine
source. The recurrence and its closed-form cumprod/cumsum solution are
derived at :ref:`sweep-cumprod`; the pair-monoid algebra that justifies
the closed form is documented on
:func:`~orpheus.sn.sweep.scan.ordinate_scan`.

CumprodScan is conditioning-robust by construction: the closed-form
backend handles the pole reset (ERR-054) and denormal underflow
(ERR-057) that a naive cumprod would lose to gradual underflow.


ScanMarch — the row-march that reuses the scan
----------------------------------------------

:class:`~orpheus.sn.loss_representation.ScanMarch` reframes the d-D DD
sweep as forward substitution along the sweep axis — *the same
first-order linear scan* — **marched over the transverse axes**:

.. math::
   :label: loss-rep-scanmarch

   \mathrm{ScanMarch}
   \;=\;
   \begin{cases}
     \mathrm{scan}(x) & d = 1 \quad(\text{the } s_y = 0 \text{ degeneration})\\[2pt]
     \mathrm{scan}(x)\circ\mathrm{march}(y) & d = 2 \\[2pt]
     \mathrm{scan}(x)\circ\mathrm{march}(y,z) & d = 3.
   \end{cases}

.. implements:: loss-rep-scanmarch
   :by: orpheus.sn.loss_representation.ScanMarch

   The class **is** the schedule. The equation's three cases are its
   admissibility surface, not a runtime branch: the :math:`d = 1` row is
   the :math:`s_y = 0` degeneration handled by the same body, and
   :meth:`~orpheus.sn.loss_representation.ScanMarch.supports` is what
   refuses a mesh the schedule does not cover — so an inadmissible
   ``(representation, mesh)`` pairing is unrepresentable rather than
   mis-executed.

.. implements:: loss-rep-scanmarch
   :by: orpheus.sn.loss_representation.ScanMarch._sweep_interior

   The :math:`\mathrm{scan}(x)\circ\mathrm{march}(y)` body in the SOLVE
   direction: the outer loop marches ``y_rows`` in the octant's sweep
   order and each row runs one
   :func:`~orpheus.sn.sweep.scan._scanmarch_row` along :math:`x`. The
   composition in the equation is literally the nesting of these two
   loops.

.. implements:: loss-rep-scanmarch
   :by: orpheus.sn.loss_representation.ScanMarch._loss_action_interior

   The same composition in the APPLY direction — same march over
   ``y_rows``, but the :math:`x`-faces are *reconstructed* from the known
   probe by a pure-reflection scan rather than solved. Both are declared
   because :eq:`loss-rep-scanmarch` is a statement about the *schedule*,
   and the schedule is what the two directions share (L21); declaring one
   would make the equation read as a property of the sweep alone.

Within each transverse row the :term:`diamond-difference <diamond difference>` x-face recurrence is
the **same Blelloch scan** that
:class:`~orpheus.sn.loss_representation.CumprodScan` uses
(:func:`~orpheus.sn.sweep.scan._scanmarch_row`); the transverse
coupling rides the affine source. The per-axis streaming the row-march
feeds is the **raw down-face coefficient** :math:`g_a = |\mu_a|/\Delta a`;
since #240 Phase 2 Step D5a the diamond factor :math:`2 = 1/w_{\rm DD}`
that scales it lives in the *scheme*, not inline in the march
(:ref:`loss-rep-scanmarch-coefficient-model`). The scheme returns the
per-axis cell-balance coupling

.. math::
   :label: loss-rep-scanmarch-solve

   c_a \;=\; 2\,g_a \;=\; \frac{2|\mu_a|}{\Delta a},
   \qquad
   S \;=\; \Sigma_t + c_x + c_y,

and from it the affine x-scan coefficients
:math:`(\alpha, \beta)` that close the row,

.. math::
   :label: loss-rep-scanmarch-solve-affine

   \alpha \;=\; 2\,c_x\cdot\mathrm{inverse\_denom} - 1,
   \qquad
   \beta \;=\; \big(Q + c_y\,\psi_{y,\rm in}\big)\,
               \frac{\mathrm{inverse\_denom}}{w},
   \qquad
   \mathrm{inverse\_denom} \;=\; \frac1S,
   \quad w = \tfrac12,

.. (vv-status rationale) Derivation step: the DD affine x-scan coefficients
   (alpha, beta). Its terminal (correct 2-D DD flux, byte-identical to
   legacy DD) is tested downstream via transport-cartesian-2d.
.. vv-status: loss-rep-scanmarch-solve-affine documented

where the diagonal :math:`S`, its reciprocal
:math:`\mathrm{inverse\_denom}`, the diamond transmission :math:`\alpha`,
the blend weight :math:`w`, and the transverse coupling :math:`c_y` all
come from a single scheme call,
:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cartesian_scan_coefficients`;
the affine source :math:`\beta` is the generic
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.source_emission`
(:math:`\beta = QV_{\rm eff}\cdot\mathrm{inverse\_denom}/w` with
:math:`QV_{\rm eff} = Q + c_y\,\psi_{y,\rm in}`). The :math:`\times
\mathrm{inverse\_denom}` reciprocal form — never a :math:`\div S`
division — is the same coefficient model the 1-D
:class:`~orpheus.sn.loss_representation.CumprodScan` rides
(:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.affine_scan_coefficients`),
so the row-march **unifies** the 1-D scan (its degenerate
:math:`c_y = 0` case) and the 2-D row-march in one primitive, is DAG-free
(no graph is built), and is now scheme-generic over every
``transverse_coupling_is_facewise`` closure rather than hard-wired to
Diamond Difference. At :math:`w = \tfrac12` these reduce to the legacy
inline DD values byte-for-byte (:math:`\alpha = 2c_x/S - 1`,
:math:`\beta = 2(Q + c_y\psi_{y,\rm in})/S`), :math:`\div\tfrac12` being an
exact power-of-2 :math:`\times 2`; the cell then closes generically in
the blend weight via
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.cell_average`
(:math:`\bar\psi = (1-w)\psi^{\rm in}_x + w\,\psi^{\rm out}_x`) and
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.outgoing_face_from_average`
(:math:`\psi^{\rm out}_y = (\bar\psi - (1-w)\psi^{\rm in}_y)/w`).

.. implements:: loss-rep-scanmarch-solve
   :by: orpheus.transport.spatial.diamond.DiamondDifference._cartesian_streaming_diagonal

   The single source of both halves of this equation: ``couplings =
   tuple(2.0 * s_a for s_a in s_axes)`` is :math:`c_a = 2 g_a`, and the
   diagonal :math:`S` is accumulated by an **explicit left fold**
   ``((Σ_t + c_0) + c_1) + …`` rather than by ``sum()``. The fold order
   is load-bearing, not incidental: it is the IEEE-754 reduction tree of
   record that
   :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch`,
   :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch`
   and the scan producer all share, which is what makes the three callers
   emit the same bytes (:ref:`loss-rep-bit-vs-principled`).

.. implements:: loss-rep-scanmarch-solve-affine
   :by: orpheus.transport.spatial.diamond.DiamondDifference.cartesian_scan_coefficients

   The :math:`\alpha` half, verbatim: ``a = 2.0 * scan_diag *
   inverse_denom - 1.0``, with ``inverse_denom = 1.0 / denom`` and
   ``w = _DD_W``. It returns the reciprocal, never a division by
   :math:`S` — the :math:`\times\,\mathrm{inverse\_denom}` form the
   equation writes.

.. implements:: loss-rep-scanmarch-solve-affine
   :by: orpheus.transport.spatial.scheme.DiscretizationSchemeBase.source_emission

   The :math:`\beta` half: ``return QV * inverse_denom / w``. It lives on
   the **base** class, not on Diamond Difference, because the affine
   source is generic over every closure that supplies an
   :math:`(\alpha, \mathrm{inverse\_denom}, w)` triple — DD's historical
   :math:`2\,QV\,\mathrm{inverse\_denom}` is just its :math:`w = \tfrac12`
   special case.

.. implements:: loss-rep-scanmarch-solve-affine
   :by: orpheus.sn.loss_representation.ScanMarch._sweep_interior

   The equation's :math:`QV_{\rm eff} = Q + c_y\,\psi_{y,\rm in}` fold —
   spelled ``Q_oct[:, :, :, j] + c_y * psi_y_in`` at the
   :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.source_emission`
   call site. That fold lives **here and nowhere else** in the package:
   the scheme owns the coefficients and the schedule owns the decision
   that the transverse face value enters through the source. Declaring
   only the two scheme doors would leave the equation's
   :math:`c_y\,\psi_{y,\rm in}` term unattributed.

The matvec twin reconstructs the interior x-faces from the *known*
probe :math:`\bar\psi` through the scheme's apply-direction **reflection
scan**
(:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.reflect_scan_coefficients`):
because :math:`\bar\psi` is known, the WDD closure
:math:`\psi^{\rm out}_x = 2\bar\psi - \psi^{\rm in}_x` is itself a
first-order recurrence — a **pure-reflection scan** with the recurrence
form of
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.outgoing_face_from_average`
at :math:`w = \tfrac12`,

.. math::
   :label: loss-rep-scanmarch-apply

   \alpha \;=\; -\frac{1-w}{w} \;=\; -1,
   \qquad
   \beta \;=\; \frac{\bar\psi}{w} \;=\; 2\bar\psi
   \qquad(\text{DD},\ w = \tfrac12),

evaluated by :func:`~orpheus.sn.sweep.scan._x_scan_faces`. The per-cell
residual then rides the uniform :math:`\div V` matvec kernel
:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch`
that every facewise closure shares,

.. math::
   :label: loss-rep-scanmarch-apply-residual

   r_{i,j} \;=\;
   (\Sigma_t + c_x + c_y)\,\bar\psi_{i,j}
   \;-\; c_x\,\psi^{\rm in}_{x,i,j}
   \;-\; c_y\,\psi^{\rm in}_{y,i,j}
   \;\equiv\; (L+C)\bar\psi
   \quad(\text{at zero source}),

.. (vv-status rationale) Derivation step: the ÷V matvec residual
   r = (L+C)psi-bar. Foundation-gated by the matvec == sweep round-trip.
.. vv-status: loss-rep-scanmarch-apply-residual documented

from which :eq:`loss-rep-resolution-a` gives :math:`L\bar\psi` — reached, in
the shipped code, by running this same kernel at :math:`\sigma = 0` rather
than by subtracting :math:`\Sigma_t\bar\psi`. ScanMarch additionally inherits the
conditioning robustness of ``ordinate_scan`` (ERR-054 / ERR-057 handled
per line for free) and is the natural home for the flux-independent
``a_attenuation`` two-stratum cache the wavefront lacks (the #206
follow-on).

.. implements:: loss-rep-scanmarch-apply
   :by: orpheus.transport.spatial.diamond.DiamondDifference._reflection_coeffs

   The :math:`w`-generic arithmetic, once: ``alpha =
   np.full_like(psi_bar, -(1.0 - w) / w)`` and ``beta = psi_bar / w``.
   It is a pure static helper in :math:`(\bar\psi, w)` with no instance
   state, which is what lets a future Step closure inherit the recurrence
   form for free.

.. implements:: loss-rep-scanmarch-apply
   :by: orpheus.transport.spatial.diamond.DiamondDifference.reflect_scan_coefficients

   DD's instantiation at :math:`w = \tfrac12`, giving the
   :math:`(-1,\, 2\bar\psi)` the equation prints. It delegates to the
   helper above rather than inlining the constants, so the
   ``-(1-0.5)/0.5 == -1.0`` and ``ψ̄/0.5 == 2ψ̄`` reductions are exact
   power-of-two operations and the result is byte-identical to the legacy
   inline spelling. Both are declared because the equation states the
   generic form **and** its DD value, and those are two different
   symbols.

.. implements:: loss-rep-scanmarch-apply-residual
   :by: orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch

   The :math:`\div V` matvec kernel every facewise closure shares:
   ``residual = denom * psi_bar - numer`` at the probe cell-average, with
   ``denom`` and the couplings from the same
   :meth:`~orpheus.transport.spatial.diamond.DiamondDifference._cartesian_streaming_diagonal`
   the SOLVE arm uses. It is the same symbol that implements
   :eq:`loss-rep-affine-cell`, and deliberately so: this equation is that
   cell relation specialised to the row-march's two axes at zero source.


.. _loss-rep-scanmarch-coefficient-model:

ScanMarch on the coefficient model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before #240 Phase 2 Step D5a (committed ``66dbd9a``, 2026-06-16) the 2-D
row-march **hard-coded** Diamond Difference's
:math:`\alpha`/:math:`\beta`/:math:`D` arithmetic inline:
``ScanMarch._sweep_interior`` and ``ScanMarch._loss_action_interior`` carried
the diamond factor :math:`2 = 1/w_{\rm DD}` and the cell-average blend
:math:`w = \tfrac12` literally in their bodies, so the row-march worked for
**DD only** — no other closure could ride it polymorphically. D5a folds those
two interior kernels onto the **scheme coefficient model** (#239) that the 1-D
:class:`~orpheus.sn.loss_representation.CumprodScan` has ridden since #158
Inc B. The scheme now *owns* the discretization constants; the row-march
becomes a pure schedule that asks the scheme for its coefficients and consumes
generic reconstruction ops. This is the carve that makes the scan-march
**scheme-generic over every** ``transverse_coupling_is_facewise`` closure
(:ref:`loss-rep-scanmarch-facewise`) — the equations :eq:`loss-rep-scanmarch-solve`,
:eq:`loss-rep-scanmarch-solve-affine` and :eq:`loss-rep-scanmarch-apply` above
are the post-fold (coefficient-model) form.

**The litmus that drove the fold** (``coding-elegance`` Pattern 2 / the
coefficient-model litmus): *if an explicit-matrix representation would have to
re-derive a calculation the sweep does, that calculation belongs on the
scheme.* The diamond :math:`2` and the blend :math:`w` are precisely such
shared calculations — the DD cell balance and the DD cell-average closure —
yet they sat duplicated in the row-march body, the 1-D scan cache, and the
batch kernels. D5a moves them to the one place they are defined.

**Raw streaming in, scheme-scaled coupling out.** The per-axis streaming the
row-march feeds the scheme is the **raw down-face coefficient**
:math:`g_a = |\mu_a|/\Delta a` — *not* the pre-scaled :math:`2|\mu_a|/\Delta a`
the inline code used to thread. The scheme applies its own diamond factor and
returns the scaled coupling :math:`c_a = 2 g_a` (:eq:`loss-rep-scanmarch-solve`).
Both interior kernels go through one scheme door per direction:

.. list-table:: The two scheme doors D5a routes the row-march through
   :header-rows: 1
   :widths: 18 30 52

   * - Direction
     - Scheme producer
     - Returns / consumes
   * - SOLVE
     - :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cartesian_scan_coefficients`
       ``(s_scan, s_transverse, sig_t)``
     - the affine x-scan tuple
       :math:`(a,\ \mathrm{inverse\_denom},\ w,\ \text{transverse\_couplings})`
       with :math:`\mathrm{scan\_diag} = 2 g_x`, :math:`c_\perp = 2 g_\perp`,
       :math:`S = \Sigma_t + \mathrm{scan\_diag} + \sum_\perp c_\perp`,
       :math:`\mathrm{inverse\_denom} = 1/S`,
       :math:`a = 2\,\mathrm{scan\_diag}\cdot\mathrm{inverse\_denom} - 1`,
       :math:`w = \tfrac12`
   * - APPLY
     - :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.reflect_scan_coefficients`
       ``(psi_bar)``
     - the pure-reflection :math:`(\alpha = -1,\ \beta = 2\bar\psi)`
       recurrence (:eq:`loss-rep-scanmarch-apply`) — the :math:`(\alpha,\beta)`
       form of
       :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.outgoing_face_from_average`
       at :math:`w = \tfrac12`, :math:`w`-generic via
       :meth:`~orpheus.transport.spatial.diamond.DiamondDifference._reflection_coeffs`

**SOLVE — the row asks, the scheme answers, the cell closes generically.**
Per y-row, ``ScanMarch._sweep_interior`` calls
:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cartesian_scan_coefficients`
to obtain :math:`(a, \mathrm{inverse\_denom}, w, (c_y,))`. The transverse-y
coupling :math:`c_y` does **not** stay separate — it folds into the affine
source: the effective volumetric source is
:math:`QV_{\rm eff} = Q + c_y\,\psi_{y,\rm in}`, and the additive scan
coefficient is the generic
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.source_emission`,
:math:`\beta = \mathrm{source\_emission}(QV_{\rm eff}, \mathrm{inverse\_denom},
w) = QV_{\rm eff}\cdot\mathrm{inverse\_denom}/w`, while
:math:`\alpha = a`. The in-row x-face recurrence
:math:`\psi^{\rm out}_x = \alpha\,\psi^{\rm in}_x + \beta` is the **same
Blelloch scan** (:func:`~orpheus.sn.sweep.scan._x_scan_faces`) the 1-D
``CumprodScan`` runs; the cell then closes **generically in the blend weight**
:math:`w` via the base reconstruction staticmethods —
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.cell_average`
recovers :math:`\bar\psi = (1-w)\psi^{\rm in}_x + w\,\psi^{\rm out}_x` and
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.outgoing_face_from_average`
sheds the downstream y-face :math:`\psi^{\rm out}_y = (\bar\psi -
(1-w)\psi^{\rm in}_y)/w` (the next row's :math:`\psi_{y,\rm in}`). This entire
closure runs through
:func:`~orpheus.sn.sweep.scan._scanmarch_row`, which gained a ``w`` parameter
in D5a (it was a hard-coded ``0.5``) and now consumes the scheme-supplied
blend.

**APPLY — reconstruct off the known probe, residual through the uniform
matvec kernel.** Per y-row, ``ScanMarch._loss_action_interior`` reconstructs
the interior x-faces from the *known* probe :math:`\bar\psi` via the scheme's
reflection scan
:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.reflect_scan_coefficients`
(:eq:`loss-rep-scanmarch-apply`), then evaluates the per-cell residual **and**
the transverse-y outflow through the uniform :math:`\div V` matvec kernel
:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch`
(:eq:`loss-rep-scanmarch-apply-residual`) — the same batched kernel the
anti-diagonal wavefront family uses for its apply direction. This kernel is the
natural home for the matvec because the apply direction has a *concrete*
:math:`\bar\psi`, so the per-cell residual is independent of its neighbours and
needs no closed-form scan; only the SOLVE direction is x-coupled (cell
:math:`i`'s inflow is cell :math:`i-1`'s outflow) and therefore needs the
closed-form affine scan coefficients.

**Why APPLY reuses a kernel but SOLVE needs a new producer.** The asymmetry is
structural, not incidental:

.. list-table::
   :header-rows: 1
   :widths: 14 43 43

   * - Direction
     - Coupling structure
     - Scheme surface
   * - APPLY
     - :math:`\bar\psi` is the **input** → per-cell residual is independent →
       a batched ÷V kernel applies directly
     - reuse the existing
       :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch`
   * - SOLVE
     - :math:`\bar\psi` is the **unknown**, x-coupled cell-to-cell → cannot
       use an independent-batch kernel
     - new closed-form scan producer
       :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cartesian_scan_coefficients`

**Single-sourcing the DD constants.** The fold rests on three pieces of
single-source-of-truth plumbing on
:class:`~orpheus.transport.spatial.diamond.DiamondDifference`:

* the cell-balance diagonal :math:`S = \Sigma_t + \sum_a 2 g_a` is built once
  by the shared private
  :meth:`~orpheus.transport.spatial.diamond.DiamondDifference._cartesian_streaming_diagonal`,
  consumed by **all three** Cartesian producers —
  :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch`,
  :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch`,
  and the new
  :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cartesian_scan_coefficients`
  — as an **explicit left fold** :math:`((\Sigma_t + 2 g_0) + 2 g_1) + \cdots`
  (NOT ``sum()``), the IEEE-754 reduction tree of record
  (:ref:`loss-rep-bit-vs-principled`);
* the blend weight :math:`w = \tfrac12` is the named module constant ``_DD_W``
  — the literal ``0.5`` has exactly one definition site, so no body can drift
  it;
* the apply-direction reflection arithmetic
  :math:`\alpha = -(1-w)/w,\ \beta = \bar\psi/w` is the :math:`w`-generic
  :meth:`~orpheus.transport.spatial.diamond.DiamondDifference._reflection_coeffs`,
  which DD's
  :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.reflect_scan_coefficients`
  calls at ``_DD_W``, so the future Step closure inherits it for free.

The curvilinear ``affine_scan_coefficients`` merge — unifying the 1-D-chain
producer and the Cartesian row-march producer into one surface — is deliberately
**deferred to #242**: ``affine_scan_coefficients`` is 1-D-chain-shaped (it
carries curvilinear geometry arguments ``abs_mu``/``face_area_downstream``/``face_area_total``/``delta_A_over_w``/``c_out``/``volume``
with no transverse slot), whereas ``cartesian_scan_coefficients`` needs the
transverse :math:`c_\perp` in the diagonal and the raw scan-axis :math:`g`, so
they are kept as separate methods for now (a ``d``-generic ``s_transverse``
tuple keeps the latter ready for d=3, where the row would sum multiple
transverse couplings).

**The bit-identity / principled-equivalence posture.** The carve is a
**principled ~1-ULP re-baseline** (``vv-principles``
§":ref:`Bit-identity vs principled-equivalence <loss-rep-bit-vs-principled>`",
precedent #158-B1), not a bit-identical move on the 2-D path. The reason is
specific: the pre-D5a ``ScanMarch._sweep_interior`` was the **one remaining**
place in the SN stack still using a :math:`\div D` **division**
(:math:`\alpha = 2 s_x / D`, :math:`\beta = 2(\cdots)/D`), *not* the
:math:`\times\,\mathrm{inverse\_denom}` **reciprocal** the 1-D
``CumprodScan`` already rode. Folding the 2-D solve onto the coefficient model
switches :math:`\div D \to \times(1/D)`, a genuine FP-association change — the
*same* sanctioned re-baseline the 1-D path took at #158 Inc B — and the matvec
likewise re-associates as it joins the ÷V ``residual_kernel_batch`` reduction.

.. admonition:: What re-baselined and what stayed byte-identical
   :class: note

   The fold re-associates :math:`\sim 1` ULP on the **2-D solve AND the 2-D
   matvec**; the converged *values* are unchanged. The discriminating
   evidence (verified against the three ``vv-principles`` criteria, not
   old-vs-new proximity):

   * **principled** — named coefficient-model ops, zero inline DD in
     ``_sweep_interior`` / ``_loss_action_interior`` / ``_scanmarch_row``;
   * **structurally-independent reference, two angles** — (i) the
     ``ScanMarch ≡ FullFieldWavefront`` oracle
     (``test_scan_march_equivalence.py``, :eq:`loss-rep-scanmarch`) pins the
     value to the analytical :math:`k_\infty = 1.875` /
     :math:`\varphi = Q/\Sigma_t` grounds, and (ii) ``test_keff_2d`` pins the
     homogeneous :math:`k_\infty = \nu\Sigma_f/\Sigma_a` (a closed-form,
     :math:`\ge 2`-group eigenvalue ground — never a 1-group degeneracy);
   * **two structurally-independent code paths agree** — the post-fold 2-D
     SI ``.solve`` :math:`\varphi` equals the Krylov ``.apply``
     :math:`\varphi` to :math:`\approx 2.8\times10^{-12}` relative (the SI
     iterate never touches the matvec override; the Krylov matvec does), and
     the drift is FP-non-associativity (iteration-count :math:`\times` ULP for
     SI, GMRES-tol :math:`\times` ULP for Krylov);
   * **the 1-D scan path is byte-identical** — it already rode
     :math:`\times\,\mathrm{inverse\_denom}`, so the fold did not touch it.
     This was the **negative control**: at D5a the slab SHA in
     ``test_affine_carve_bit_identity.py`` *was* byte-unchanged, proving the
     re-baseline was confined to the 2-D path the division-to-reciprocal switch
     actually moved. (Both halves are now history: #333 retired that module's
     digests for stored values, and ``579d5eaf`` has since moved the slab arm
     too — by 3/17 ULP in the Gauss-Legendre nodes/weights, unrelated to D5a.)

This matches the affine-carve file's own history: the 2-D Krylov golden hashes
re-baselined at #240 Phase 2 Step B for the apply re-association; D5a extends
that to the **SI-2D solve** arm too, because the solve fold is what switched the
division to the reciprocal.

**Verification gates.** The carve is pinned by, in order:

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Gate
     - Tag
     - What it pins
   * - ``test_scan_march_equivalence.py``
     - ``foundation``,
       ``verifies("loss-rep-scanmarch", -solve, -apply)``
     - ScanMarch :math:`\equiv` FullFieldWavefront oracle at ``nulp`` /
       absolute tolerance, on :math:`\ge 2`-group heterogeneous anisotropic
       **non-square** meshes (the x↔y swap moat, failure Mode 2). The existing
       parametrize already covers the Mode-9 degeneracy break (a non-square,
       multi-group, reflective tuple) — no new case was needed.
   * - ``test_streaming_operator.py::TestT4bPreT4RegressionSnapshot``
     - regression
     - the cart2d matvec frozen reference plus three slab arms; the cart2d
       ``*_apply_bulk`` snapshots were **regenerated in-commit** to the
       post-D5a value (the regenerate-in-commit discipline,
       :ref:`loss-rep-bit-vs-principled`), boundary byte-identical.
   * - ``test_affine_carve_bit_identity.py``
     - regression (stored reference; strict until #333)
     - *at D5a* the negative control — the 1-D slab SHA was byte-unchanged
       while the two 2-D SHAs (``si_2d_p1_aniso_het``,
       ``krylov_2d_p1_aniso_het``) re-baselined off the
       division-to-reciprocal switch. ⛔ The SHAs no longer exist: #333
       re-posed this module onto stored VALUES because a digest cannot
       report a magnitude.

The two equation labels :eq:`loss-rep-scanmarch-solve` and
:eq:`loss-rep-scanmarch-apply` are the gate's ``verifies(...)`` targets, so the
rewrite above keeps both labels (the bodies changed to the coefficient-model
form; the *claims* — the scan-march coefficients and their solve/apply
realizations — are unchanged and the oracle still pins them).

**The zero-inline-DD consequence.** With the fold complete,
``ScanMarch._sweep_interior``, ``ScanMarch._loss_action_interior`` and
``_scanmarch_row`` carry **no** inline ``2.0*``, **no** ``Diamond`` reference,
and **no** hard-coded ``0.5`` in code (only in docstrings as :math:`w = \tfrac12`
notes). Polymorphic readiness is therefore satisfied *by construction*: the
moment a ``Step.cartesian_scan_coefficients`` /
``Step.reflect_scan_coefficients`` pair lands, Step rides the row-march with
**zero** further ``ScanMarch`` change. This also gives the routing gate
(:ref:`loss-rep-scanmarch-facewise`) its second, independent enforcement leg:
Linear-Discontinuous's exclusion from the row-march is now enforced both by
``ScanMarch.supports`` (the trait gate) **and** by the structural *absence* of
inline DD — the two routes agree, so a non-facewise scheme can neither be
selected for the row-march nor silently compute DD inside it.


MovingFrontierWindow — the rolling-frontier wavefront
-----------------------------------------------------

:class:`~orpheus.sn.loss_representation.MovingFrontierWindow` is the
anti-diagonal (level-scheduled) wavefront over the per-octant
:class:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph` derived at
:ref:`sweep-wavefront`. It carries only a **rolling**
:math:`(d{-}1)`-frontier of interior face fluxes (a 2-diagonal at d=2),
advanced anti-hyperplane by anti-hyperplane via
:meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_windowed`. The
frontier is the moving realisation of the interior face cochain
:math:`C^1_{\rm int}` — its theory and the post-``WavefrontFlux``
succession are at :ref:`wavefront-flux-cochain`. Its historical claim
to fame was a **~30 % peak-memory win over the full-field oracle**; the
Fork-B2 measurement (:ref:`loss-rep-fork-b2`) showed that the *same*
memory profile is shared by the row-march, which is why the window is
now a selectable peer rather than the default.


FullFieldWavefront — the verification oracle
--------------------------------------------

:class:`~orpheus.sn.loss_representation.FullFieldWavefront` walks the
**same** anti-diagonal DAG schedule as the window, but retains the
**whole** interior face cochain — the *fuller view*. It seeds only the
octant's domain in-edge slot (``_octant_face_cochain``); by the upwind
invariant every other slot is written before any read, so the
zero-initialised buffer is byte-identical to the historical whole-trace
:math:`\iota_*` seed. It is slower and more memory-hungry by design —
its purpose is to be the reference the d-specific production
optimisations are cross-checked against:

* the 1-D :class:`~orpheus.sn.loss_representation.CumprodScan`
  (principled-equivalence at nulp);
* the 2-D :class:`~orpheus.sn.loss_representation.MovingFrontierWindow`
  (``window ≡ full`` bit-identity);
* the multi-D :class:`~orpheus.sn.loss_representation.ScanMarch`
  (principled-equivalence at nulp).

Keeping a *fuller view of the concept* as a pinned verification oracle
is the deliberate **aggressive-retirement exception**: the production
paths could not be cross-checked structurally without it. It is the one
genuinely d-generic body (``supports`` is any-d Cartesian), so it is
also the admission spine for synthetic d=3 correctness tests before any
3-axis mesh exists (the angular :term:`quadrature` is already 3-cosine with all
8 sign-octants — what is missing at d=3 is ``Mesh3D``, not the
quadrature), and the representation a d=3 Cartesian mesh falls through
``default_for`` to (C3.6: :class:`ScanMarch` narrowed its ``supports``
to the d=2 truth of its kernels).


.. _loss-rep-selection:

Selection: one predicate, three consumers
==========================================

Applicability is a **declared, queryable capability** — "make illegal
states unrepresentable" applied to method selection. Each representation
answers one classmethod, over the mesh **and the spatial closure it was
handed**:

.. code-block:: python

   class CumprodScan(_LossRepresentation):
       @classmethod
       def supports(cls, mesh, spatial_closure):
           if not mesh.is_1d:
               return Compatibility(False, "requires a 1-D mesh")
           # geometry × closure capability first, so a curvilinear-incapable
           # scheme is refused with its OWN reason rather than "not 1-D".
           geometry = _curvilinear_capability(mesh, spatial_closure)
           if not geometry.ok:
               return geometry
           return Compatibility(
               spatial_closure.is_affine_scannable,
               "requires a 1-D mesh with an affine-scannable cell-update scheme",
           )

   class ScanMarch(_LossRepresentation):
       @classmethod
       def supports(cls, mesh, spatial_closure):
           # 1-D arm: SINGLE-axis prefix-scannability (LD's 1-D scan IS valid).
           if mesh.is_1d:
               geometry = _curvilinear_capability(mesh, spatial_closure)
               if not geometry.ok:
                   return geometry
               return Compatibility(
                   spatial_closure.is_affine_scannable,
                   "requires an affine-scannable cell-update scheme on a "
                   "1-D mesh (any geometry)",
               )
           # d≥2 arm: the DISTINCT cross-axis separability claim (DD/Step,
           # NOT LD — see the scheme-gate subsection below).
           return Compatibility(
               mesh.is_cartesian
               and mesh.ndim == 2
               and spatial_closure.transverse_coupling_is_facewise,
               "2-D scan-march requires a scheme whose transverse coupling is "
               "facewise (separable into independent per-axis 1-D scans) — the "
               "slopeless cell-average closures (Diamond Difference, Step); "
               "Linear-Discontinuous's bilinear slope coupling needs the "
               "wavefront (the d≥3 row-march kernels are deferred — the "
               "full-field spine serves d≥3)",
           )

   class _DAGWavefront(_LossRepresentation):       # MovingFrontierWindow's base
       @classmethod
       def supports(cls, mesh, spatial_closure):
           return Compatibility(
               mesh.is_cartesian and mesh.ndim == 2,
               "requires Cartesian geometry, d = 2",
           )

   class FullFieldWavefront(_DAGWavefront):
       @classmethod
       def supports(cls, mesh, spatial_closure):
           return Compatibility(mesh.is_cartesian, "requires Cartesian geometry")

.. note:: **The closure is an ARGUMENT, not a mesh read (P4.9b,
   2026-08-28).**

   ``supports`` used to read ``mesh.scheme``.  It now takes the closure
   the posed operator was constructed with, because *selection is a
   method-flavoured question* and the ruling that keeps the generator on
   the hub (:ref:`sn-p49b-operator-poses-with-closures`) puts only
   space-and-layout facts on the hub route.  The two spellings agree on
   every production path — the operator's closure IS the hub's object,
   by construction, since
   :meth:`~orpheus.sn.operators.streaming.StreamingOperator.pose` reads
   it there — so this is a *routing* change, not a behavioural one.  Its
   value is that a deliberately doctored pair now selects the strategy
   that matches what will actually be swept, instead of the strategy the
   mesh would have picked.

   The same argument moves the representation's own construction: the
   base carries ``(mesh, spatial_closure, angular_closure)``,
   :func:`~orpheus.sn.loss_representation.default_for` takes all three,
   and a ``pose(mesh)`` classmethod mirrors the operator's for the
   test-side construction that used to pass a bare mesh.

The compatibility signal is the *genuine* criterion — the coordinate
system (:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.is_cartesian`, i.e.
``curvature is None``), the dimensionality
(:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.ndim`), **and the cell-update
scheme's capability traits** — **not** the ``sweep_graphs is None``
substrate proxy that the pre-carve code keyed on. The two scan
representations read a *scheme* trait, not just geometry:
``CumprodScan`` and the 1-D arm of ``ScanMarch`` require
:attr:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.is_affine_scannable`,
and the :math:`d \ge 2` arm of ``ScanMarch`` requires the **distinct**
:attr:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.transverse_coupling_is_facewise`
— the split that closes the silent 2-D Linear-Discontinuous misroute
(:ref:`loss-rep-scanmarch-facewise`).
:class:`~orpheus.sn.loss_representation.Compatibility` is an
``(ok, reason)`` pair; the ``reason`` lets a teaching frontend gray-out
a method *and explain why* ("Moving-frontier window — requires Cartesian
geometry, d = 2"), which is pedagogically load-bearing — ORPHEUS teaches
reactor physics.

**One predicate, three consumers (single source of truth):**

#. **Frontend** —
   ``[R for R in LOSS_REPRESENTATIONS if R.supports(mesh, closure).ok]``
   lists the applicable methods. A cylinder (non-Cartesian) → only
   ``CumprodScan`` and ``ScanMarch``; the dropdown shows exactly those.

#. **Factory default** —
   :func:`~orpheus.sn.loss_representation.default_for` returns the
   **first** entry in the ordered registry
   :data:`~orpheus.sn.loss_representation.LOSS_REPRESENTATIONS` whose
   ``supports`` admits the mesh, falling back to the oracle so it is
   never stuck:

   .. list-table:: ``default_for`` outcomes (first-supports-match) — **facewise scheme (DD / Step)**
      :header-rows: 1
      :widths: 22 40 22

      * - mesh
        - applicable (registry order)
        - ``default_for``
      * - Cart-1D
        - ``{CumprodScan, ScanMarch, FullField}``
        - ``CumprodScan``
      * - Cart-2D
        - ``{ScanMarch, MovingFrontierWindow, FullField}``
        - ``ScanMarch``
      * - Cart-3D
        - ``{ScanMarch, FullField}`` (window is d=2 only)
        - ``ScanMarch``
      * - Cyl/Sph-1D
        - ``{CumprodScan, ScanMarch}``
        - ``CumprodScan``

   .. admonition:: The outcome depends on the **scheme**, not only the mesh
      :class: note

      The table above is for a mesh carrying a **facewise** cell-update
      scheme (Diamond Difference — the production default — or Step). Since
      #240 D5-0 the ``supports`` predicates also read the scheme's capability
      traits, so a non-facewise scheme changes which representations apply:

      .. list-table::
         :header-rows: 1
         :widths: 26 40 22

         * - mesh + scheme
           - applicable (registry order)
           - ``default_for``
         * - Cart-2D + **DD/Step** (facewise)
           - ``{ScanMarch, MovingFrontierWindow, FullField}``
           - ``ScanMarch``
         * - Cart-2D + **LD** (slope-wise)
           - ``{MovingFrontierWindow, FullField}`` — ``ScanMarch`` **refuses**
           - ``MovingFrontierWindow``

      A 2-D **Linear-Discontinuous** mesh is refused by ``ScanMarch.supports``
      (its row-march interior runs inline Diamond Difference, which would
      silently drop LD's slope) and falls through to a **wavefront**
      (``MovingFrontierWindow`` / ``FullFieldWavefront``). The wavefront's LD
      kernel is itself d=1-only today, so the 2-D LD sweep raises an honest
      ``NotImplementedError`` rather than returning DD values — see
      :ref:`loss-rep-scanmarch-facewise` for why DD/Step are facewise and LD
      is not.

   The **registry order is the policy**:

   .. code-block:: python

      LOSS_REPRESENTATIONS = (
          CumprodScan,            # 1-D production default
          ScanMarch,              # 2-D Cartesian production default
          MovingFrontierWindow,   # selectable peer (d = 2)
          FullFieldWavefront,     # never-stuck any-d oracle fallback
      )

   At d=1 ``CumprodScan`` wins (registered first; ``ScanMarch`` would
   also apply but degenerates to the same scan with a march shell). At
   d=2 ``ScanMarch`` wins — the **Fork-B2 flip** (2026-06-11) that moved
   ``ScanMarch`` ahead of ``MovingFrontierWindow`` in the tuple. At d=3
   (when ``Mesh3D`` lands) the first two refuse and ``default_for``
   falls through to the d-generic ``FullFieldWavefront`` spine — pinned
   today at the ``supports`` level
   (``test_unified_sweep_dispatch.py::TestD3SupportsMatrix``, C3.6).
   The day a d=3 row-march or window lands, widening its ``supports``
   is the *only* change needed for Cart-3D's default to upgrade — one
   predicate, no caller touched.

#. **Construction guard** — ``_LossRepresentation.__post_init__``
   re-runs ``supports(self.mesh, self.spatial_closure)`` and raises
   :class:`~orpheus.sn.loss_representation.IncompatibleRepresentation`
   on a false verdict, so even a bypassed UI cannot build an illegal
   pairing. Combined with the frozen-dataclass immutability, the
   ``(representation, mesh, closure)`` triple is *correct by
   construction* — and since P4.9b the closure it validates is the one
   the strategy will actually sweep with, not whichever one the mesh
   happens to carry.

That ``supports`` predicate **is** the ``is_1d`` / ``curvature``
dispatch that the pre-carve code scattered across ``transport_sweep``
plus five operator gates — now declared **once** per representation.
"Add 3-D window support" becomes "extend one representation, widen one
predicate," not a hunt through every call site.


.. _loss-rep-scanmarch-facewise:

The scan-march scheme gate: facewise vs slope-wise transverse coupling
----------------------------------------------------------------------

The :math:`d \ge 2` arm of
:meth:`~orpheus.sn.loss_representation.ScanMarch.supports` reads a **scheme**
trait,
:attr:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.transverse_coupling_is_facewise`,
not just geometry. This subsection records *why* the row-march needs a
genuinely different qualification than the 1-D scan does, and how a single
conflated predicate silently misrouted a 2-D Linear-Discontinuous mesh into
computing Diamond Difference — issue #240 Phase 2 Step D5-0, a
**routing-honesty fix** that closed a live correctness hole by making the
illegal pairing unrepresentable.

.. admonition:: The bug, in one line
   :class: important

   Before D5-0 the :math:`d \ge 2` scan-march was gated on the SINGLE-axis
   trait ``is_affine_scannable``, which Linear-Discontinuous **satisfies**.
   So ``default_for(2-D LD mesh)`` selected ``ScanMarch``, whose row-march
   interior runs *inline Diamond Difference with no scheme dispatch* — a 2-D
   LD problem **silently computed DD**, dropping LD's bilinear slope (a
   vv-principles failure Mode 2, *variable / formulation swap*, masquerading
   as a correct answer). D5-0 mints the **distinct** cross-axis trait
   ``transverse_coupling_is_facewise`` and narrows the :math:`d \ge 2` arm to
   read it, so the LD/scan-march pairing can no longer be formed.

Two different questions: 1-D prefix-scannability vs cross-axis separability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The conflation at the root of the bug is that **one** trait was made to
answer **two** structurally distinct questions. They look similar — both are
"can a scan consume this scheme?" — but they live on different axes:

.. list-table:: The two scan-family qualifications a scheme can carry
   :header-rows: 1
   :widths: 30 34 36

   * - Trait
     - The question it answers
     - What it licenses
   * - :attr:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.is_affine_scannable`
     - **Single-axis**: is the cell-average an affine function of a *single*
       upstream face flux, :math:`\psi_{\rm out} = a\,\psi_{\rm in} + b`
       (Blelloch §1.5)?
     - The 1-D prefix scan (``CumprodScan``; the 1-D arm of ``ScanMarch``).
   * - :attr:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.transverse_coupling_is_facewise`
     - **Cross-axis**: in :math:`d \ge 2`, does a NON-swept axis couple
       through a *0th-order face value* (so the :math:`d`-D closure is
       tensor-product separable into independent per-axis scans)?
     - The :math:`d \ge 2` row-march (``scan(x) ∘ march(y)``).

The decisive fact is that **these answers can differ for the same scheme**.
A scheme can be perfectly prefix-scannable along one axis (the slope it
carries is eliminated *locally*, leaving a clean single-upstream recurrence)
yet have a multi-dimensional closure that does **not** separate across axes.
Linear-Discontinuous is exactly that scheme:
``is_affine_scannable = True`` (a 1-D fact) but
``transverse_coupling_is_facewise = False`` (a :math:`d \ge 2` fact). Diamond
Difference, by contrast, satisfies *both* — which is precisely why the bug
hid: on the production-default scheme the two traits coincide, so the
conflated predicate gave the right answer on every shipped path and only the
LD path exposed the gap.

Why the row-march needs cross-axis separability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The row-march :math:`\mathrm{scan}(x) \circ \mathrm{march}(y)`
(:eq:`loss-rep-scanmarch`) sweeps one transverse row at a time. Within a row
it runs the *same* first-order linear scan
(:func:`~orpheus.sn.sweep.scan._scanmarch_row`) the 1-D
:class:`~orpheus.sn.loss_representation.CumprodScan` uses, and the coupling to
the previously-marched row enters **the scan's affine source** as a single
term :math:`c_y\,\psi_{y,\rm in}` (:eq:`loss-rep-scanmarch-solve-affine`). Read
off the solve coefficient
:math:`\beta = (Q + c_y\,\psi_{y,\rm in})\,\mathrm{inverse\_denom}/w`: the
*only* way axis :math:`y`
reaches the x-scan is through that one number :math:`\psi_{y,\rm in}` — the
0th-order **face value** entering the row from below. The march absorbs every
non-swept axis into the scan source as one scalar trace per cell.

This is correct **if and only if** the cell's transverse coupling really is a
single face value. Formally, the row-march is exact when the :math:`d`-D cell
closure is **tensor-product separable**:

.. math::
   :label: loss-rep-facewise-separable

   M_{d}\;=\;\bigoplus_{a}\, M^{(1)}_{a}
   \quad\Longleftrightarrow\quad
   \text{the }d\text{-D update}\ =\ \text{independent per-axis 1-D scans,
   chained by scalar face traces,}


.. no-implementation:: loss-rep-facewise-separable
   :kind: definition

   **Nothing implements this** — it defines the criterion, not a routine.
   The page's own vv-status rationale: *"Definitional separability criterion
   (the transverse_coupling_is_facewise trait). Definitional, not
   orphan-gated."* The trait a scheme sets is an ASSERTION that it satisfies
   this; the biconditional itself is what the trait is judged against.

.. (vv-status rationale) Definitional separability criterion (the
   transverse_coupling_is_facewise trait). Definitional, not orphan-gated.
.. vv-status: loss-rep-facewise-separable documented

so that the per-axis updates commute and a scan along :math:`x` marched over
:math:`y` reconstructs the same solution the joint :math:`d`-D system would.
``transverse_coupling_is_facewise`` is the name for exactly this property:
*the non-swept axis enters as a 0th-order Dirichlet face trace, not as a
higher-order moment the swept-axis row must consume.*

Diamond Difference / Step: facewise (separable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Diamond Difference (and Step, when it is built) is a **slopeless
cell-average** closure: each cell carries a single unknown, the cell-average
flux :math:`\bar\psi`, with the downstream face reconstructed from it by the
diamond mean :math:`\psi^{\rm out} = 2\bar\psi - \psi^{\rm in}`. Its
:math:`d`-D balance (:eq:`dd-cartesian-2d`) is the standard tensor-product
central-difference structure (Lewis & Miller §§4.5, 8): the coupling from a
transverse axis :math:`y` is the **single 0th-order face value**
:math:`c_y\,\psi_{y,\rm in}` (:math:`c_y = 2 g_y`) that folds straight into the
scan source. In the
batched DD kernel
(:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch`) this
is the explicit per-axis left fold
:math:`\mathrm{numer} = Q + \sum_a 2 g_a\,\psi^{\rm in}_a` — every axis
contributes one additive face term, exactly the separable form
:eq:`loss-rep-facewise-separable`. So a row-march along :math:`x` that carries
each :math:`y`-face value into the source is **exact** for DD/Step, and the
scheme declares ``transverse_coupling_is_facewise = True``
(:class:`~orpheus.transport.spatial.diamond.DiamondDifference`).

Linear Discontinuous: slope-wise (non-separable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linear Discontinuous (DG-P1-upwind) represents the in-cell :term:`angular flux` as a
**linear function** rather than a constant, so each cell carries a *second*
spatial moment, the slope :math:`\hat\psi`, alongside the average
:math:`\bar\psi` (the 1-D two-moment system and its Schur-complement
reduction are derived in the
:class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`
docstring). In **1-D** that slope is a *local* quantity:
the Schur complement of the per-cell :math:`2\times2` eliminates
:math:`\hat\psi` analytically, leaving the clean single-upstream affine
recurrence :math:`\psi_{\rm out} = a\,\psi_{\rm in} + b` — which is exactly
why LD is ``is_affine_scannable`` along one axis.

In :math:`d \ge 2` that elimination no longer decouples the axes. The cell's
transverse face flux **varies linearly along the in-cell swept coordinate**
(that is the defining feature of a P1 representation), so the coupling from a
non-swept axis is a **1st-order slope moment** :math:`\hat\psi_y`, *not* a
single face value. The slope row of the swept axis must consume it; the
:math:`d`-D closure is an irreducible, axis-coupled per-cell block whose Schur
complement does **not** diagonalize across axes — it fails the separability
:eq:`loss-rep-facewise-separable`. There is no single scalar trace the march
can fold into the scan source that would carry the transverse slope. LD
therefore declares ``transverse_coupling_is_facewise = False`` (it inherits
the conservative default;
:class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`), and a
multi-dimensional LD problem must ride the **DAG wavefront**, which resolves
the genuine joint cell dependencies, not the scan-march.

.. note::

   **Scope.** The multi-D LD cell closure itself — the per-cell
   :math:`d`-dimensional bilinear system — is **not shipped** (it is the #240
   D5b / D6 deliverable, in design). The Cartesian multi-D LD is the
   tensor-product **bilinear (Upstream LD / UBLD)** object,
   :math:`2^d` moments on the basis :math:`\{1, x, y, xy\}` — not the
   simplex :math:`1+d` form (which fails the thick-diffusion limit on
   quadrilaterals). This page records only *that* LD's transverse coupling
   is a slope moment (hence non-separable), which is all the routing gate
   needs; the discretization of the multi-D LD block is deferred to its own
   theory section when D5b lands.

Making the illegal pairing unrepresentable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fix is a textbook application of ``coding-elegance`` Pattern 4
(*illegal states unrepresentable*): the bug was a **dimensional predicate
standing in for a scheme-capability question**. Pre-D5-0 the :math:`d \ge 2`
arm read ``is_cartesian and ndim == 2`` — a pure geometry test that admitted
*any* 2-D Cartesian scheme, including the one whose interior the row-march
cannot honour. Minting the distinct ``transverse_coupling_is_facewise`` and
narrowing the arm to read it (:meth:`~orpheus.sn.loss_representation.ScanMarch.supports`,
the refreshed code block above) means a 2-D LD mesh now answers
``supports().ok == False``, so ``default_for`` never returns ``ScanMarch`` for
it and the construction guard (``__post_init__``) rejects an explicit
``ScanMarch(2-D LD mesh)``. The pairing is gone by construction.

The trait is declared on **both** the concrete base
:class:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase` (the typed-and-
defaulted single source of truth) and the
``@runtime_checkable`` :class:`~orpheus.transport.spatial.scheme.DiscretizationScheme`
Protocol (kept symmetric with the other three capability traits). The
``@runtime_checkable`` Protocol carries a known footgun: on Python 3.12+,
``isinstance`` validates member *presence*, not type — so a scheme declaring
``transverse_coupling_is_facewise = "yes"`` would pass the ``isinstance``
check and then read *truthy* in ``supports``, re-opening a **narrower**
instance of the very silent-misroute the trait was minted to close. That
footgun is shut by ``TestCapabilityTraitsAreGenuineBools`` in
``tests/sn/sweep/core/test_discretization_scheme_protocol.py``: every
*registered* production scheme is asserted to declare all four capability
traits as a **genuine** ``bool`` (``isinstance(value, bool)`` — rejecting both
truthy ``int`` and ``np.bool_``), so a non-bool trait fails the foundation
gate rather than mis-routing in production.

.. admonition:: The default is conservative opt-in
   :class: note

   Both scan-family capability traits default to ``False`` on the base
   (``is_affine_scannable`` and ``transverse_coupling_is_facewise`` alike). A
   scheme must **declare** the capability to claim it; a newly-added scheme
   that forgets to set ``transverse_coupling_is_facewise`` is therefore
   *safely excluded* from the :math:`d \ge 2` scan-march and falls back to
   the wavefront — "slow but correct," never "fast but wrong." This is the
   exclusionary default that makes the opt-in safe.

Why scheme-named, not strategy-named
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The trait is named for the **scheme property** (separable transverse
coupling), *not* for the consuming strategy (a rejected alternative was
``is_scan_march_compatible``). Naming a scheme property after one of its
consumers is a frame-leak: it bakes the current caller into the abstraction
and forces a rename the moment a second consumer appears. And a second
consumer is already confirmed — the **diffusion ADI / line-SOR
preconditioner** (#240's next consumer) decides whether it can sweep one axis
at a time by reading *exactly this same predicate*: a line-relaxation that
marches one axis is correct only when the transverse coupling is facewise. By
naming the trait for the property, that future consumer reuses it with no
rename. The strategy-independence of the property is itself a tested fact:
``TestSchemeTraitProbe`` reads the trait directly off the scheme class with
**no** ``ScanMarch``, ``supports``, or mesh in scope, proving the answer is a
genuine scheme property a second consumer can ask of a scheme in isolation.

Verification — the routing-honesty gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because D5-0 is a *routing* change (it alters which representation a mesh
selects, not any computed value), its gates are **selection** and
**refusal** assertions, all ``foundation``-tagged (software-structure
invariants, no theory ``:label:``) and ``-O``-safe (``pytest.fail``, never
bare ``assert`` — vv-principles failure Mode 8). Two test files carry them:

* In ``tests/sn/sweep/core/test_unified_sweep_dispatch.py``,
  ``TestD3SupportsMatrix`` pins the routing honesty four ways:

  - ``test_scan_march_refuses_2d_non_facewise_scheme_fake`` — on a synthetic
    scheme, ``supports`` **refuses** ``facewise=False`` and **admits**
    ``facewise=True`` (the anti-pattern-#11 *both-directions* pair: a
    refusal gate that only ever refuses validates the raising, not the
    invariant);
  - ``test_scan_march_refuses_2d_ld_real_mesh`` — on a real
    ``SNMesh(Mesh2D, LS-S4, scheme=LinearDiscontinuous())``,
    ``ScanMarch.supports(...).ok is False`` (the CONFIRMED-LIVE misroute,
    now closed);
  - ``test_2d_ld_default_for_routes_to_wavefront`` —
    ``default_for(2-D LD)`` lands on a wavefront
    (``MovingFrontierWindow`` / ``FullFieldWavefront``), **never**
    ``ScanMarch``;
  - ``test_2d_ld_sweep_raises_not_silently_dd`` — the **headline honesty
    claim**: ``solve_sn_fixed_source`` on a 2-D LD mesh now **raises**
    ``NotImplementedError`` (the loud d=1-only signal) rather than returning
    DD numbers via the inline-DD row-march. *Silent-wrong became
    loud-not-yet-implemented.*

* ``TestSchemeTraitProbe`` (same file) is the strategy-free probe described
  above: DD reports ``True`` standalone, LD reports ``False`` standalone,
  and ``test_facewise_distinct_from_affine_scannable`` asserts the two traits
  **diverge on LD** (``is_affine_scannable=True``,
  ``transverse_coupling_is_facewise=False``) while coinciding on DD — the
  split is *observable* only on a scheme where the two answers differ, which
  is LD. Were they to coincide on LD, the conflation that drove the misroute
  would still be latent.

* In ``tests/sn/sweep/core/test_discretization_scheme_protocol.py``,
  ``TestCapabilityTraitsAreGenuineBools`` is the genuine-``bool`` teeth that
  shut the ``@runtime_checkable`` presence-only footgun (above).

The change is **bit-identical for every exercised path**: it is a routing
predicate that touches no computed flux on any path that ran before (a 2-D LD
mesh previously *ran* — wrongly — and now *raises*; every other path is
unchanged). The strict bit-identity gate's pre-existing set was unmoved; the
only delta is the seven added D5-0 tests (closeout memo
``.claude/agent-memory/method-implementer/issue_240_phase2_step_d5_0_closeout.md``).

The honest interim state and what closes it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

D5-0 deliberately stops at *routing*. It does **not** supply the multi-D LD
kernel — so a 2-D LD mesh, having been correctly diverted to the wavefront,
hits the wavefront's LD ``cell_kernel_batch``, which is itself d=1-only and
raises the existing ``NotImplementedError("…supports d=1 (slab/1-D) only;
got d=2…")`` from
:class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`. This is
the **correct interim state**: a loud "not yet implemented" is strictly better
than a silent wrong answer (a different scheme's values returned under LD's
name). The follow-on:

* **D5b** supplies the multi-D bilinear LD kernel (the UBLD per-cell block),
  closing the wavefront raise — at which point 2-D LD *runs* on the wavefront,
  correctly, and ``ScanMarch`` still (correctly) refuses it.
* **D5a** folds the DD scan-march onto the coefficient model so that
  ``ScanMarch`` becomes scheme-generic — at which point LD's exclusion from the
  row-march is enforced by the *absence* of inline DD, not only by
  ``supports``; the trait gate and the structural absence then agree by two
  independent routes.


.. _loss-rep-one-walk-one-instance:

The one-walk and one-instance theorems
======================================

The structural payoff of the S6 re-layering is two theorems, each a
*type fact* enforced by construction and pinned by a discriminating spy
test (not a tautology — each test FAILED at the pre-carve HEAD and
flipped to PASS in its landing commit).

One walk (S6.4)
---------------

ONE d-generic frame —
:meth:`orpheus.sn.loss_representation._OctantWalk._interior_walk` —
serves **both** the sweep and the matvec for every multi-D
representation. For each octant it projects the quadrature octant to its
in-plane signs, branches the pure-z degenerate octants, derives the
per-axis in/out domain faces, reads the octant's inflow, runs the
interior traversal, and sheds the outflow. The two directions fork
**only** at:

* the **cell kernel** — the per-octant interior traversal the calling
  representation supplies (the window's frontier walk, the scan-march's
  row-march, the oracle's full cochain), in its solve
  (:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch`)
  or apply
  (:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch`)
  direction; and
* the **emit policy** — what the direction accumulates (the sweep's
  angular/moment output via ``_SweepEmit``; the matvec's
  :math:`(L+C)\psi` bulk plus the O.4b boundary defect).

.. admonition:: The direction is an OBJECT, never a boolean
   :class: warning

   The solve/apply fork is carried by **kernel and emit objects**
   (``_CellSolve`` / ``_CellResidual``,
   :class:`~orpheus.sn.loss_representation._SweepEmit`), never by an
   ``is_solve`` flag threaded down the walk. A boolean direction flag is
   the coding-elegance Smell #3 anti-pattern (special-case dispatch
   masquerading as abstraction). ``test_one_octant_walk.py`` carries an
   **AST tripwire** that parses ``_OctantWalk``'s source and fails if any
   ``is_solve`` / ``is_apply`` / ``is_matvec`` name appears — so the
   carve cannot silently degrade back into the flag pattern.

The discriminating test
``test_sweep_and_loss_action_hit_one_octant_walk`` spies the call-time
``self`` of ``_interior_walk`` and asserts that both ``L.apply`` and
``A.solve`` on a 2-D Cartesian mesh exercise it. The three-layer stack
beneath the walk (storage walk / level operation / pure kernel pair) is
documented at :ref:`sweep-dispatch-relayering`; the graph layer
(:class:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.for_shape`,
per-shape ``lru_cache`` of immutable ``MappingProxyType`` octant→DAG
maps) is family-owned, so the mesh stays pure geometry.

One instance (S6.5)
-------------------

The operator holds **one** representation instance —
:attr:`StreamingOperator.loss_representation
<orpheus.sn.operators.streaming.StreamingOperator.loss_representation>` (a
``cached_property`` = ``default_for(sn_mesh, spatial_closure,
angular_closure)``, the operator's own three fields since P4.9b) —
consumed by:

* :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply` (the matvec
  :math:`(L+C)\psi`);
* :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve` (the forward
  substitution :math:`(L+C)^{-1}q`), via the delegating
  :attr:`StreamingCollisionOperator.loss_representation
  <orpheus.sn.operators.streaming.StreamingCollisionOperator.loss_representation>`
  property; and
* the boundary Gauss–Seidel resolvent.

Because representations are stateless frozen dataclasses, "one instance"
is a **structural type-fact goal** — the L21 invariant becomes
construction-enforced — not a performance fix (the per-shape DAG cache
already de-duplicated the heavy state). The discriminating test
``test_apply_and_solve_share_one_representation_instance`` spies the
call-time ``self`` of *both* doors and asserts object identity; it
**failed pre-S6.5** (the doors each called ``default_for`` independently
— two distinct frozen-dataclass ids per solve), and flips PASS once both
consume the operator's one instance.

.. note::

   **Deliberate scope boundary.** The operator-free ``transport_sweep``
   entry RETIRED at step 6 (R-6.1) with its one production caller: the
   ``solve_sn`` post-convergence reconstruction re-routed onto the
   within-group implicit operator ``M`` (``build_within_group_system(...).implicit_operator`` —
   6a), so every remaining sweep runs through an operator's ONE
   representation instance and the one-instance theorem now covers all
   doors.


.. _loss-rep-orientation-two-frames:

The orientation axis — the adjoint completes the 2×2
====================================================

The one-walk theorem above unified the two **forward** faces —
:math:`SOLVE` and :math:`APPLY` — for the multi-D representations. It
said nothing about the *adjoint* direction, because when it landed
(#222 S6.4) the transpose did not yet exist. Issue #280 completes the
picture: it adds the missing inner solve
:math:`(L+C)^{-\mathsf T}b` and, in doing so, resolves
:math:`(L+C)`'s actions into a clean **2×2 with a non-free third
axis**, from which a *shared, coherent adjoint frame* falls out the way
the forward frame already had one.

The four faces are ``{forward, transpose} × {solve, apply}``:

.. list-table:: The four faces of :math:`(L+C)`
   :header-rows: 1
   :widths: 20 40 40

   * -
     - **solve** (invert the triangular cell system)
     - **apply** (the matvec / row action)
   * - **forward** (topological DAG order)
     - :meth:`~orpheus.sn.loss_representation.LossRepresentation.sweep`
       — forward substitution :math:`(L+C)^{-1}q`
     - :meth:`~orpheus.sn.loss_representation.LossRepresentation.loss_action`
       — :math:`(L+C)\psi`
   * - **transpose** (reversed DAG order + face in↔out swap)
     - :meth:`~orpheus.sn.loss_representation.LossRepresentation.sweep_transpose`
       — :math:`(L+C)^{-\mathsf T}b` (**#280 2.5b** — the
       previously-empty cell)
     - :meth:`~orpheus.sn.loss_representation.LossRepresentation.loss_action_transpose`
       — :math:`(L+C)^{\mathsf T}\varphi`

The reverse-DAG transpose is verified against the *same* per-ordinate
cell DAG the forward walks
(:meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.dag_walk_cell_indices`):
the adjoint walks ``dag_walk_cell_indices(direction_sign=s)`` and then
``reversed(cells)``. This is the **discrete Euclidean transpose**, NOT
:math:`\mu`-reversal and NOT the continuous adjoint (see the warning
below — it is the #1 landmine of this whole carve).

Why it fragmented — the honest story
------------------------------------

The forward unification (#222 S6.4) *was* real: it genuinely fused
``{sweep, matvec}`` for the multi-D representations into one
``_OctantWalk._interior_walk`` (the one-walk theorem above). But its
scope was ``{forward} × {multi-D}`` and it **stopped there** — the
adjoint did not exist yet, and the 1-D solve is a parallel-prefix scan
that does not fit a per-cell octant frame. The adjoint
(:meth:`~orpheus.sn.loss_representation.LossRepresentation.loss_action_transpose`,
Wave O / #208) and the 1-D matvec (#206 Phase C) arrived **later, as
verbatim relocations of pre-existing procedural bodies**, each a
standalone method because there was no shared frame to inject them
into. So the fragmentation was **a real-but-partial unification that
was never extended across the adjoint axis or the 1-D scan band** —
``L21`` ("matvec ≡ sweep, one operator") was *achieved* for multi-D
forward but only *claimed* for the 1-D walk and the adjoint direction.

.. _loss-rep-two-frames:

The two frames — each shared across orientation
-----------------------------------------------

The clean-2×2 hypothesis is overturned by a **third axis — execution
strategy** ``{scan, cell-loop, wavefront-graph}`` — but that axis is
**not free**; it is determined by ``(kernel, dimensionality)``:

* **solve → scan.** The 1-D forward
  :meth:`~orpheus.sn.loss_representation.LossRepresentation.sweep`
  (``_OneDimScanWalk._run``) is a **Blelloch parallel-prefix scan**:
  inverting the linear cell recurrence is *what a scan is for*. Forcing
  it into a per-cell loop would lose the cache vectorisation the
  precomputed two-stratum cache exists to provide (a measured
  regression; the scan-vs-wavefront wrong-fit lesson).
* **apply → cell-loop.** The 1-D
  :meth:`~orpheus.sn.loss_representation.LossRepresentation.loss_action`
  (``_apply_walk``) and its adjoint are **per-cell loops** over
  ``dag_walk_cell_indices`` — a matvec has a concrete probe
  :math:`\bar\psi`, so it recomputes coefficients cell-by-cell.
* **multi-D → wavefront-graph.** ``_OctantWalk._interior_walk`` over
  the anti-diagonal
  :class:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph`.

.. admonition:: The coherence axis is ORIENTATION, not kernel
   :class: important

   Because the execution strategy is pinned by ``(kernel, dim)``, the
   forward ``_run`` (scan) and ``_apply_walk`` (loop) are **NOT
   redundant twins to merge** — they are *solve-scan* vs *apply-loop*,
   legitimately different algorithms. The genuine unification runs
   along the **orientation** axis (forward ↔ adjoint), the one *free*
   axis: **two frames, each shared across orientation**.

The **apply-loop frame** =
``{_apply_walk (forward matvec), loss_action_transpose (adjoint
matvec)}`` — ONE orientation-parametrised per-cell loop over the one
DAG, forking **only** at orientation:

* reversed cell order (``reversed(cells)``);
* the boundary **in↔out swap** (forward reads inflow / writes outflow;
  the adjoint reads *outflow* cotangents / writes *inflow* cotangents);
* the curvilinear **Carlson coupled-pole mirror routing** (the
  :math:`\pm 1` sweeps couple cross-wise through the x-mirror pairing
  ``_ensure_pole_mirror`` derives from ``ordinate_permutation``; the
  adjoint reverses that cross-direction dependency); and
* the **second triangular factor** —
  :meth:`~orpheus.sn.angular.closure.MorelMontryAngularSweep.angular_adjoint`,
  the reverse of the Morel–Montry angular recurrence (zero for slab).

Phase 2.5a landed this frame, **bit-identical in BOTH orientations**
(the frozen ``walk_matvec_*`` snapshots are the anchor;
``tests/sn/sweep/core/test_one_dim_loop_walk.py`` carries a wrap-spy
proving both orientations execute the *one* frame, plus an **AST
tripwire** banning ``is_adjoint`` / ``is_forward`` / ``is_transpose``
/ ``is_reverse`` identifiers — orientation is an **OBJECT**, never a
boolean flag, the same discipline the one-walk theorem enforces for the
solve/apply fork).

The **solve-scan frame** =
``{_run (forward solve — the Blelloch scan), sweep_transpose (adjoint
solve — the reverse-scan)}``. Phase 2.5b landed
:meth:`~orpheus.sn.loss_representation.LossRepresentation.sweep_transpose`
as the **coherent reverse-scan** — *not* a reverse-loop bolted onto the
apply path. Its substrate is
:func:`~orpheus.sn.sweep.scan.ordinate_scan_transpose`, the affine
scan's own adjoint (an ``ordinate_scan`` on reversed-shifted
coefficients), so it inherits ``_run``'s Blelloch pair-monoid substrate
and denormal robustness — one source of truth, not a duplicated reverse
loop. The seed-block sibling is
:func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_transpose`
(the Hébert §3.9.4 starting-direction march's adjoint).

The orientation × kernel × execution taxonomy
---------------------------------------------

Reading the axes together: **orientation** is free (forward and
transpose share one frame within each band); **execution strategy** is
the non-free third axis, pinned by ``(kernel, dimensionality)``. These
bands stay distinct by *algorithmic necessity*, not debt:

.. list-table::
   :header-rows: 1
   :widths: 26 26 26 22

   * - Frame (kernel × dim)
     - Forward
     - Transpose (adjoint)
     - Execution
   * - **solve, 1-D**
     - ``_run`` (``sweep``)
     - ``_run_transpose`` (``sweep_transpose``)
     - prefix-**scan** / reverse-scan
   * - **apply, 1-D**
     - ``_apply_walk`` (``loss_action``)
     - ``loss_action_transpose``
     - per-cell **loop** over ``dag_walk_cell_indices``
   * - **solve/apply, multi-D**
     - ``_OctantWalk._interior_walk``
     - the mirror-octant reverses (#310 C3/C4/C5): full-cochain ORACLE +
       windowed PRODUCTION ``loss_action_transpose`` (the UNCHANGED
       ``walk_full`` / ``walk_windowed`` over ``graph(−signs_eff)`` —
       scheme-complete: DD scalar faces AND the LD moment-tailed
       cochain, any ``d``), and the row-march reverse (``ScanMarch``,
       reversed rows + ``_x_scan_faces_transpose``)
     - anti-diagonal **wavefront-graph** / row-march

The empty-looking symmetry of the 2×2 is thus *honest*: the coherence
axis (orientation) is shared everywhere it is implemented, and the
execution axis is a genuine structural fork that a unified walk must
**express**, not erase.

.. admonition:: The transpose is the discrete EUCLIDEAN transpose
   :class: warning

   :meth:`~orpheus.sn.loss_representation.LossRepresentation.sweep_transpose`
   and
   :meth:`~orpheus.sn.loss_representation.LossRepresentation.loss_action_transpose`
   compute :math:`(L+C)^{\mathsf T}` — **reversed DAG order + face
   in↔out swap over the SAME per-ordinate DAG**. This is **NOT**
   :math:`\mu`-reversal (reflecting the angular quadrature) and **NOT**
   the continuous transport adjoint. ``direction_sign`` picks the
   physical octant DAG; *reversing the within-octant cell order* is
   what gives the transpose. The physical Hilbert **G-adjoint**
   :math:`A^{*} = G_V^{-1}A^{\mathsf T}G_W` rides *on top of* this bare
   Euclidean transpose through the function spaces' metrics (see the
   swap law below) — the sweep itself carries **no metric code**. This
   is the #1 landmine of the carve; conflating the three transposes is
   the fastest way to a plausible-but-wrong adjoint.

The deferral ledger — what still raises, by design
--------------------------------------------------

The unified walk must **express** a partial/deferred grid, not a full
symmetric 2×2. Every entry below is a typed, loud
``NotImplementedError`` that the unification **preserves** — an honest
deferral, not a coverage gap:

* :class:`~orpheus.sn.operators.scheduled_invertible.ScheduledInvertibleOperator`
  **transpose** — the schedule-folded ``M = (L+C-B_lower)`` reverse-scan
  is the #280 sibling deferral (no consumer), so a
  :class:`~orpheus.sn.operators.sweep_operator.SweepOperator` over it
  stays non-adjointable.

That is the WHOLE ledger for the adjoint *matvec*: since #310 C5 the
``loss_action_transpose`` grid is complete over every registered
``scheme × representation`` pairing that can be constructed (the only
structural absence — ``ScanMarch`` × LD-2D — is a *forward*
construction refusal, the facewise supports-gate, so there is no
forward to transpose; pinned by its own gate row).  The two-factor
predicate machinery (``has_transpose_kernel ∧ has_transpose_walk``)
stays armed so a FUTURE unregistered-kernel scheme or reverse-less
representation refuses eagerly at ``.H`` and loudly at apply.

Retired from this ledger: the **LD-slab adjoint** entry closed at
#310 C2 (the registered LD batch VJP + the moment-tailed 1-D reverse —
the ``loss_action_transpose`` moment arm now carries the trailing
:math:`2^d` spatial-moment axis the old entry denied); the **multi-D
Cartesian PRODUCTION adjoint** entry closed at #310 C4 — the windowed
reverse rides the mirror graph's own ``window_plan`` (pinned
``window ≡ full`` bit-identical against the C3 oracle), the
``ScanMarch`` row-march reverse rides ``_x_scan_faces_transpose`` +
the backwards transverse chain (pinned principled-equivalent + its own
dense-``Mᵀ``/pairing rows, plus a d=3 spine row); and the **LD-2D
reverse** entry closed at #310 C5 — the moment-tailed face cochain
reverses through the SAME mirror-octant frame (the C2 kernel VJP was
d-generic all along; C5 landed the gate battery: LD-2D dense-``Mᵀ`` +
the assembled-``Mᵀ`` keystone + reverse ``window ≡ full`` at
``n_face_moments = 2`` + the moment-drop tooth with its EXACT
slope-free blindness control + the cross-moment ``x̂ŷ`` frame-sign
tooth whose deviation splits exactly by octant backward-count parity),
whereupon the one deferral predicate
``_multi_d_multi_moment_reverse_deferred`` — minted at C4 precisely so
the trait and the apply-guard could not drift while the gap lived —
was RETIRED with both flip-safety faces moving together by
construction.

.. _loss-rep-inverse-adjoint-swap:

The inverse-adjoint wiring — the swap law
-----------------------------------------

Phase 2.5c surfaces the reverse-scan through the operator algebra
without adding a single line of new solve machinery, by reading the
#226 inverse-as-operator taxonomy literally.
:class:`~orpheus.sn.operators.sweep_operator.SweepOperator` **is**
:math:`(L+C)^{-1} = A^{-1}` (the object
:meth:`(L+C).inverse() <orpheus.sn.operators.streaming.StreamingCollisionOperator.inverse>`
returns). Since the composite is endomorphic, its Euclidean transpose
is :math:`(A^{-1})^{\mathsf T} = (A^{\mathsf T})^{-1}` — so
:meth:`SweepOperator.apply_transpose <orpheus.sn.operators.sweep_operator.SweepOperator.apply_transpose>`
is *exactly* the inner's
:meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve_transpose`
(the 2.5b reverse-scan). Nothing else is needed.

The consequence is the **swap law**, an *object identity of the
algebra* rather than a numerical coincidence of two implementations:

.. math::
   :label: loss-rep-adjoint-inverse-swap

   (A^{*})^{-1} \;=\; (A^{-1})^{*}
   \qquad\Longleftrightarrow\qquad
   \texttt{A.H.inverse()} \;\equiv\; \texttt{A.inverse().H}.

.. (vv-status rationale) Structural object identity of the operator algebra
   (A.H.inverse() == A.inverse().H). Foundation-gated by the G-metric
   reciprocity pin (test_g_adjoint_reciprocity), verified structurally
   through the forward matvec, not by asserting the two sides agree.
.. vv-status: loss-rep-adjoint-inverse-swap documented

.. implements:: loss-rep-adjoint-inverse-swap
   :by: orpheus.numerics.operator.AdjointOperator.inverse

   The right-hand side spelled literally — the method's terminal
   statement is ``return inner_inverse.H``. This is what makes the swap
   law an **object identity** rather than a theorem two implementations
   happen to satisfy: there is no second code path computing
   :math:`(A^{*})^{-1}` that could drift from :math:`(A^{-1})^{*}`,
   because the wrapper *is* ``A.H`` and its ``inverse()`` routes to
   ``A.inverse().H`` by construction. The two guard clauses above that
   statement refuse — with
   :class:`~orpheus.numerics.operator.NotInvertible`, naming the swap law
   — rather than returning a weaker object, so the law never holds only
   approximately.

It holds by construction because
:meth:`AdjointOperator.inverse() <orpheus.numerics.operator.AdjointOperator.inverse>`
returns ``self.inner.inverse().H`` — the wrapper *is* ``A.H``, so its
inverse routes to ``A.inverse().H``. The **metric** adjoint-solve then
falls out **for free**:

.. math::
   :label: loss-rep-metric-adjoint-solve

   \bigl(A^{-1}\bigr)^{*} b
     \;=\; G^{+}\,\bigl(A^{-1}\bigr)^{\mathsf T}\!\bigl(G\,b\bigr)
     \;=\; G^{+}\;\texttt{solve\_transpose}\!\bigl(G\,b\bigr)

.. (vv-status rationale) Structural identity (the metric adjoint-solve
   conjugation, which falls out for free). Foundation-gated by the same
   G-metric reciprocity pin (test_g_adjoint_reciprocity).
.. vv-status: loss-rep-metric-adjoint-solve documented

(the composite is endomorphic, so its domain and codomain metrics are
one and the same :math:`G` — the direct-sum bulk ⊕ trace ⊕ seed metric,
pseudo-inverted on the singular partial-current trace), because the
existing
:meth:`AdjointOperator.apply <orpheus.numerics.operator.AdjointOperator.apply>`
already conjugates ``inner.apply_transpose`` with the domain/codomain
metrics. So there is **no** ``AdjointOperator.solve`` and **no** metric
code in the sweep — the original A3 plan's "Deliverable 3" (a separate
metric transpose-solve) *dissolved*. The honest two-factor predicates
guard it (ruling R11):

* :attr:`SweepOperator.is_adjointable <orpheus.sn.operators.sweep_operator.SweepOperator.is_adjointable>`
  ``= isinstance(inner, StreamingCollisionOperator) and inner.is_adjointable``
  (the schedule-folded arm has no ``solve_transpose`` → stays
  non-adjointable; LD / multi-D Cartesian defer via
  ``inner.is_adjointable``);
* :attr:`AdjointOperator.is_invertible <orpheus.numerics.operator.AdjointOperator.is_invertible>`
  ``= invertible(inner) and adjointable(inner.inverse())`` — a name
  earns its invariant: ``.inverse()`` cannot promise more than
  ``inner.inverse().H`` can deliver.

The swap law is verified structurally, not by asserting the two sides
agree: the load-bearing gate is a **G-metric reciprocity pin** using
only the *forward* matvec — for :math:`x = A.H.inverse().apply(b)`,
:math:`\langle A\psi, x\rangle_G = \langle\psi, b\rangle_G` — which
never calls the transpose-solve path, so it is structurally independent
of the code it checks. The dense :math:`(L+C)^{-\mathsf T}` oracle
(built from the forward ``apply`` alone via ``to_flat``/``from_flat``)
is the keystone catcher against a bug copied into both
``apply_transpose`` and ``sweep_transpose``.

.. implements:: loss-rep-metric-adjoint-solve
   :by: orpheus.numerics.operator.AdjointOperator.apply

   The conjugation, verbatim: apply the codomain metric, call the inner
   operator's ``apply_transpose``, apply the domain's inverse metric —
   :math:`G^{+}_V \odot \mathrm{apply\_transpose}(G_W \odot y)`. The
   metrics are fetched from the function spaces rather than stored, so
   the one wrapper serves a flat ndarray metric and the composite bulk
   :math:`\oplus` trace :math:`\oplus` seed metric alike, pseudo-inverting
   on the singular partial-current trace. This is why the equation's
   left-hand side needs no ``AdjointOperator.solve``: the adjoint-solve
   is the adjoint-**apply** of the inverse operator.

.. implements:: loss-rep-metric-adjoint-solve
   :by: orpheus.sn.operators.sweep_operator.SweepOperator.apply_transpose

   Supplies the equation's second equality, :math:`(A^{-1})^{\mathsf T} =
   \texttt{solve\_transpose}`. Its body is ``return
   self.inner.solve_transpose(b)`` behind an ``isinstance`` narrowing
   that raises :class:`~orpheus.numerics.operator.MissingAdjoint` for the
   schedule-folded composite. Without this the conjugation above would
   have nothing to conjugate — the two declarations are the two factors
   of one chain, not two views of one symbol.

.. note::

   **The source-subspace subtlety (#284, carried into the transpose).**
   ``solve`` *computes* the outflow slots (the slab boundary-outflow /
   the sphere seed ``corner_out``) while ``apply`` treats them as free
   DOFs, so ``solve = (L+C)⁻¹`` only on the **source subspace**
   (``M_solve · M_apply = I`` on bulk rows to :math:`\sim 5\times
   10^{-16}`; the outflow slots differ :math:`\mathcal O(1)`).
   ``sweep_transpose`` is the *faithful* transpose of the solve — it
   matches the apply-oracle :math:`M^{-\mathsf T}` on every
   source-carried slot and deviates **only** on the outflow column
   (where ``M_solve``'s outflow column is provably zero). Every
   reverse-solve gate therefore compares source-carried slots
   (bulk ⊕ seed cells ⊕ inflow corner), NOT the full ``to_flat``.

.. _loss-rep-initial-guess-warm-start:

The ``initial_guess`` / warm-start architecture split
-----------------------------------------------------

Wiring the adjoint made the last dependence on the old seed lag visible,
and 2.5c retired it. The principle is a clean layering:

* **A direct, exact inverse has nothing to seed.**
  :class:`~orpheus.sn.operators.sweep_operator.SweepOperator` and
  :meth:`StreamingCollisionOperator.solve <orpheus.sn.operators.streaming.StreamingCollisionOperator.solve>`
  **accept-and-drop** the canonical seeded ``apply(x, /, *,
  initial_guess=None)`` keyword (#285). The keyword stays uniform across
  the inverse family (``SupportsSeededApply``) so a driver threads the
  previous iterate the same way every step — but on an exact sweep it is
  discarded, exactly as
  :class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
  and :class:`~orpheus.sn.operators.windowing.WindowedSweep` discard it.
* **The genuine warm start lives at the iteration layer.** A learned or
  previous-outer :math:`x_0` — the "3 vs 30 iterations" lever, real
  because the solution manifold is lower-dimensional than the full
  problem space — belongs to
  :meth:`SourceIteration.solve(initial_guess=x_0) <orpheus.numerics.iteration.SourceIteration.solve>`
  and the iterative
  :class:`~orpheus.numerics.green_operator.GreenOperator` (its
  Richardson splitting), never on a direct sweep.

This split is only *possible* because Issue #282 route (a) (Phase 2.5d)
made the curvilinear :math:`\psi_{1/2}` starting direction a
**direct** computation from the source
(:func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`),
retiring the old two-point extrapolation of the *previous* iterate —
which was precisely the last thing the SN sweep *used* an
``initial_guess`` for.

.. warning::

   Route (a) is justified **structurally** — it is the honest
   single-pass direct inverse (spherical cold within-group residual
   :math:`5.18\times10^5 \to 2.5\times10^{-16}`), which is what makes
   ``sweep_transpose`` and the assembled triangularity certificate
   well-posed. It is **NOT** justified by angular accuracy: a seed *is*
   an angular closure, so it changes the :math:`\mathcal O(N)`
   truncation, and at the tests' GL8 order the old extrapolated seed is
   in fact marginally *closer* to the :math:`N\to\infty` limit. Do NOT
   read the re-baseline as "more accurate" — the eigenvalue re-pose is
   certified by an angular-order :math:`N`-sweep at fixed mesh (both
   seeds converge to the same transport eigenvalue), never by an MMS
   convergence rate (which is blind to the seed by construction).


.. _loss-rep-facefield-codim1:

The starting-direction ψ½ as a codim-1 face object
==================================================

The within-group phase space spans three subspaces —

.. math::

   V \;=\; V_{\rm bulk} \,\oplus\, V_{\rm trace} \,\oplus\, V_{\rm sd}

(the same direct sum whose metric :math:`G` is
:eq:`loss-rep-metric-adjoint-solve`; the canonical statement is the
augmented composite :eq:`sn-direct-seed-augmented-composite` on the discrete-
ordinates page).  The first two summands are familiar — the cell-centred
bulk and the spatial boundary trace, together the 2-block
:class:`~orpheus.transport.full_field.FullField` (System A) every
representation on this page acts on.  This section is about the third,
:math:`V_{\rm sd}`, the starting-direction ψ½ block (#282 route (a); the
physics and the direct solve are on
:ref:`sn-direct-seed-solve`).  Since the Phase B.2d
eviction :math:`V_{\rm sd}` is **not** a block on ``FullField`` — it is
**System B**, its own 2-block composite coupled to System A as a
:class:`~orpheus.numerics.coupled_system.CoupledField`, and the walk
consumes its leaves through **explicit kwargs** (the six-signature
protocol below), never as a third summand of the carrier.  The
**architectural** fact this section documents survives the eviction
unchanged: ψ½ is a **codim-1 face object**, a sibling of the spatial
trace, and it does **not** belong in the cell-centred bulk.

Current truth — a face object, not a bulk passenger
---------------------------------------------------

:math:`V_{\rm sd}` is the angular flux :math:`\psi_{1/2}` at the
:math:`\mu = \pm 1` **angular edge** of each carrying level.  Three facts
make it a face object — sibling to the spatial boundary trace, distinct
from the bulk:

* **It lives on a codim-1 locus.**  The bulk is codim-0 (cell centres,
  the full :math:`(N, n_g, n_x)` phase space); the spatial trace is
  codim-1 (the :math:`r`-faces); ψ½ is codim-1 too — the angular-domain
  edge :math:`\mu = \mu_{\rm start}`.  Its storage — the split
  :class:`~orpheus.numerics.spaces.radial_characteristic_space.RadialCharacteristicInteriorSpace`
  (cells) and
  :class:`~orpheus.numerics.spaces.radial_characteristic_space.RadialCharacteristicBoundarySpace`
  (the :math:`r = R` corner) — deliberately mirrors the spatial trace's
  flat-buffer :class:`~orpheus.numerics.face_layout.FaceLayout` discipline
  (typed ``(level, sign)`` views in place of ``FaceLabel`` views).
* **Its metric is a STATE metric, not the through-flux.**  ψ½ is a
  first-class radial state field, so its Hilbert metric is the SPD radial
  cell volume :math:`G_{\rm sd} = V_{\rm cell}` (mirroring the bulk's
  :math:`V\,w`, restricted to the :math:`\mu = \pm 1` ray).  The
  through-flux coefficient :math:`(1-\mu^2)|_{\rm pole} = 0` that vanishes
  at the pole is an *operator coefficient* (the angular-redistribution
  strength — the straight-characteristic physics), **not** the state
  metric; conflating the two was the retired "ghost metric" bug
  (ERR-067).  The full M1/M2/M3 distinction is on
  :ref:`the discrete-ordinates page <sn-direct-seed-pole-state-metric>`.
* **It is present per level by the R12a predicate**
  (:ref:`sn-direct-seed-r12a`; the level's march start consumes an
  independent seed iff it is neither an edge-node start nor
  η-degenerate — the two ``MarchStart`` facts of Q5.4, which retired
  the earlier :math:`\tau_{\rm raw} \in (0,1)` float encoding), **not**
  per geometry — sphere-GL carries one block; a σ_y-folded cylinder
  rule carries one per level; full-circle and level-symmetric cylinder
  rules and every Cartesian mesh carry none (and since Q5.6.3 the two
  non-carrying cylinder classes are refused at ``SNMesh`` admission,
  so every *constructible* curvilinear mesh carries on every level).

The clean-bulk consequence is the load-bearing architecture.  The bulk
:math:`V_{\rm bulk}` is what homogenization, condensation, and moment
extraction consume, and they must see an **untouched** cell-centred
field.  Folding ψ½ into the bulk — for instance via a pole-node
Gauss–Lobatto quadrature, which is *numerically affordable*
(:ref:`sn-direct-seed-lobatto-study`) — would make the bulk a *mixed* field
carrying an inert, redistribution-special pole passenger that every bulk
consumer would have to demux back out.  Keeping ψ½ a **separate** face
object is the Cardinal-Rule-2 choice that keeps the bulk clean; the
Lobatto study is what makes that separation a *chosen* architecture
rather than a forced one.

.. admonition:: Design direction — mesh-derived presence via a ``PhaseSpaceCarrier`` protocol (PARTIALLY built)
   :class: note

   Parts of this design direction have **landed**; the rest is recorded so
   a future session inherits the reasoning.  The **structural codim-1
   parent** :class:`~orpheus.transport.fields._bases.FaceField` now
   **exists** (commit ``4081c0d``) — the split ψ½ leaves
   (``RadialCharacteristicInteriorField`` / ``RadialCharacteristicBoundaryField``,
   the ``interior ⊕ boundary`` loci) and the spatial-trace boundary
   families are its subclasses, sharing the flat-buffer ``FaceLayout``
   discipline instead of hand-copying it.  (Since 4e-e1b the name
   ``RadialCharacteristicField`` denotes the **composite** that binds those
   two leaves — System B's carrier — **not** a ``FaceField`` subclass.)  The
   :func:`~orpheus.numerics.face_layout.face_streaming_normal` measure also
   exists, but as the spatial-trace partial-current metric **only** (see
   the refuted-unification note below).  The per-leaf **metrics are correct
   and landed**: the spatial trace carries
   :math:`\lvert\Omega\cdot n\rvert\,w` and the ψ½ block its SPD state
   metric :math:`G_{\rm sd} = V_{\rm cell}`.  And the Phase B.2d eviction
   already reached the design's **unconstructable-by-design presence** goal
   — by a different route: ψ½ is System B (its own composite), so a
   presence-mismatched carrier is a *type error*, not a runtime-checked
   ``Optional`` (the six-signature protocol below replaces the old
   presence-reconciling guards).  What is **not** built is the
   ``PhaseSpaceCarrier`` protocol that would let the mesh *enumerate* its
   phase-space blocks and the composite mirror them automatically.  The
   durable synthesis is ``.claude/plans/archive/facefield_codim1_design.md`` (§5),
   read against the ERR-067 metric correction.

   **The codim-1 parent (landed).**  The typed-field hierarchy
   (:mod:`orpheus.transport.fields`) now shares the **codim-1 parent**
   :class:`~orpheus.transport.fields._bases.FaceField` off
   :class:`~orpheus.numerics.field.Field`: the spatial-trace family
   (:class:`~orpheus.transport.fields._bases.BoundaryField`) and the
   **split** ψ½ family
   (:class:`~orpheus.transport.fields._bases.RadialCharacteristicInteriorField`
   / :class:`~orpheus.transport.fields._bases.RadialCharacteristicBoundaryField`)
   are all its subclasses, inheriting the flat-buffer ``FaceLayout``
   discipline once instead of hand-copying it.  The two 2-block **carriers**
   (System A's ``FullField``, System B's ``RadialCharacteristicField``) are
   ``Composite`` products of those leaves — **not** ``FaceField``
   subclasses::

       Field
       ├── BulkField    — codim-0 (cell centres);   metric = per-leaf  V·w
       └── FaceField    — codim-1 (faces/edges);    STRUCTURE only (flat layout + presence)
            ├── AngularBoundaryField / ScalarBoundaryField            (spatial faces)  metric |Ω·n|·w
            └── RadialCharacteristicInteriorField / …BoundaryField     (ψ½ split, angular edges)  metric V_cell (SPD)

       Composite[Interior, Boundary]   — a two-block product of Fields (NOT a Field)
       ├── FullField                 = Composite[BulkField, BoundaryField]                        (System A)
       └── RadialCharacteristicField  = Composite[…InteriorField, …BoundaryField]   (System B — ψ½ carrier, reminted 4e-e1b)

   ``FaceField`` owns the ``FaceLayout`` flat-buffer discipline **once**,
   so the face families are no longer siblings-by-accident.  It unifies
   **structure only** — the metric stays per-leaf (next note).

   **The metric does NOT unify — it is per-leaf (refuted unification,
   ERR-067).**  An earlier draft proposed collapsing both codim-1 metrics
   into one ``face_streaming_normal`` kernel, reading the pole's angular
   through-flux coefficient :math:`(1-\mu^2)|_{\rm pole} = 0` as the ψ½
   block's Hilbert metric (the "``G_sd == 0`` at 0 ULP" win).  That is
   **refuted**: :math:`(1-\mu^2)|_{\rm pole}` is an *operator coefficient*,
   not a state metric, and the correct ψ½ metric is the SPD
   :math:`G_{\rm sd} = V_{\rm cell}` (:ref:`sn-direct-seed-pole-state-metric`).
   The through-flux equals the state metric only when the face's operator
   self-block is trivial — true for the spatial trace (a restriction map),
   false for ψ½ (a banded radial transport operator :math:`A_{\rm ss}`).
   So :func:`~orpheus.numerics.face_layout.face_streaming_normal` stays the
   spatial-trace measure **only**, and the ``FaceField`` unification is
   **structural** (flat layout + presence), **not** metric: the metric is
   a per-leaf property (spatial trace → partial current; pole →
   :math:`V_{\rm cell}`), exactly as the bulk's metric is its own per-leaf
   :math:`V\,w`.

   **Mesh-enumerated blocks (still planned).**  The remaining, unbuilt
   half: the composite's block list would be *the mesh's phase-space DOF
   structure, reified* — the mesh enumerates its blocks (bulk always;
   spatial trace always; angular trace iff :math:`\tau_{\rm raw} \in (0,1)`
   on that level) and the composite mirrors it, via a ``PhaseSpaceCarrier``
   protocol in ``transport`` that ``SNMesh`` (in ``sn``) satisfies — sn →
   transport, never the reverse.  The B.2d eviction already made presence
   **unconstructable-by-design** (a live-ray ``ψ_A`` is a type error); what
   ``PhaseSpaceCarrier`` would add is building the carrier *from* the
   mesh's DOF enumeration directly, rather than the mesh and the composite
   agreeing by separate construction.

.. _sn-loss-rep-ray-decoupled-block:

How the walk sees ψ½ — the ray-decoupled block (step 6)
------------------------------------------------------------

Since the B.2d eviction the ``Optional`` third block and its
presence-vs-mesh reconciliation guards are **retired**: ψ½ is System B,
so a carrier cannot even *hold* a starting-direction block, and there is
nothing to reconcile.  Step 6 (#34) completed the collapse: the walk's
**explicit leaf-kwarg channel is gone**.  Every walk surface — ``sweep``
/ ``sweep_transpose`` / ``loss_action`` / ``loss_action_transpose`` and
the operator entries over them — is the ray-decoupled :math:`(A,A)`
diagonal block, on every mesh, with **no seed parameters at all**:

* the **matvec** substitutes a ZERO seed into the Morel–Montry thread
  (bit-identical to the retired dead-slot arithmetic — the closure reads
  zeros), and the emitted ray rows do not exist (the coupling is the
  explicit :class:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicSeeding`
  grid block);
* the **sweep** starts the M-M recurrence at exactly :math:`\psi_{1/2} = 0`
  on a carrying level (the LC diagonal-block semantics — no fold entry
  exists there);
* the **transpose** surfaces DISCARD the M-M thread cotangent (the zero
  thread was a *fixed input* of the decoupled map, so its cotangent
  propagates nowhere; the :math:`\text{Seeding}^{\mathsf T}` pullback is
  the explicit grid block's
  :meth:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicSeeding.apply_transpose`);
* ``sweep_transpose`` returns the plain pair ``(Q_bar, m_boundary)`` —
  the third ``m_seed`` member retired with the channel.

The JOINT system has exactly ONE spelling: the within-group M grid
``[[LC, Seeding], [None, A_BB]]``
(:class:`~orpheus.numerics.coupled_system.CoupledOperator`, built by
:func:`~orpheus.sn.coupled_system.build_within_group_system`) — its
``apply`` is the block matvec, its ``solve`` the block back-substitution
(System B's :meth:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator.solve`
first, then the walk's bare sweep on
:math:`q_A - \text{Seeding}\,\psi_B`), and its transposed twins mirror.
**Presence is structural**: the grid is 2×2 exactly when the mesh
carries a ray (R12a — the builder's P2 shape), a seedless grid is the
plain ``(L+C)``, and a "joint call" on the wrong geometry is not a
guarded runtime error but an *unspellable program* — the
``RadialCharacteristic*`` constructors refuse seedless meshes and the
carriers cannot be built there.

The retired estate (step 6): the two kwarg pairs
(``radial_characteristic_source``/``radial_characteristic_flux`` forward,
``seed_cot``/``seed_cot_out`` transposed) across the six representation
signatures and the four operator entries; the three-function guard
family (``_require_radial_characteristic`` at 5b, then
``_refuse_radial_characteristic`` and ``_require_leg_pair`` — the
both-or-neither pair law died with the legs it guarded); the walk's fused
``_seed_rows_forward``/``_seed_rows_transpose`` glue (the kernels live
single-sourced in :mod:`orpheus.sn.sweep.psi_half_angle_seed`,
consumed by ``A_BB``); the walk's in-solve System-B engine (the up-front
``A_BB.solve`` + per-level seed read) and its transposed sibling (the
``seed_cot`` augmentation + ``A_BB.solve_transpose``); and the
operator-free module-level ``transport_sweep`` wrapper, whose
carrying-mesh self-derivation existed precisely to feed the fused joint
channel — its callers route the typed surfaces
(:meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve` for
the block, the grid's ``solve`` for the joint march; the eigenvalue
finalize re-routed onto ``build_within_group_system`` at 6a).

The mesh remains the single authority on presence
(:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.radial_characteristic_levels`);
what changed at step 6 is that nothing *checks* against it anymore —
the type system carries the biconditional.

.. _loss-rep-fork-b2:

The S6.9 measurement and the Fork-B2 decision
=============================================

The carve built ``ScanMarch`` as an **opt-in** representation (Fork B1)
with the default unchanged, precisely so the default could be decided on
*measured* numbers rather than a plausibility argument — the governing
principle: *construct each strategy as general as its algorithm
naturally allows; select narrow; specialize only on measured internal
cost.* The S6.9 benchmark
(``derivations/diagnostics/diag_s69_scanmarch_vs_window_bench.py``,
median over repeats, ``python -O``; full table in #222 comment
4683241855) measured ``ScanMarch`` / ``MovingFrontierWindow`` ratios:

.. list-table:: ScanMarch ÷ MovingFrontierWindow (lower is faster / leaner; median, ``-O``)
   :header-rows: 1
   :widths: 30 16 16 16

   * - config
     - sweep
     - matvec
     - peak mem
   * - 24×24 LS4 2g
     - 0.61
     - 0.55
     - 0.98
   * - 48×48 LS4 2g
     - 0.57
     - 0.59
     - 0.99
   * - 96×96 LS4 2g
     - 0.58
     - 0.60
     - 0.99
   * - 48×48 LS8 2g
     - 0.69
     - 0.67
     - 0.99
   * - 48×48 LS16 2g
     - 0.84
     - 0.78
     - 0.99
   * - 48×48 LS8 4g
     - 0.71
     - 0.72
     - 0.99

End-to-end fixed-source (48×48 LS8 2G heterogeneous): **10.5 s vs
12.8 s = 0.82×**.

Three findings shaped the decision:

#. **No memory edge at d=2.** The rolling frontier and the row-march are
   *both* :math:`O(\text{row})` working set at d=2 (``peak ≈ 0.98–0.99``).
   The window's memory advantage only ever held against the
   **full-field oracle** (which is ~1.3–1.4× both). The window's reason
   for being the default — peak-memory — does **not** distinguish it from
   the row-march.

#. **The ScanMarch advantage narrows with angular order** (LS4 ~0.58 →
   LS16 ~0.84). The scan's per-ordinate closed-form work scales with the
   ordinate count :math:`N`, while the level-batched wavefront amortises
   across ordinates per anti-hyperplane — so the gap closes as :math:`N`
   grows. The win is real across the measured range but is not
   asymptotically unbounded.

#. **End-to-end dilution to 0.82×.** The per-sweep kernel win is
   amortised by the scattering / moment-projection / reflect overhead
   that surrounds the within-group solve, so the wall-clock win on a full
   fixed-source solve is smaller than the bare-sweep ratio.

.. note:: Re-measured 2026-08-09 (#347) — the three findings hold, the
   end-to-end **number** does not.

   The benchmark was unrunnable between the 2026-04 restructuring and
   #347 (dead ``orpheus.sn.operator`` import; the ``loss_action`` verb
   had been re-signed by the #257 S8b σ-free carve). Re-run on the
   repointed instrument, same grid, ``python -O``:

   * **Findings 1 and 2 reproduce.** Peak-memory parity ``0.98–1.00``;
     sweep ratios ``0.57–0.83`` against the published ``0.57–0.84``,
     with the same LS4 → LS16 narrowing.
   * ⛔ **Finding 3's value is stale: the end-to-end ratio now measures
     1.00×**, not ``0.82×`` (48×48 LS8 2G het, median of 3 —
     ``38.85 s`` window vs ``38.78 s`` ScanMarch). The finding's
     *direction* is unchanged and in fact sharper — the surrounding
     scattering / moment-projection / reflect work now dilutes the
     kernel win **completely**. Treat ``0.82×`` as a 2026-06-11 reading,
     not a current one. (The absolute times are ~3× the 2026-06 ones;
     that gap is unattributed — this run was not on a quiesced machine,
     so it is not evidence of a regression either way.)
   * ⚠ A full-grid run also produced a single ``2.02×`` sweep outlier at
     96×96 LS4 2g. It is **noise, not a signal**: three focused repeats
     of that one config read ``0.74 / 0.63 / 0.77``. Recorded because
     one back-to-back grid pass is not a reliable instrument at the
     largest config — repeat before believing any single row.

   None of this disturbs the Fork-B2 decision below: it turned on the
   memory parity (finding 1) and on keeping two genuinely-different
   schedules, neither of which is a wall-clock claim.

The decision (Fork-B2, USER, 2026-06-11):

.. admonition:: Fork-B2 decision (verbatim)
   :class: important

   *"There is no need to retire the moving frontier window method. We can
   have multiple methods (this is the whole point of having them
   selectable), as long as they are different methods, with slightly
   different principles, and the code is proper. But you can flip the
   default to ScanMarch."*

So the flip is the **one-line registry reorder** (move ``ScanMarch``
ahead of ``MovingFrontierWindow`` in
:data:`~orpheus.sn.loss_representation.LOSS_REPRESENTATIONS`);
``default_for`` is unchanged (still first-supports-match); the 1-D
default is untouched (``CumprodScan`` stays first). The window is
**kept** as a genuinely-different schedule (anti-diagonal wavefront vs
row-march) over the same lower-triangular operator — its end-to-end
coverage rides the forced-window leg of
``test_scan_march_end_to_end.py`` plus the explicit ``window ≡ full``
oracles. This is the **architecture-over-implementation** discipline
(Cardinal Rule 2): selectability is the whole point, and two proper
methods with different principles are an asset, not redundancy to be
pruned.


.. _loss-rep-bit-vs-principled:

Verification architecture: bit-identity vs principled-equivalence
=================================================================

The carve's verification turns on the vv-principles distinction between
**bit-identity** (an implementation property) and
**principled-equivalence** (a math property), applied at two different
granularities.

Across *schedules*: principled-equivalence
------------------------------------------

Different schedules (CumprodScan vs ScanMarch vs the wavefront) are
**NOT bit-comparable**: the row-march and the anti-diagonal reconstruct
the same cell dependencies *in a different order*, and IEEE-754 addition
is non-associative, so the converged values differ at FP-association.
Demanding ``array_equal`` across schedules would be the *wrong* gate. The
cross-schedule gates are therefore:

* **nulp / absolute-tolerance oracle** —
  ``test_scan_march_equivalence.py`` pins ``ScanMarch`` against the
  unconditionally-stable
  :class:`~orpheus.sn.loss_representation.FullFieldWavefront` oracle
  (G2.c), on ≥2G heterogeneous **anisotropic** configs with **non-square**
  meshes (the x↔y swap moat — failure Mode 2). The tolerance is absolute
  (``rtol=1e-11``, ``atol=1e-12``), not nulp, because near-zero boundary
  shed elements amplify a ~1e-15 absolute difference into a spurious
  ~16000-ULP reading.

* **solver-tol Mode-9 FP-invariance** —
  ``test_scan_march_end_to_end.py`` drives the full production solvers
  with the schedules swapped and asserts the **converged scalar flux**
  is schedule-invariant to solver tolerance (a row-march MUST NOT move
  the bulk :math:`\phi` or :math:`k` — only the per-sweep
  FP-association, which the outer iteration washes out).  ⚠ The
  measurand is the **bulk**, deliberately: G4.b is all-reflective, so
  ``[M]`` :math:`A` is exactly singular there and the boundary *trace*
  is only invariant up to :math:`\ker A` (:ref:`sn-loss-kernel-gauge`);
  the bulk is unaffected because the kernel is pure-trace.
  Per vv-principles failure Mode 9, the
  FP-invariance is verified on configs that *break* the degenerate
  coincidence, never the isotropic-reflective box:

  - **G4.a** — P1-anisotropic + heterogeneous (fuel|moderator) + vacuum
    streaming, with the non-flat degenerate-gate guard;
  - **G4.b** — all-reflective + a level-symmetric cubature, so the
    outflow shed order is load-bearing (the ERR-056 shared-face failure
    class); the honest **d=2 limitation** is stated in the test — the
    full diagonal-cubature shared-face stressor is a d=3 case, deferred
    until ``Mesh3D`` exists (the quadrature is already 3-cosine; the
    schedule-level d=3 shared-face assignment is pinned synthetically by
    ``test_sweep_schedule_nd.py``, C3.6);
  - **G6** — the closed-form
    :math:`\kinf = \lambda_{\max}(A^{-1}F)` (homogeneous, ≥2G — no
    1-group eigenvalue evidence, per the cardinal-rule degeneracy) plus
    SI ≡ Krylov flux-shape agreement on the heterogeneous non-flat 2G
    config.

  The forcing is a **context manager** (a fixture would force the
  *reference* leg too and compare the window to itself), with explicit
  non-vacuity counters asserting the forced path actually ran. Since the
  Fork-B2 flip the polarity is inverted: the *default* leg runs
  ``ScanMarch`` and the *forced* leg runs ``MovingFrontierWindow`` — so
  the window peer gets its end-to-end coverage from the same gate, and
  the FP-invariance claim (being symmetric) is unchanged in meaning.

.. note::

   **MMS reaches the flux-shape layer, not the eigenvalue.** The
   ScanMarch verification matches the vv-principles claim taxonomy: the
   oracle and FP-invariance gates establish *convergence-order* and
   *flux-shape* invariance under the schedule swap; the **eigenvalue**
   anchor is the *closed-form* :math:`\kinf` leg (a structural-independence
   ground), not the SI≡Krylov twin agreement (which is necessary but not
   sufficient — two ORPHEUS paths agreeing is cross-implementation
   agreement, not correctness).

Within a *schedule*: bit-identity
---------------------------------

Two storage *policies* of the **same** schedule must agree to the byte —
they run the identical cell math in the identical level order, differing
only in how much of the cochain they retain. These ARE ``array_equal``
oracles:

* **window ≡ full** — ``test_2d_full_field_oracle.py`` (end-to-end
  sweep + matvec) and ``test_sweep_graph_window_equivalence.py`` (the
  graph-level walk, d=1 / d=2 / synthetic d=3) pin
  :class:`~orpheus.sn.loss_representation.MovingFrontierWindow` exactly
  against :class:`~orpheus.sn.loss_representation.FullFieldWavefront`.

* **the kernel pair is the FP reduction tree of record** —
  ``test_cell_kernel_batch.py::TestKernelSourceOfRecord`` freezes a
  ``sha256`` of the source of
  :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch`
  and
  :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch`.
  Their explicit left fold ``((Σ_t + s_0) + s_1) + …`` is
  bit-identity-load-bearing: an algebraically-equivalent rearrangement
  (a ``sum()`` instead of the fold) passes every value-tolerance test yet
  silently invalidates the 1-ULP regression contract. This is the **one
  legitimate source-hash pin** in the SN stack — the kernels *are* the
  reduction tree every byte-identity anchor inherits from.

* **the affine converged-bytes golden** —
  ``test_affine_carve_bit_identity.py`` *froze a* ``sha256`` of the
  production default's converged ``angular_flux`` / ``scalar_flux``
  bytes. At the Fork-B2 flip the four 2-D hashes were **regenerated in
  the flip commit** (a schedule change shifts the converged bytes at
  FP-association level — principled-equivalent, not a numerics change),
  with a history block naming the output-identity evidence (the G4
  Mode-9 gates + the G2.c nulp oracle); the 1-D slab hashes were
  byte-unchanged (the flip's blast-radius pin). The discipline was
  **regenerate-in-commit, never pin stale**, and it was followed three
  times.

  ⛔ **The digests were retired 2026-08-12 (#333); the module now stores
  VALUES.** The discipline was sound but the *instrument* had a shelf
  life: a zero-change claim pinned against a frozen past is falsified by
  the first legitimate upstream change, and a hash cannot then say
  whether the new value is fine — the magnitude is uncomputable because
  the old values were never kept. Regenerate-in-commit survives as the
  discipline; a source hash is only defensible where the hashed thing is
  itself the contract (the kernel-source pin above), not where it stands
  in for a numerical value.

Structural spies
----------------

The L21 theorems are pinned by the one-walk
(``test_one_octant_walk.py``) and one-instance
(``test_one_representation_instance.py``) call-time-``self`` spies, both
``foundation``-tagged software-structure invariants (no theory
``:label:``), both ``-O``-safe (``pytest.fail``, never bare ``assert``,
so they fire under the canonical ``python -O`` invocation — vv-principles
failure Mode 8).

.. _loss-rep-declared-implementers:

Declared implementers — and the three equations that have none
--------------------------------------------------------------

Every labelled equation above carries one or more ``.. implements::``
directives naming, by dotted path, the symbol that realises it. These are
not decoration. The knowledge graph otherwise **guesses** the
code-to-equation link from shared name tokens, and a declaration stands
that guessing down for the whole equation — so the directives are the
difference between a traceability edge somebody asserted and one a string
match invented.

How wrong the guessing was, measured
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Worth seeing once, because the failure is not the one the phrase
"heuristic link" suggests. Measured on this page's graph immediately
before the declarations landed (2026-08-17, ``.nexus/graph.db`` after a
forced ``-E`` build):

.. list-table:: Inferred ``implements`` edges vs. the declared truth, over the 14 declared equations
   :header-rows: 1
   :widths: 46 18 18 18

   * - Quantity
     - Count
     - Precision
     - Recall
   * - Inferred edges the heuristic wrote
     - 397
     - 1.5 %
     - —
   * - Implementers actually there (declared above)
     - 28
     - —
     - —
   * - …of those, found by the heuristic
     - **6**
     - —
     - **21 %**
   * - …of those, **missed** by the heuristic
     - **22**
     - —
     - —

The mechanism is visible in a single comparison. The guess set for
:eq:`loss-rep-LpC` and the guess set for :eq:`loss-rep-facewise-separable`
are **identical, 23 for 23** — as are the sets for :eq:`loss-rep-leaf-sum`
and :eq:`loss-rep-removal-sigma`. Those four equations have nothing
mathematically in common; one is the definition of the loss operator,
another is a tensor-product separability criterion. What they share is
that every label on this page begins ``loss-rep-``, and the token
``loss``/``representation`` matches every symbol in the
:mod:`orpheus.sn.loss_representation` package. The heuristic had therefore
computed *the membership list of one Python package* and attached it,
unchanged, to seventeen different mathematical statements.

The sharpest consequence is on :eq:`loss-rep-LpC` itself: of its 23
guesses, **zero** are either of its two real implementers. Both
:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` and
:class:`~orpheus.sn.operators.streaming.StreamingOperator` live in
``orpheus.sn.operators.streaming``, so the package-membership set that the
token match produces *cannot contain them by construction*. A guess of
that kind is not a weak claim about the equation; it is not a claim about
the equation at all.

The completeness obligation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because a declaration stands the inference down **per equation**, not per
pair, the author of the first directive on an equation takes on an
obligation: declare *every* implementer. An equation realised in two
places and declared in one shows only the one, since the guess that used
to cover the second has been switched off. The declarations above are
exhaustive by construction, and several equations carry two, three or four
of them for exactly that reason — the DD and LD arms of one affine
extraction, the forward and transpose halves of one single-source claim,
the scheme door and the schedule that folds its transverse term.

.. _loss-rep-unimplemented-labels:

Three true, labelled, implemented-by-nothing statements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three of this page's labels are deliberately left **undeclared**, and each
for a *different* reason. The trio is the durable content of the exercise,
because it says something an inference cannot know: **a statement can be
true, labelled, and implemented by nothing** — and the kinds of statement
for which that is the correct outcome are enumerable.

.. list-table:: Why these three name no implementer
   :header-rows: 1
   :widths: 22 20 58

   * - Equation
     - Class
     - Why nothing implements it
   * - :eq:`loss-rep-leaf-sum`
     - **superseded path**
     - A derivation identity about a route the tree no longer takes.
       :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply`
       OVERRIDES :meth:`~orpheus.numerics.operator.OperatorSum.apply`, so
       the inherited leaf sum is unreachable for :math:`(L+C)`; and since
       #257 S8b :meth:`StreamingOperator.apply <orpheus.sn.operators.streaming.StreamingOperator.apply>`
       never sources :math:`\sigma_t` at all — it is ``loss_action(0, ψ)``
       — so the equation's own underbrace,
       :math:`L.\mathrm{apply}(\psi) = M(\sigma_t)\psi - \sigma_t\psi`,
       describes arithmetic no shipped call performs. The identity is
       still *true* (it is :eq:`loss-rep-affine` read at two
       :math:`\sigma`), which is why the label stays and is not retired;
       what is gone is the code that once evaluated it.
   * - :eq:`loss-rep-removal-sigma`
     - **notation**
     - A definition, not a computation — and there is no production caller
       of the removal form yet (:ref:`loss-rep-removal-form-matvec`).
       :meth:`~orpheus.transport.mesh.material_xs_field.MaterialXSField.foldable_sigma`
       returns only the *subtrahend* :math:`\Sigma_{s,0}^{g\to g}`, never
       the difference. Every site in the tree that *does* form
       :math:`\sigma_t - \Sigma_{s,0}` — **four**, all measured
       2026-08-17 — computes something else with it: three are the **DSA
       low-order removal** :math:`\hat\sigma_R`
       (``orpheus/sn/acceleration/dsa.py`` in production plus its two
       algebra-of-record mirrors in ``orpheus/derivations/discrete/sn/dsa.py``),
       and the fourth is a **capture cross-section**, ``SigC``, built for a
       synthetic 1-group MMS
       :class:`~orpheus.data.macro_xs.mixture.Mixture` in
       ``orpheus/derivations/continuous/mms/sn.py``. That fourth one is the
       instructive one: identical arithmetic, a *material datum* rather
       than an operator diagonal, numerically coincident with
       :math:`\sigma_r` only because the problem has one group. The
       production DSA module states its own scope explicitly — the fold is
       "legitimate HERE and only here", reached through the low-order build
       and **never** the sweep (issue #215). Declaring any of the four would
       attribute the S\ :sub:`N` loss operator's removal diagonal to a
       different operator, at ``confidence = 1.0``.
   * - :eq:`loss-rep-facewise-separable`
     - **declared tag**
     - The separability criterion is never evaluated. It is carried as a
       declared ``ClassVar[bool]`` —
       :attr:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.transverse_coupling_is_facewise`,
       default ``False``, with
       :class:`~orpheus.transport.spatial.diamond.DiamondDifference`
       overriding to ``True`` and
       :class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`
       inheriting the conservative default. Code *reads* the tag (that is
       what :meth:`ScanMarch.supports <orpheus.sn.loss_representation.ScanMarch.supports>`
       gates on), but the tensor-product argument that decides its value
       was made by a human and recorded as a boolean. The implementer of
       the *criterion* is this page.

.. note::

   **The three keep their guesses, and that is a tooling gap, not a
   verdict.** Standing the inference down is a side effect of
   *declaring*, so an equation with nothing to declare has no way to say
   so: these three still carry their full inferred sets. Read the table
   above, not the graph, for those three.

   ⚠ **And their guess count is not even stable** — it grows when the
   page is *improved*. Measured across the build that landed this
   section: writing the table above added ordinary ``:meth:``
   cross-references to
   :meth:`~orpheus.sn.loss_representation.LossRepresentation.streaming_action`
   and
   :meth:`~orpheus.transport.mesh.material_xs_field.MaterialXSField.foldable_sigma`,
   and both symbols promptly appeared as *new inferred implementers* —
   the first on all three undeclared equations, the second on
   :eq:`loss-rep-removal-sigma` specifically, whose label shares the
   token ``sigma``. Citing a symbol in prose in order to explain why it
   is **not** the implementer is enough to make the heuristic name it as
   one. Do not quote a live guess count anywhere; quote the
   pre-declaration measurement in the table above, which is frozen
   history, or re-run the query.

   A first-class way to record "implemented by nothing, for reason R"
   would make this classification machine-readable instead of
   prose-readable, and would stop the count drifting. It is worth having:
   the three classes here — **superseded path**, **notation**, **declared
   tag** — are exactly the cases where an inference's silence is the
   correct answer and its guess is the expensive one.


.. _loss-rep-history:

History and rationale: what was tried, measured, decided
========================================================

Apply and solve use **different** closures by design (historical)
-----------------------------------------------------------------

.. note:: **Superseded architecture (Wave D / early Wave E).**  This
   subsection describes a design in which ``apply`` (a separate
   finite-difference operator) and ``solve`` (the WDD sweep) were
   **two distinct discretisations** of :math:`L+C`.  That split was
   dissolved by the #206 Phase C **matvec ≡ sweep** unification: today
   ``apply`` and ``solve`` run the **one** loss-representation walk
   (:meth:`~orpheus.sn.loss_representation.LossRepresentation.loss_action`
   for the matvec, :meth:`~orpheus.sn.loss_representation.LossRepresentation.sweep`
   for the inverse — "one walk", a code fact per L21), so there is no
   longer a separate FD operator and no by-design bit-difference between
   the two directions.  The FD-operator family
   (``transport_operator_matvec*`` and its unified successor) was deleted
   in the typed-field (#197) and walk-unification (#280 campaigns)
   refactors.  The historical two-closure narrative below is retained for
   the ERR-026 closure-bias reasoning it records.  The ERR-026
   *attribution* it carries was itself later re-adjudicated: the wrong
   curvilinear fixed point was the closure-**seed** family, not the WDD
   closure bias (ERR-058 / #195 —
   :ref:`sn-err-058-closure-seed-closeout`), and post-fix
   SI :math:`\equiv` Krylov is **bit-identical** (#196 —
   :ref:`sn-issue-196-eigenvalue-equivalence`) — so the reconciliation
   this narrative anticipated ("Krylov on ``apply`` avoids the closure
   bias") was falsified: the two paths are ONE discrete system.

This was the load-bearing architectural fact about
:class:`StreamingCollisionOperator`, and the reason the operator's
:meth:`apply` was **not** bit-identical to its :meth:`solve`.

The Wave-D-era SN dispatch in ORPHEUS shipped **two distinct
discretisations** of the same continuous operator
:math:`L+C = \Omega\cdot\nabla + \Sigma_t`, built at different times
for different consumers:

* The **finite-difference operator** (the ``transport_operator_matvec_*``
  family in :mod:`orpheus.sn.operators.streaming`, since deleted) was
  built for the BiCGSTAB inner
  solver path (:meth:`SNSolver._solve_bicgstab_*`).  It used
  upwind cell-center FD on Cartesian and arithmetic face averages
  with **τ-symmetric Morel-Montry angular interpolation** on
  curvilinear (the FD family's own documentation and the module-head
  warning were retired with the code).

* The **sweep operator**
  (:meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve`,
  dispatching its selected representation through the
  :class:`~orpheus.transport.spatial.scheme.DiscretizationScheme`
  Protocol with :class:`~orpheus.transport.spatial.diamond.DiamondDifference`
  as the default strategy) uses the **WDD asymmetric closure**
  :math:`\psi_{n+1/2} = (\overline\psi - (1-\tau)\,\psi_{n-1/2})/\tau`.
  This is the historical SN sweep's closure: the upper-triangular
  forward-substitution form that lets a sweep run in
  :math:`O(N\cdot N_{\rm cells})` work.

In the fine-mesh limit both discretisations converge to the same
continuous operator.  On coarse meshes they differ:

* For Cartesian the difference is :math:`O(h)` (upwind FD has
  the same first-order consistency as DD on uniform meshes;
  divergence appears on non-uniform meshes).
* For curvilinear the WDD asymmetric closure has a closure-bias-
  driven self-consistent fixed point that is **not** the
  fine-mesh-limit transport solution (ERR-026).

The reconciliation is **Wave E Issue 15**: the SN solver's inner
loop migrates from "sweep with WDD closure" to "Krylov on apply,
sweep as preconditioner".  Krylov on :meth:`apply` uses the
symmetric closure (the one that agrees with analytical references)
as the system to solve; the WDD sweep is invoked only as a
preconditioner that accelerates the Krylov iteration without
poisoning the converged solution with its closure bias.  ERR-026
closes when Wave E lands; the 2 xfail-strict tripwires at
:file:`tests/sn/verification/mms/test_curvilinear_aniso_convergence.py`
are the gating bug-catchers for that closure.

Wave E and beyond — landed and forward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Wave E Issue 14 lifted** the iteration primitives (power
  iteration, source iteration) into stand-alone operator-algebra
  consumers in :mod:`orpheus.numerics.iteration`, decoupling them
  from :class:`SNSolver`.
* **Wave E Issue 15 wired** :class:`SNSolver` to
  :class:`StreamingCollisionOperator`: the inner solve becomes Krylov on
  :meth:`apply` with sweep as preconditioner, which removes the WDD
  asymmetric closure from the converged-solution path for the
  reflective-BC eigenvalue case.  Wave E Round 3 will extend the
  equation-map layout to the vacuum-BC path that closes ERR-026
  for fixed-source curvilinear MMS.
* **Forward → landed (Wave O / O.2b, #208)**: the analytic
  reverse-direction adjoint matvec replaced the dense-transpose
  fallback (:meth:`StreamingCollisionOperator.apply_transpose
  <orpheus.sn.operators.streaming.StreamingCollisionOperator.apply_transpose>`,
  single-sourced through the representation's
  ``loss_action_transpose``); the reciprocity gates live in
  :file:`tests/sn/operators/test_streaming_operator.py` and
  :file:`tests/sn/sweep/core/test_phase_c_gates.py` Gate 1.3.

The carve arc (the WHY of the final shape)
------------------------------------------

The carve replaced a *scattered, procedural* dispatch — the same 1-D vs
multi-D decision spelled three different ways (``transport_sweep``
branching on ``reduced is not None``; the matvec branching in five
operator gates on ``not is_1d``; the oracle reachable only through
hand-built test adapters) — with one polymorphic abstraction. The
phases, each independently bit-identical-gated:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Phase
     - What it did
   * - **S0**
     - ``test-architect`` verification plan (the proactive dispatch for
       an operator-algebra carve crossing a subsystem boundary).
   * - **S1**
     - The skeleton: protocol + ``_DAGWavefront`` base + the four
       leaves as **thin wrappers** over the existing sweep code +
       ``is_cartesian`` + ``supports`` / ``default_for`` / registry;
       ``transport_sweep`` rewired. Bit-identical.
   * - **S2**
     - The matvec side: ``strategy.residual`` collapses the **5 matvec
       gates** to one delegating call. Bit-identical.
   * - **S3**
     - Solve-vs-solve equivalence **retires the hand adapters**; the
       full-field spine becomes the genuine d-generic oracle.
   * - **S4**
     - The window generalised to ``frontier_dim = d−1`` (a point at d=1,
       a line at d=2, a surface at d=3); d=2 stays bit-identical.
   * - **S5.1**
     - ``ScanMarch`` built oracle-pinned as **opt-in** (Fork B1) — the
       default unchanged until measured.
   * - **S6**
     - The re-layering: S6.2 rename ``SweepStrategy → LossRepresentation``
       / ``residual → loss_action``; S6.3 the walk moved **off** the
       operator (``operator.py`` became pure algebra, the ``−C`` glue
       collapsed 5×→1×); S6.4 **one walk** + family-owned DAG +
       ``diamond.py`` pure kernel pair (``sweep.py`` dissolved,
       ``WavefrontFlux`` retired with the cochain succession); S6.5
       **one instance**.
   * - **S6.9**
     - Measure + the Fork-B2 default flip (this page's
       :ref:`loss-rep-fork-b2`).

Rejected alternatives (and why)
-------------------------------

* **An enum threaded into** ``transport_sweep``. Rejected: it adds a
  *second* branch axis to a function that already branches on
  dimensionality — cyclomatic complexity, not abstraction. The
  polymorphic-dispatch representation declares the choice once per type.

* **A boolean** ``is_solve`` **flag in the shared walk.** Rejected
  (coding-elegance Smell #3): the direction is carried by kernel and
  emit **objects**; an AST tripwire enforces the absence of the flag.

* **Retiring the window.** Rejected by the Fork-B2 decision: a
  genuinely-different schedule over the same operator is the *point* of
  selectability, and the window is correct and proper on its own.

The ``WavefrontFlux`` succession
--------------------------------

The S6.4 carve **retired** the typed ``WavefrontFlux`` field and its
``InteriorFaceSpace``, but the *concept* — the interior 1-cochain
:math:`C^1_{\rm int}` (with :math:`C^1 = C^1_{\rm int}\oplus C^1_\partial`
remaining valid theory) — survives in two realisations: the rolling
``_MovingFrontier`` front (the window) and the
``_octant_face_cochain`` history (the oracle). The cochain theory anchor
:ref:`wavefront-flux-cochain` is **kept**; its derivation is preserved as
the theory of both realisations.

The realised extension point (2b)
---------------------------------

The designed-but-deferred fifth representation *was* envisioned as an
**ExplicitMatrix**: the sparse-assembled :math:`(L+C)`, whose ``sweep``
would be a triangular direct solve. It was expected to prove that the
representation abstraction is genuinely a *set of schedules over one
operator* — an assembled lower-triangular matrix is the most literal
realisation of the native frame.

Stencil-assembly 2b **realised that reasoning**, but *not* as a fifth
:class:`~orpheus.sn.loss_representation.LossRepresentation`. The assembled
matrix landed instead as the third **consumption mode**
(:ref:`loss-rep-three-modes`) — a numerics carrier
:class:`~orpheus.numerics.assembled_operator.SparseAssembledOperator`
emitted by
:func:`~orpheus.sn.loss_representation.assembly.assemble_ordinate_blocks`
— rather than a new schedule in the registry, because assemble **reuses**
the four schedules' machinery (it walks the same
:class:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph`)
rather than adding a fifth traversal. There was no fifth ``supports``
predicate to write. And the anticipated *direct-solve cross-check* is
exactly the G2 gate: ``scipy.linalg.solve_triangular`` on the assembled
walk-order-triangular matrix reproduces the production sweep at
:math:`\sim 6\times10^{-16}`, discharging #284 at the object level
(:ref:`loss-rep-assembly-verification`). The teaching use case — a
literal display of the triangular structure — is served by the carrier's
``matrix`` attribute / ``as_matrix()`` directly.


Literature
==========

The cell-level mathematics these representations *schedule* is sourced
from the standard discrete-ordinates references, anchored in the
:class:`~orpheus.transport.spatial.diamond.DiamondDifference` docstring and
:doc:`/theory/methods/sn/index`:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Reference
     - Used for
   * - Lewis & Miller (1984), *Computational Methods of Neutron
       Transport*, §4.5 / §5.3 / §6.4
     - The Morel–Montry angular closure (§4.5), the Diamond / weighted-DD
       / Step / LD closures and the negative-flux failure mode (§5.3),
       and the canonical MMS ansatz set (§6.4).
   * - Hébert (2009), *Applied Reactor Physics*, Ch. 3 §3.9.4
     - The curvilinear S\ :sub:`N` cell-balance and DD difference
       relations.
   * - Blelloch (1990), *Prefix Sums and Their Applications*,
       CMU-CS-90-190 §1.5
     - The first-order linear-recurrence scan
       (:func:`~orpheus.sn.sweep.scan.ordinate_scan`) that
       ``CumprodScan`` and ``ScanMarch`` both reuse — the pair-monoid
       closed form.
   * - Adams & Larsen (2002), *Fast iterative methods for
       discrete-ordinates particle transport calculations*, §III
     - The within-group iteration framing in which the sweep
       :math:`(L+C)^{-1}` is the transport operator and the matvec
       :math:`(L+C)` its twin, and the Krylov-on-``apply`` /
       sweep-preconditioning survey the two-closure history
       (:ref:`loss-rep-history`) invokes.
   * - Trefethen & Bau (1997), *Numerical Linear Algebra*, §3.2
     - The matrix-free Krylov view under which ``apply`` is the
       operator's action and the sweep its preconditioner — the frame
       of the two-closure reconciliation record.
   * - Maginot, Ragusa & Morel (2016), *A non-negative moment-preserving
       spatial discretization scheme for solving the
       discrete-ordinates equations* (Upstream / UBLD)
     - The irreducible multi-dimensional coupling of the
       Linear-Discontinuous slope moments — the tensor-product bilinear
       (UBLD) :math:`d`-D LD closure that the routing gate
       (:ref:`loss-rep-scanmarch-facewise`) excludes from the row-march.
   * - Adams (2001), *Discontinuous finite element transport solutions in
       thick diffusive problems*, NSE 137(3)
     - The thick-diffusion-limit behaviour establishing why the multi-D LD
       closure is the bilinear :math:`\{1, x, y, xy\}` form (not the simplex
       :math:`1+d`), i.e. why the transverse coupling is a slope moment, not
       a face value.


See also
========

* :doc:`/theory/methods/sn/index` — the cell-level WDD algebra, the 1-D
  cumprod recurrence (:ref:`sweep-cumprod`), the 2-D anti-diagonal
  wavefront (:ref:`sweep-wavefront`), and the three-layer walk /
  level-op / kernel stack (:ref:`sweep-dispatch-relayering`).
* :doc:`/theory/foundations/operator_algebra` — the Wave-O typed operator algebra
  (:eq:`operator-apply`, :eq:`operator-solve`,
  :eq:`operator-apply-transpose`) and the interior face-flux cochain
  (:ref:`wavefront-flux-cochain`).
* :mod:`orpheus.sn.loss_representation` — the module that implements
  every representation, the selection layer, and the orchestration
  entry.
* ``.claude/plans/archive/sn_sweep_strategy.md`` — the authoritative locked
  design (decisions, verification strategy, phases S0–S6).
