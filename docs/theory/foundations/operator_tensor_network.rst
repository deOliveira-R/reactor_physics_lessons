.. _operator-tensor-network:

=====================================================
Tensor-Network Decomposition of S\ :sub:`N` Operators
=====================================================

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
      concept: operator_tensor_network
      role: "tensor-network / factored-shape decomposition of the S_N transport operators"
      depends_on: [operator_algebra]
      status: "extracted from operator_algebra.rst; content verbatim, provisional header"


This page develops the **factored / tensor-network shape decomposition**
of the S\ :sub:`N` transport operators: *which* algebraic shape each
operator leaf actually takes once it is lifted out of a procedural,
single-axis numpy body into the operator-algebra types
(:class:`~orpheus.numerics.operator.TensorProductOperator`,
:class:`~orpheus.numerics.operator.SumOfTensorProductsOperator`,
:class:`~orpheus.numerics.operator.OperatorSum`) developed in
:doc:`/theory/foundations/operator_algebra`. The headline result is that
**no single uniform tensor-product factorisation across space, angle and
energy** exists: the shipped state is *five algebraically distinct shapes*,
chosen per operator by what the underlying physics actually couples — and
the streaming operator, whose in-sweep recurrence couples the axes
sequentially, resists a clean tensor product altogether.

- :ref:`tensor-network-shape-table` — the per-operator shape catalogue and the
  MA-Q1 master condition that decides when a tensor product is admissible.
- :ref:`tensor-network-streaming-deep-dive` — streaming's in-sweep WDD recurrence
  and why it is not tensor-separable.
- :ref:`tensor-network-orchestrated-apply` — the single bidirectional pass, its
  design rationale, and the retired per-direction split (#238).
- :ref:`tensor-network-curvilinear-deep-dive` — the curvilinear angular
  redistribution (Morel–Montry) thread.


.. _tensor-network-decomposition:

Tensor-Network Decomposition of SN Operators
============================================

Wave T (May 2026, commits ``fa13e78`` / ``0b2848b`` / ``9f85c5d`` /
``03bcdba`` / ``cb18fdb`` / ``c55b505`` / ``90e7d4e``) lifted the four
SN operator leaves — boundary realizers, fission, scattering,
streaming — from procedural single-axis numpy bodies into the
operator-algebra types documented in
:doc:`/theory/foundations/operator_algebra`
(:class:`~orpheus.numerics.operator.TensorProductOperator`,
:class:`~orpheus.numerics.operator.SumOfTensorProductsOperator`,
:class:`~orpheus.numerics.operator.OperatorSum`). The migration is
the consumer side of the Wave-0 + Depth B D-B infrastructure
(:class:`~orpheus.numerics.operator.TensorProductOperator` shipped at
commit ``bc1253e``, 2026-05-10;
:class:`~orpheus.numerics.space.TensorProductSpace` shipped at commit
``c2f968a``, 2026-05-27; the first production consumer was the D-B+1
specular BC at ``boundary_realizer.py:164-166``).

What landed is **not** the uniform :math:`A = \sum_k A_x^{(k)} \otimes
A_\omega^{(k)} \otimes A_g^{(k)}` aspiration of Grand Report v3
§15-§16A. The shipped state is **five algebraically distinct shapes**,
chosen per operator by what the underlying physics actually couples.
Future agents who assume "all SN operators are
:class:`SumOfTensorProductsOperator`" — including Wave O
(`Issue #208 <https://github.com/deOliveira-R/ORPHEUS/issues/208>`_)
operator-role typing — will be wrong. This section names the master
condition that decides the shape, catalogues which shape each operator
uses, and documents the architectural rationale for the per-direction
streaming split.


Key Facts
---------

- **The SN flux state lives on a tensor-product space** :math:`V = X
  \otimes \Omega \otimes G` (Grand Report v3 §15 line 2003-2019). The
  shipped array layout ``(N, ng, nx, ny)`` is the implicit numpy
  realisation: the angular axis :math:`\Omega` is leading, the group
  axis :math:`G` is next, and the spatial axis :math:`X` trails (see
  :ref:`theory-sn-index-convention`).

- **Five algebraic shapes** are now in production simultaneously,
  selected by the per-operator physics coupling — see
  :ref:`tensor-network-shape-table` below for the catalogue.

- **The MA-Q1 master condition** (load-bearing for every future
  consumer):

  .. epigraph::

     :class:`SumOfTensorProductsOperator` (SOTP) requires Cartesian-
     product per-axis decomposition: every summand factors as a
     product of independent per-axis operations. *Coupled physics* —
     per-material XS lookup that ties group to spatial cell,
     sequential WDD recurrence that ties spatial cells, M-M half-grid
     recurrence that ties angular :term:`ordinates <ordinate>` — falls back to
     :class:`OperatorSum` over bespoke :class:`LinearOperator`
     summands, **NOT** SOTP.

- **Zero production consumers** of
  :class:`SumOfTensorProductsOperator`. The §15.2 SOTP form is
  contradicted by the actual coupling structure of scattering (T.3)
  and streaming (T.4). Only T.1 (BC realizers) and T.2 (fission rank-1)
  cleanly admit the clean tensor-product factorisation.

- **Wave O typing constraint**: operator-role types
  (``BulkOperator`` / ``FullOperator`` / ``BoundaryOperator``
  Protocols, Issue #208) **MUST** accept non-SOTP summands. Any
  contract that requires "all summands are
  :class:`TensorProductOperator`" forecloses scattering and streaming.

- **In-sweep streaming (the retired per-direction split, #238)**: the
  forward (μ_x > 0) and backward (μ_x < 0) WDD recurrences are walked in
  ONE bidirectional pass inside
  :meth:`~orpheus.sn.loss_representation._OneDimScanWalk._apply_walk` (the
  fused ``(L+C)ψ`` matvec, the apply-direction twin of the sweep). Wave T
  briefly exposed the two directions as separately-applicable typed leaves
  (``M_spatial = _SpatialSweepDirection(+1) + _SpatialSweepDirection(-1)``)
  to anticipate Wave-O adjoint / DSA-class consumers — but #240's adjoint
  uses the fused ``loss_action_transpose`` and no production code ever
  applied the leaves separately, so #238 retired the split. The
  per-direction coupling structure it documented (the backward sweep's seed
  depends on the forward sweep's outer-face WDD outflow) is still real and
  is why the matvec is one shared pass, not two independent forward sweeps.

- **Hybrid 2-D Cartesian**: T.4 lifted 1-D only. The 2-D Cartesian
  path (then ``StreamingOperator._apply_2d_cartesian``) stayed
  procedural FD with cell-centre-proxy semantics, guarded by a
  defensive source-hash pin (A2D-1) against silent author drift.
  *(Superseded: O.4b replaced the FD path with the trace-correct
  graph walk; S6.3 moved the walk onto the loss representation; at
  S6.4(a) the A2D-1 pin was RETIRED — its tripwire job transferred to
  the ``window ≡ full`` matvec output oracle in
  ``test_2d_full_field_oracle.py``, which catches actual drift by
  ``assert_array_equal`` against the structurally-distinct full-field
  oracle instead of pinning source text on what is now a SHARED
  octant walk.)*


.. _tensor-network-shape-table:

Per-operator shape catalogue
----------------------------

The five algebraic shapes that ship today, grouped by Wave T substep.
Each row names the operator, the algebraic shape its kernel/apply
takes, a concrete example, and the physics coupling that forced the
shape choice.

.. list-table:: Wave T per-operator shape catalogue
   :header-rows: 1
   :widths: 18 22 30 30

   * - Operator
     - Algebraic shape
     - Example
     - Why this shape
   * - BC realizers (vacuum,
       specular, white,
       albedo, periodic)
     - :class:`TensorProductOperator`
       (single TP), or a bare
       :class:`ZeroOperator` for
       vacuum
     - ``PermutationOperator(local_perm, axis=0,
       domain=Γ₊(f), codomain=Γ₋(f)) &
       IdentityOperator()`` for specular, on the
       **reduced** ordinate axis :math:`\Gamma_+`;
       a two-hook ``ZeroOperator`` for vacuum
       (:math:`\Gamma_+ \to \Gamma_-`, no TP at
       all)
     - Each BC acts on the ordinate axis; the
       trailing group / face axes broadcast. Per
       §16A.10 ``B = G_patch ⊗ K_omega ⊗ K_g``
       with two factors degenerate to
       :class:`IdentityOperator`. Since campaign
       phase B3.2 the ordinate factor lives on
       the law's **narrowed** domain
       :math:`\Gamma_+`, not the full face
       (:ref:`bc-domain-narrowing`); vacuum has
       no non-trivial factor left to carry, so it
       drops out of the TP form entirely.
       ⚠ **Carry the binding into the example you
       copy.** The identity factor declares no
       space, so the product takes the ordinate
       factor's (:ref:`tensor-product-spaces`) —
       and an unbound build is not merely
       untyped, its ``.H`` degrades silently to
       the Euclidean transpose.
   * - Fission (:math:`F`)
     - :class:`TensorProductOperator`
       (single rank-1 dyad)
     - ``outer(χ, ReactionRateFunctional(νΣ_f)) &
       IdentityOperator()``
       (``IsotropicFission.kernel`` — the fission energy
       binding since CS4c step 4;
       ``FissionOperator.kernel`` delegates)
     - Per Grand Report v3 §15 line 2008
       :math:`F = |\chi\rangle\langle\nu\Sigma_f|`. The
       group-axis contraction-then-broadcast is
       exactly :class:`RankOneOperator`; spatial
       axes broadcast.
   * - Scattering kernel
       (:math:`S_{\rm aniso}`)
     - :class:`OperatorProduct`
       :math:`R \circ \Lambda_{\ell\ge 1} \circ M`
     - the moment-space integral kernel
       ``frame.conjugate(Λ)``
       (:attr:`TransferOperator.kernel
       <orpheus.transport.operators.transfer.TransferOperator.kernel>`),
       projecting to harmonic
       moments, applying per-ℓ transfer, reconstructing per-ordinate
     - **MA-Q1 fallback**: the per-material per-ℓ
       einsum
       :meth:`TransferMaterialField.moment_source
       <orpheus.transport.material_field.TransferMaterialField.moment_source>`
       (``MaterialXSField.apply_legendre_scattering_moments`` until the
       CS4c step-3 O-6 move, then ``ScatteringMaterialField.moment_source``
       until #426 step 2 collapsed the two transfer channels' fields
       into one)
       couples the group axis (matrix multiply on
       :math:`\Sigma_{s,\ell}[g'\to g]`) with the
       spatial axis (via
       :attr:`cells_by_material` indexing). No
       SOTP factorisation respects disjoint axes;
       the §15.2 SOTP target form fails the
       :class:`TensorProductOperator` contract.
   * - Streaming spatial
       part (in-sweep;
       #238 retired the
       ``M_spatial`` leaf)
     - Fused WDD recurrence in
       :meth:`~orpheus.sn.loss_representation._OneDimScanWalk._apply_walk`
     - The forward (μ_x > 0) + backward (μ_x < 0)
       sweeps walked in ONE bidirectional pass
       (formerly the ``M_spatial`` per-direction
       :class:`OperatorSum`)
     - **MA-Q1 fallback**: the WDD recurrence
       :math:`\psi_{\text{face,out}} = 2\bar\psi
       - \psi_{\text{face,in}}` sequentially
       couples cells along x. It is NOT a clean
       :math:`(D_x \otimes \Omega_x \otimes I_g)`
       3-factor TP — the sweep operator is the
       leaf factor. The two directions cannot be
       applied independently (the backward seed
       depends on the forward outer-face outflow),
       which is why the matvec is one shared pass.
   * - Streaming angular
       redistribution
       (in-sweep; #238
       retired the
       ``M_angular_redist``
       leaf)
     - In-sweep Morel–Montry thread inside
       :meth:`~orpheus.sn.loss_representation._OneDimScanWalk._apply_walk`
       (sphere / cylinder; zero for slab / 2-D
       Cartesian)
     - The per-cell M-M contribution from
       :meth:`AngularClosureBase.cell_contribution`,
       added to the cell balance during the walk
     - **MA-Q1 fallback**: the Carlson
       starting-direction march (Hébert 2009
       §3.9.4 Eqs. 3.432-3.435) that seeds the
       M-M half-grid sequentially couples
       angular ordinates ``α_{m+1/2}`` from
       ``α_{m-1/2}`` with σ_t-dependent
       absorption coefficients. Not a diagonal
       angular factor; a 3-factor TP wrap would
       false-assert separability the recurrence
       doesn't support. Verified end-to-end by the
       anisotropic curvilinear MMS
       (:ref:`sn-mms-curvilinear-aniso-verification`).


The MA-Q1 master condition
--------------------------

The pattern across T.3, T.4-spatial, and T.4-curvilinear is the same:
*coupled physics produces summands that fail the disjoint-axes
contract of* :class:`TensorProductOperator`. Naming this explicitly
prevents future agents from re-attempting the §15.2 SOTP form on each
of these operators.

.. (vv-status rationale) Master condition gate for Wave-O typing
   decisions. Verified by the absence of SOTP-shaped consumers in
   production after T.3 + T.4 land — exhaustively documented in
   ``.claude/plans/wave_t_tensor_network.md`` §6 T.3 + T.4 deviations.
.. vv-status: tensor-network-ma-q1-master-condition documented

.. math::
   :label: tensor-network-ma-q1-master-condition

   \text{SOTP applies} \;\Longleftrightarrow\;
   \text{each summand factors as} \;
   f(x_1,\dots,x_d) \;=\; f_1(x_1)\otimes\cdots\otimes f_d(x_d).

When the physics violates the right-hand side — and three of the four
Wave-T-touched operators do — the algebraic home is
:class:`OperatorSum` over :class:`LinearOperator` summands, NOT
:class:`SumOfTensorProductsOperator`. The §15.2 target form is
*aspirational* in the grand report; Wave T documents that the actual
coupling structure of multigroup transport with per-material
cross-sections does not admit it for scattering and streaming.

**Three coupled-physics archetypes** ship in Wave T:

1. **Per-material XS coupling** (T.3 scattering). The per-material
   einsum :math:`\sum_{g'}\Sigma_{s,\ell}^{m(\vec r)}[g'\to g]
   \phi_{\ell,g'}^{m}(\vec r)` ties the group axis (matrix multiply)
   to the spatial axis (per-cell material id lookup). The factor
   :math:`\Lambda_\ell` cannot be written as a group-axis-only
   operator without information loss.

2. **Sequential WDD recurrence** (T.4 streaming, spatial). The
   :term:`diamond-difference <diamond difference>` closure :math:`\psi_{\text{face,out}} =
   2\bar\psi_{\text{cell}} - \psi_{\text{face,in}}` makes the cell
   :math:`i+1` value depend on the cell :math:`i` value. A
   per-direction sweep summand IS the WDD recurrence as a single
   :class:`LinearOperator` — not a factor on a per-cell tensor axis.

3. **M-M half-grid recurrence** (T.4 streaming, curvilinear angular).
   The Carlson starting-direction march (Hébert 2009 §3.9.4
   Eqs. 3.432-3.435) that seeds it recurs sequentially along the angular
   axis within each μ-level, each face depending on the previous one and
   on σ_t. The leaf factor is the entire recurrence — a single
   :class:`LinearOperator` — not a diagonal angular operator.
   (Provenance, for the record: the :math:`\alpha`-dome recursion
   :math:`\alpha_{m+1/2} = \alpha_{m-1/2} - \mu_m w_m` is Hébert's
   Eqs. 3.423-3.424 and carries **no** σ_t; the weighted :math:`\tau`
   the half-grid recurrence uses is Morel--Montry's, not Hébert's — see
   :ref:`sn-tau-source-of-record`.)

In each case, the algebraic home is the SAME — :class:`OperatorSum`
over bespoke :class:`LinearOperator` summands — and the
:class:`TensorProductOperator` form is structurally inaccessible
without information loss.


.. _tensor-network-streaming-deep-dive:

Streaming deep dive — in-sweep WDD recurrence (the retired per-direction split)
-------------------------------------------------------------------------------

.. note:: #238 — the ``M_spatial`` / ``M_angular_redist`` typed-leaf split was retired.

   Wave T briefly exposed the streaming matvec as separately-applicable
   typed leaves: ``StreamingOperator.M_spatial`` (an :class:`OperatorSum`
   of two per-direction-sign sweep summands) and
   ``StreamingOperator.M_angular_redist`` (a bespoke curvilinear
   angular-redistribution leaf). The split was designed to anticipate
   Wave-O adjoint propagation, DSA-class preconditioners, and
   per-direction debugging. **#238 retired it**: no production code ever
   applied the leaves separately (the #240 G-adjoint rides the fused
   ``loss_action_transpose``; the open #200 block-inverse preconditioner
   and #2 DSA never landed), so keeping the leaves alive solely to feed
   their own structural tests was the same orphan smell one level down.
   The streaming + curvilinear Morel–Montry angular redistribution is now
   computed **in-sweep** inside the fused matvec
   :meth:`~orpheus.sn.loss_representation._OneDimScanWalk._apply_walk` (the
   apply-direction twin of the sweep), and the angular-redistribution term
   is verified end-to-end by the anisotropic curvilinear MMS
   (:ref:`sn-mms-curvilinear-aniso-verification`,
   ``catches("ERR-026")``) — the surviving structural-independence ground.

The *physics* the split documented is unchanged and load-bearing: the
WDD spatial recurrence and the M-M angular recurrence are both
sequentially coupled, which is why neither admits a clean
:class:`TensorProductOperator` factorisation (the MA-Q1 master condition
above). The two equations below pin that coupling, and they are the
reason the matvec is a single sequential walk rather than a stack of
independent per-axis broadcasts.

**Where the WDD recurrence comes from**. For a single ordinate with
:math:`\mu_x > 0`, the discrete cell balance over cell :math:`i`
(width :math:`\Delta x_i`, total cross-section :math:`\Sigma_t(i)`,
cell-averaged in-group source :math:`\bar Q_i`) is

.. (vv-status rationale) The discrete WDD cell balance for one ordinate.
   Derivation step feeding the sentineled wdd-forward-recurrence; the
   discrete sweep it produces is verified downstream (dd-slab-scalar).
.. vv-status: wdd-cell-balance documented

.. math::
   :label: wdd-cell-balance

   |\mu|\,\bigl[(\psi_{\text{face,out}})_i - (\psi_{\text{face,in}})_i\bigr]
   \;+\; \Sigma_t(i)\,\Delta x_i\,\bar\psi_i
   \;=\; \Delta x_i\,\bar Q_i .

This single equation carries **two** unknowns —
:math:`(\psi_{\text{face,out}})_i` and the cell average
:math:`\bar\psi_i`. The diamond-difference (DD) closure supplies the
second relation, asserting the cell average is the arithmetic mean of
the two faces,

.. (vv-status rationale) The diamond-difference closure
   ψ̄ = ½(ψ_in + ψ_out). Definitional closure relation; with
   wdd-cell-balance it yields the sentineled wdd-forward-recurrence,
   verified downstream (dd-slab-scalar).
.. vv-status: wdd-diamond-closure documented

.. math::
   :label: wdd-diamond-closure

   \bar\psi_i \;=\;
     \tfrac12\bigl[(\psi_{\text{face,in}})_i + (\psi_{\text{face,out}})_i\bigr]
   \;\Longleftrightarrow\;
   (\psi_{\text{face,out}})_i \;=\; 2\,\bar\psi_i - (\psi_{\text{face,in}})_i .

Substituting the closure into the balance and solving for the cell
average gives the **forward WDD recurrence**:

.. math::
   :label: wdd-forward-recurrence

   \bar\psi_i \;=\;
     \frac{\Delta x_i\,\bar Q_i + |\mu|\,(\psi_{\text{face,in}})_i}
          {\Delta x_i\,\Sigma_t(i) + |\mu|},
   \qquad
   (\psi_{\text{face,out}})_i \;=\;
     2\,\bar\psi_i \;-\; (\psi_{\text{face,in}})_i

with :math:`(\psi_{\text{face,in}})_{i+1} =
(\psi_{\text{face,out}})_i` (the outflow of cell :math:`i` is the
inflow of cell :math:`i+1`).

**Why this forbids a tensor product**. The recurrence is a
*sequential* dependence: the cell-:math:`i+1` average cannot be
formed until the cell-:math:`i` outflow is known, which itself
depends on cell :math:`i-1`, and so on back to the boundary inflow.
Unrolling the recurrence to its closed form makes the structure
explicit — the cell average is a lower-triangular linear functional
of the upstream source and the inflow face:

.. math::

   \bar\psi_i \;=\;
     \frac{|\mu|}{\Delta x_i\Sigma_t(i)+|\mu|}\,\psi_{\text{bdy,in}}
     \prod_{j<i}\frac{2|\mu| - \Delta x_j\Sigma_t(j) - |\mu|}
                     {\Delta x_j\Sigma_t(j)+|\mu|}
     \;+\;\bigl(\text{source terms}\bigr),

i.e. the action on the spatial axis is the *whole* lower-triangular
sweep operator :math:`T_x`, not a per-cell diagonal that broadcasts.
There is no factorisation :math:`(D_x \otimes \Omega_x \otimes I_g)`
in which :math:`D_x` acts independently on each spatial cell: the
off-diagonal products :math:`\prod_{j<i}(\cdots)` carry information
*between* cells and *depend on the ordinate* :math:`\mu` through the
denominators. The spatial factor and the angular index are entangled
inside the recurrence — the disjoint-axes contract of
:class:`TensorProductOperator` (:eq:`tensor-network-ma-q1-master-condition`)
fails. The cell-balance algebra at
:func:`orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`
hides this recurrence inside the named denom-numer primitives, but
the recurrence is the load-bearing structure that makes the streaming
matvec a sweep rather than a broadcast.

.. vv-status: wdd-forward-recurrence documented

**Forward-backward coupling at the outer face**. The forward (μ_x > 0)
and backward (μ_x < 0) sweeps cannot be applied independently: the
backward sweep's seed depends on the forward sweep's outer-face WDD
outflow. Concretely, the forward sweep marches from the inner boundary
to the outer boundary, terminating with the outer-face outflow
:math:`(\psi_{\text{face,out}})_{N-1}`; the backward sweep then marches
from the outer boundary back inward, and on a reflective or curvilinear
mesh its seed inflow at the outer face is determined by that same
forward outflow. The two directions therefore cannot run as two parallel
independent forward sweeps — the data dependency from the forward
terminus into the backward seed is intrinsic to the recurrence. This is
why
:meth:`~orpheus.sn.loss_representation._OneDimScanWalk._apply_walk`
walks both directions in ONE bidirectional pass, regardless of whether
the directions are exposed as separate operator leaves. (Exposing them
as leaves does not remove the coupling — it only forces the leaves to
share state, which is precisely the smell that motivated the
now-retired orchestrator; see :ref:`tensor-network-orchestrated-apply`.)

.. note::

   The forward-outflow-seeds-backward-inflow coupling was the
   *pre*-O.4a.2 mechanism, where the backward sweep's inflow was
   ``bc_outer.apply(forward_outflow)`` inside one matvec. Wave O step
   O.4a.2 (Issue #208) **deleted that intra-call reflective re-apply** —
   the boundary law :math:`B` is now a sibling :math:`-B` operator and
   the backward sweep reads the *given* outer inflow trace directly. The
   forward outflow today feeds the **outflow self-consistency defect** on
   the outflow trace row, not a reflected inflow seed.
   See :ref:`bc-extraction`.


.. _tensor-network-orchestrated-apply:

One bidirectional pass — design rationale and the retired per-direction split
-----------------------------------------------------------------------------

The fused matvec
:meth:`~orpheus.sn.loss_representation._OneDimScanWalk._apply_walk` runs
the bidirectional sweep **once**, returning the full :math:`(L+C)\,\psi`.
This is the whole production story today; the rest of this section records
*why* Wave T briefly exposed a richer surface, *what* that surface looked
like, and *why* #238 retired it. The history is load-bearing: it
documents a structural fact — that the per-direction split could never
have been cheaper or simpler than the fused pass — so a future session
does not re-introduce the split on the mistaken belief that it buys
modularity for free.

What Wave T originally built (Design B)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wave T exposed the streaming matvec as a **subclass of**
:class:`OperatorSum`, ``_MSpatialOperatorSum``, whose two summands were
``_SpatialSweepDirection(+1)`` (the μ_x > 0 forward sweep) and
``_SpatialSweepDirection(-1)`` (the μ_x < 0 backward sweep), so that

.. math::

   M_{\rm spatial} \;=\;
     \texttt{_SpatialSweepDirection}(+1)
     \;+\;
     \texttt{_SpatialSweepDirection}(-1) .

Carrying the two directions as named ``.a`` / ``.b`` attributes of an
:class:`OperatorSum` made them visible to type introspection by Wave O
(:ref:`bc-extraction`), adjoint propagation, and any DSA-class
preconditioner that might want to address one direction at a time. The
subclass **overrode** :meth:`OperatorSum.apply`, because the default
implementation

.. math::

   \texttt{OperatorSum.apply}(x) \;=\;
     \texttt{self.a.apply}(x) \;+\; \texttt{self.b.apply}(x)

would have cost **1.5× the unified matvec walltime**. The reason is the
forward-backward coupling of :ref:`tensor-network-streaming-deep-dive`: each
standalone ``_SpatialSweepDirection.apply`` internally ran the *entire*
bidirectional sweep (it had to, to obtain the seed coupling) and then
masked the opposite-direction ordinates to zero. Summing the two
standalone summands therefore re-ran the forward sweep twice. The
orchestrator's override ran the bidirectional sweep once via the shared
``_MSpatialOperatorSum._compute_LpC`` and returned the full
:math:`(L+C)\,\psi`, avoiding the duplication; the standalone
per-direction summands were preserved only as a slow fallback for
testing, adjoint inspection, and per-direction debugging.

The forward-sweep outer-face WDD outflow that had been a hidden local
of the legacy unified matvec was lifted by the orchestrator into the
*named shared state* of ``_compute_LpC`` (``coding-elegance`` Pattern 6
— single source of truth: hidden coupling points must become named).
The full bidirectional matvec is mathematically equivalent to
:math:`M_{x,+}\,\psi + M_{x,-}\,\psi`; the orchestrator returned that
value bit-exact (preserving the unified matvec's reduction order), while
the masked per-direction summands summed to the same value at
FP-non-associativity ULP.

The five anticipated consumers — and why none materialised
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The split was speculative architecture: it was built to *anticipate*
future consumers that would want a separately-applicable per-direction
or per-term operator leaf. The honest record of what was expected, and
what actually happened, is:

.. list-table:: Anticipated consumers of the per-direction / per-term split
   :header-rows: 1
   :widths: 26 38 36

   * - Anticipated consumer
     - Why a separate leaf was expected to help
     - What actually shipped (why it didn't materialise)
   * - **Wave-O adjoint** (Issue #208 / #240 G-adjoint)
     - A per-direction leaf was expected to make
       :math:`M_{\rm spatial}^{\mathsf T}` introspectable
       direction-by-direction, so the adjoint could be assembled by
       transposing each summand.
     - The #240 G-adjoint rides the **fused**
       :meth:`~orpheus.sn.loss_representation._OneDimScanWalk.loss_action_transpose`
       — the apply-direction twin transposed as a whole walk. The
       transpose of a lower-triangular sweep is an upper-triangular
       sweep over the *same* coupling; splitting it per direction first
       gains nothing and re-introduces the shared-state coupling.
   * - **#2 consistent DSA**
     - A diffusion-synthetic accelerator was expected to consume an
       isolated spatial-streaming leaf as the operator to precondition.
     - DSA never needs the streaming term split from collision, let
       alone the forward direction split from the backward: the landed
       accelerator (:mod:`orpheus.sn.acceleration.dsa`) consumes the
       iterate **displacement** through the fused sweep, and its
       low-order operator is the **derived edge-centered SN-side
       consistent system** (the R4 ruling — the standalone in-algebra
       :math:`A_{\rm diff} = L + C - S - B` of
       :doc:`/theory/methods/diffusion_1d` — four terms because that
       solver sums its two collision gains into one :math:`S`, so it
       carries no separate :math:`N_{2n}` — remains the right
       *standalone* discretization but was MEASURED divergent as a
       correction operator at :math:`\sigma_t h \gtrsim 2`; two
       defining laws, two operators). Correctness-safe by construction
       either way: the correction :math:`\to 0` at convergence. No
       per-direction streaming leaf anywhere in the loop.
   * - **#200 block-inverse preconditioner**
     - A per-direction leaf was expected to feed a block-inverse Krylov
       preconditioner that addressed each direction block.
     - The full Morel–Montry sweep is *already* the natural
       :math:`O(N)` exact inverse :math:`(L+C)^{-1}` (the sweep solves
       in one pass). A spatial/angular block split of an operator whose
       inverse is already a single cheap pass would be **weaker, not
       cheaper** — a block preconditioner approximates an inverse that
       the sweep computes exactly. #200 remains open and, when it lands,
       has no reason to want the split.
   * - **Per-direction debugging**
     - Inspecting one sweep direction in isolation while debugging.
     - The fused walk is debuggable directly (the per-level / per-cell
       visits are observable inside the single pass); a standalone
       direction summand that re-runs the whole bidirectional sweep and
       masks the other direction is a *worse* debugging surface than the
       single pass, because the masking hides the coupling under test.
   * - **Slow per-direction test fallback**
     - A standalone per-direction ``apply`` as a structural cross-check
       on the fused pass.
     - The cross-check that mattered — that the fused
       :math:`(L+C)\,\psi` is correct — is supplied by the
       anisotropic curvilinear MMS
       (:ref:`sn-mms-curvilinear-aniso-verification`), a
       *structurally-independent* L1 ground. A bit-identity invariant
       between the fused pass and the sum of its own per-direction
       summands (see :ref:`tensor-network-curvilinear-deep-dive`) only verified
       that the split *reconstructed itself*, not that either branch was
       correct.

Why the split was retired — the orphan-smell one level down
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After #206 moved the 1-D matvec walk into the loss representation, the
``_SpatialSweepDirection`` / ``_MSpatialOperatorSum`` /
``AngularRedistributionOperator`` leaves had **zero production
consumers**. Every production matvec went through the fused walk; the
leaves existed only to be applied by their own structural tests — the
algebra-decomposition invariant
:math:`(L+C)\,\psi \equiv M_{\rm spatial}\,\psi + M_{\rm angular\_redist}\,\psi`
(see :ref:`tensor-network-curvilinear-deep-dive`). This is the classic *orphan
smell*: machinery kept alive solely to feed the tests that exist solely
to verify that machinery. Cardinal Rule 2 (architecture is critical)
treats a self-referential test loop as a code smell — the tests prove
the decomposition is internally consistent without proving anything a
production consumer relies on.

The decisive observation is that the split was never going to be the
cheaper *or* the more modular path, for a structural reason rather than
an incidental one:

- **It could not be cheaper.** The forward-backward coupling
  (:ref:`tensor-network-streaming-deep-dive`) means a standalone per-direction
  apply must run the whole bidirectional sweep anyway. The orchestrator
  existed precisely to undo the 1.5× penalty of the naïve sum — i.e. to
  claw back to the cost of the single fused pass it was wrapping. The
  fused pass is the floor; the split can at best tie it.
- **It could not be more modular.** Exposing the directions as leaves did
  not decouple them — it forced them to *share named state*
  (``_compute_LpC``'s lifted outflow), so the "modular" surface was a
  pair of objects that could not be evaluated independently. That is the
  illegal state ``coding-elegance`` Pattern 4 warns against:
  representing two summands as separable when the math makes them
  inseparable.

So #238 removed the orchestrator and both leaves entirely. The single
bidirectional pass survives as the production path; the 1.5× cost the
override avoided is now moot — there is only one pass to run — and the
single-source-of-truth property is satisfied *more directly*: one walk,
one source, no orchestrator coordinating two summands that were never
independent. The angular-redistribution term that the curvilinear leaf
isolated is computed in-sweep (see :ref:`tensor-network-curvilinear-deep-dive`)
and verified end-to-end by the anisotropic curvilinear MMS
(:ref:`sn-mms-curvilinear-aniso-verification`, ``catches("ERR-026")``)
— the surviving structural-independence ground.

.. note:: **If a future consumer genuinely needs a per-direction or
   per-term leaf.** Should a #200 block-inverse preconditioner or a #2
   DSA variant ever surface a *real* need for a separately-applicable
   spatial / angular leaf (one that does not re-run the full
   bidirectional sweep), the correct move is **not** to resurrect
   ``_MSpatialOperatorSum``. The structural obstruction above — that the
   forward-backward coupling makes the directions inseparable — has to be
   addressed first: the consumer would need a formulation in which the
   coupling itself is the object being preconditioned (e.g. the
   in-algebra diffusion operator for DSA — now built as
   :math:`A_{\rm diff} = L + C - S - B`, #290 P4,
   :doc:`/theory/methods/diffusion_1d`; four terms because diffusion
   fuses its two collision gains into one :math:`S` and so has no
   separate :math:`N_{2n}` — per the architecture decided on
   Issue #2), not a re-split of the sweep into directions that secretly
   share state.
   The fused
   :meth:`~orpheus.sn.loss_representation._OneDimScanWalk.loss_action`
   /
   :meth:`~orpheus.sn.loss_representation._OneDimScanWalk.loss_action_transpose`
   pair is the principled surface; build the new consumer against it.


.. _tensor-network-curvilinear-deep-dive:

Curvilinear angular redistribution — in-sweep Morel–Montry thread
-----------------------------------------------------------------

For sphere / cylinder geometries the curvilinear M-M (Morel–Montry)
half-grid angular redistribution is woven into the cell balance during
the fused walk (it returns zero for slab / 2-D Cartesian). The per-cell
M-M coefficients come from
:meth:`AngularClosureBase.cell_contribution` (Pattern 6 — single source
of truth for the M-M coefficients). #238 retired the bespoke
``AngularRedistributionOperator`` leaf that re-walked the matvec only to
isolate this term; the redistribution is the same in-sweep computation,
now without a separately-applicable wrapper.

**Where the angular-redistribution term comes from**. In curvilinear
geometry the streaming operator acquires a term with no slab analogue:
the angular derivative that accounts for the rotation of the local
direction frame as a particle streams along a curved trajectory. For
the sphere this is :math:`\frac{1-\mu^2}{r}\,\partial\psi/\partial\mu`;
for the cylinder it is :math:`-\frac1r\,\partial(\xi\psi)/\partial\varphi`.
Discretised over the cell volume (see the canonical derivation in
:doc:`/theory/methods/sn/curvilinear_one_group`, Step 2–3,
Eq. :eq:`balance-general`), it
becomes a *half-grid* difference between angular faces
:math:`m\pm\tfrac12`,

.. (vv-status rationale) The Morel–Montry angular-redistribution
   discretization ∫(1−μ²)/r ∂ψ/∂μ dV ≈ α_{m+½}ψ_{m+½} − α_{m−½}ψ_{m−½}.
   Definitional; the canonical derivation is
   /theory/methods/sn/curvilinear_one_group, and the term is verified
   end-to-end by the anisotropic curvilinear MMS (catches ERR-026).
.. vv-status: mm-angular-redistribution documented

.. math::
   :label: mm-angular-redistribution

   \int_{V_i}\frac{1-\mu^2}{r}\frac{\partial\psi}{\partial\mu}\,dV
   \;\approx\;
   \alpha_{m+\frac12}\,\psi_{m+\frac12}
   \;-\;
   \alpha_{m-\frac12}\,\psi_{m-\frac12},

with the geometry factor :math:`\Delta A_i / w_n` restoring per-ordinate
flat-flux consistency (without it, the cancellation that makes a spatially
uniform angular flux exact holds only in the *sum* over ordinates, not
per ordinate — the Morel–Montry flux dip near :math:`r=0`). The
half-angle fluxes :math:`\psi_{m\pm 1/2}` are not free unknowns: they are
fixed by a closure that ties each half-angle to its neighbour, which is
the source of the sequential coupling below.

**Why not a tensor product**. The Carlson starting-direction march
(Hébert 2009 §3.9.4, Eqs. 3.432-3.435) and the Morel--Montry closure it
seeds together produce an angular recurrence

.. math::
   :label: mm-half-grid-recurrence

   \alpha_{m+1/2} \;=\;
     f(\alpha_{m-1/2},\;\Sigma_t,\;\psi_{m-1/2,\,\text{upstream}})

within each μ-level :math:`p`. The :math:`\alpha_{m\pm 1/2}` are the
Carlson coupled-pole half-angle coefficients, and the recurrence on
angular ordinates is sequential along the half-grid axis with
σ_t-dependent absorption. The factor that produces
:math:`\alpha_{m+1/2}` from :math:`\alpha_{m-1/2}` IS the entire
recurrence; there is no clean per-angular-axis diagonal factor that
respects the disjoint-axes contract. A 3-factor TP wrap would
**false-assert separability** the recurrence doesn't support
(``coding-elegance`` Pattern 4 — do not represent illegal states).

This is the *angular* analogue of the spatial WDD obstruction of
:ref:`tensor-network-streaming-deep-dive`: where the WDD recurrence couples
spatial cell :math:`i+1` to cell :math:`i`, the M-M recurrence couples
angular half-face :math:`m+\tfrac12` to :math:`m-\tfrac12`. Both are
lower-triangular sweeps over a single axis with ordinate- and
material-dependent coefficients, and **both run inside the same fused
walk**, sequentially nested: the outer loop is the spatial sweep, and at
each spatial cell the inner M-M thread advances the angular half-grid.
That nesting is precisely what a :class:`SumOfTensorProductsOperator`
cannot express — a sum of per-axis tensor factors is a flat algebraic
form with no notion of one axis's recurrence running *inside* another's.

.. vv-status: mm-half-grid-recurrence documented

**Per-cell algebra**. The walk visits every :math:`(p,\,i)` pair
(μ-level × spatial cell) and calls
:meth:`AngularClosureBase.cell_contribution`. The cell-balance algebra
at :func:`orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`
decomposes additively into three terms:

.. math::
   :label: tensor-network-cell-balance-three-terms

   {\rm denom} \;=\;
     {\rm streaming\_denom\_term} \;+\;
     {\rm angular\_denom\_term} \;+\;
     {\rm collision\_denom\_term}

.. (vv-status rationale) The additive numerator decomposition
   numer_upstream = spatial_upstream + angular_numer_upstream. Structural
   decomposition, matching the sentineled tensor-network-cell-balance-
   three-terms (its denominator twin).
.. vv-status: tensor-network-cell-balance-numerator documented

.. math::
   :label: tensor-network-cell-balance-numerator

   {\rm numer\_upstream} \;=\;
     {\rm spatial\_upstream\_term} \;+\;
     {\rm angular\_numer\_upstream}

The angular-redistribution contribution to the cell balance is

.. (vv-status rationale) The angular-redistribution contribution
   m_angular_redist extracted from cell_balance_for_streaming. Structural
   decomposition; the term is verified end-to-end by the anisotropic
   curvilinear MMS (catches ERR-026).
.. vv-status: tensor-network-angular-redist-contribution documented

.. math::
   :label: tensor-network-angular-redist-contribution

   m_{\rm angular\_redist} \;=\;
     \frac{1}{V_i}\bigl[
        {\rm angular\_denom\_term} \cdot \psi_{\rm cell}
      - {\rm angular\_numer\_upstream}
     \bigr]

with :math:`{\rm angular\_denom\_term} = (\Delta A / w)\,c_{\rm out}`
and :math:`{\rm angular\_numer\_upstream} = (\Delta A / w)\,c_{\rm in}\,
\psi_{m-1/2,\,i,\,g}` per the M-M closure (see
:class:`orpheus.sn.angular.closure.AngularClosureBase`
for the closure data). It is an interior-cell operation that does not
traverse the spatial boundary; only the spatial sweep writes face
residuals.

.. vv-status: tensor-network-cell-balance-three-terms documented


Curvilinear bookkeeping (formerly the M_spatial-via-subtraction smell)
----------------------------------------------------------------------

When the retired ``M_spatial`` leaf existed, its curvilinear value was
defined by *subtracting* the angular-redistribution share from the
unified :math:`(L+C)\,\psi`:

.. math::
   :label: tensor-network-mspat-curvilinear-subtraction

   M_{\rm spatial}\,\psi \;=\; (L+C)\,\psi \;-\;
                               M_{\rm angular\_redist}\,\psi
   \qquad (\text{curvilinear})

.. vv-status: tensor-network-mspat-curvilinear-subtraction documented

#238 retired both leaves, so this subtractive definition no longer ships:
the fused matvec emits :math:`(L+C)\,\psi` directly with the
angular-redistribution term already folded into the cell balance, and the
spatial / angular split is never materialised. The algebra-decomposition
invariant test that bounded the old subtraction
(:math:`(L+C)\,\psi \equiv M_{\rm spatial}\,\psi + M_{\rm angular\_redist}\,\psi`
at principled-equivalence ULP) retired with the leaves; the surviving
guarantee that the angular term is correct is the anisotropic curvilinear
MMS (:ref:`sn-mms-curvilinear-aniso-verification`), which exercises the
full curvilinear :math:`(L+C)` end-to-end.


The 2-D Cartesian path (Q1) and its trace semantics
---------------------------------------------------

.. note:: **Retraction (2026-08-10, Issue #346).**  This section
   described the 2-D Cartesian matvec as a procedural hybrid whose
   boundary trace was *passive*, and named
   ``StreamingOperator._apply_2d_cartesian`` as its body.  Both claims
   are stale.  ``_apply_2d_cartesian`` was retired at S6.4(a) — the name
   resolves nowhere in the tree — and the trace is now the matvec's
   inflow **source** as well as the carrier of its output residual.  The
   Wave-T reasoning is preserved below in past tense; the shipped
   contract is stated first.

**The shipped contract.**  The Cartesian apply frame is
dimension-generic: :meth:`~orpheus.sn.loss_representation._OctantWalk.loss_action`
in :mod:`orpheus.sn.loss_representation` walks the octants over the
sweep-dependency graph and every loss representation supplies only its
interior kernel.  Its boundary semantics are **bare** (O.4b Phase E),
and the trace is active on both sides:

* each octant reads its **inflow from the given trace**
  ``psi.boundary`` — there is no ``bc.apply`` inside the matvec,
  because the reflective coupling is the sibling :math:`-B` operator
  (:ref:`bc-realizer-layer`);
* the domain-edge outflow is captured into ``streamed`` (OUTFLOW slots
  only), and the **output** boundary block is the O.4b active-trace
  residual — OUTFLOW slots carry the defect ``streamed − given``,
  INFLOW slots the identity ``given``.

The bulk therefore no longer proxies for the trace in either direction:
:math:`(L+C)` reads the trace, writes a trace-block residual, and
:math:`-B` supplies the re-entry.

**What this section said before, and why (Wave T, ~2026-04).**  T.4
lifted the 1-D path only.  The 2-D Cartesian path was then procedural
cell-centred upwind FD with **cell-centre-proxy boundary semantics**:
the matvec body read ``psi.bulk.values[:, :, 0, iy]`` as the outgoing
trace at xmin and the BC's ``apply(outgoing)`` filled the
incoming-direction bulk cells, leaving the face views of
``psi.boundary`` passive — their values did not enter the bulk
computation.

The trace-correct face_view formulation (face_view enters the bulk
computation as the boundary trace, with a boundary residual driving
face_view ↔ bulk consistency) caused a 10% k_inf drift in
experiments (recorded during the pre-reorganisation ``orpheus/sn/operator.py``
work, since split into ``orpheus/sn/operators/``).  That rewire was
therefore deferred: it required the BC realizers to gain a "proper
composable algebra" — a payload distinct from T.4's per-direction lift
— and bundling the two would have violated the
unify-after-two-instances discipline (only one working 2-D path existed
at the time).

⚠ **Status.**  The prerequisite landed.  The "proper composable
algebra" is the descriptor / realizer / operator architecture of
:ref:`bc-realizer-layer`, with the rank-N descriptor algebra at
:ref:`bc-rank-n-algebra`; the trace-correct formulation is the shipped
contract stated at the head of this section.  The recorded 10% drift
belongs to the pre-algebra attempt, and **no post-landing reproduction
of it is on record** — what is on record is that the current path is
gated by the closed-form :math:`k_\infty` pillar
(``tests/sn/verification/analytical/test_kinf_homogeneous.py``).  A
future session wanting the causal story has to re-run the experiment on
the landed form; this page does not answer it.

**Defensive A2D-1 source-hash pin**. T.4d added a structural
regression test that recorded the source-code signature of
``_apply_2d_cartesian`` and asserted it remained unchanged, so an
accidental modernisation of the 2-D path could not ship silently.
*(RETIRED at S6.4(a): the body became the SHARED ``_OctantWalk``
apply frame, where a source-hash trips on every legitimate refactor
with no behavior signal; the ``window ≡ full`` matvec
``assert_array_equal`` oracle is the successor tripwire.)*


Verification approach
---------------------

Wave T's verification chain combines three independent grounds:

1. **Pre-T.4 snapshot bit-identity** (Route A). The substep T.4a
   captured the value of :meth:`apply` and :meth:`solve` on fixed
   :math:`(\text{seed}, \text{mesh}, \text{material})` triples across
   slab, sphere, cylinder, and 2-D Cartesian, plus 1G / 2G / asymmetric
   :math:`\Sigma_s` / vacuum / white / specular variants. Each
   subsequent T.4 substep is gated on :func:`numpy.array_equal` against
   those snapshots — the existing numerics are the local
   bit-identity reference.

2. **Principled-equivalence ULP** for cases where reductions reorder.
   Per the ``vv-principles`` skill §"Bit-identity
   vs principled-equivalence": when the operator-algebra fold inserts
   a :func:`numpy.add` at a different position than the legacy fused
   einsum, the new value is verified by the three-criteria gate
   (principled at every step / structurally-independent reference /
   FP-non-associativity dimensionally explainable). For the
   curvilinear :math:`M_{\rm spatial} = (L+C) - M_{\rm angular\_redist}`
   subtraction, the algebra-decomposition invariant passes at
   ~16×ULP.

3. **Structural-independence ground at L1**. The pre-snapshot
   regression tests are bit-identity against the OLD code; they
   cannot catch a bug that was ALREADY in the old code and survived.
   The L1 ground truth is two-pillared (per
   the ``vv-principles`` skill):

   - **Closed-form pillar**: :math:`k_\infty = \nu\Sigma_f / \Sigma_a`
     on homogeneous reflective slab / sphere / cylinder. Verified at
     ``tests/sn/verification/analytical/test_kinf_homogeneous.py``. This is the
     eigenvalue reference — MMS does NOT prove eigenvalues per
     the ``vv-principles`` skill §"What each pillar
     proves".

   - **MMS pillar**: P1 anisotropic manufactured-source convergence
     at ``tests/sn/verification/mms/test_mms_aniso.py``,
     ``tests/sn/verification/mms/test_curvilinear_aniso_convergence.py``,
     ``tests/sn/verification/mms/test_mms_heterogeneous.py``, and
     ``tests/sn/verification/mms/test_mms_2d.py``. The MMS source is structurally
     independent of the operator-algebra path (derived by SymPy in
     ``orpheus/derivations/continuous/mms/sn.py``); it catches flux-shape and
     convergence-order errors that snapshot bit-identity cannot.

4. **Algebraic-identity gates** (new in Wave T). Each touched
   operator passes the algebra contracts:

   - :meth:`~orpheus.numerics.operator.SumOfTensorProductsOperator.assert_separable`
     passes on every TP-shaped operator (BC realizers, fission). The
     method's home is the SUM-of-tensor-products operator, not the bare
     :class:`~orpheus.numerics.operator.TensorProductOperator`: a single
     tensor product is the one-term case of the sum, and the separability
     contract is stated once, on the type that can hold either. It stays
     structurally inapplicable to :class:`OperatorSum`-of-bespoke-leaves
     (T.3 scattering kernel; the T.4 streaming matvec, fused since #238)
     — see the T.5.1 deviation note in
     ``.claude/plans/archive/wave_t_tensor_network.md`` §6, which records
     that Q6/Q1–Q5 shipped ``OperatorSum``-of-bespoke-leaves instead of
     the spec's sum-of-tensor-products for those two leaves.

   - **(#238 retired)** the Wave-T algebra-decomposition invariant
     :math:`(L+C)\,\psi \equiv M_{\rm spatial}\,\psi +
     M_{\rm angular\_redist}\,\psi` pinned the typed-leaf split; with the
     split removed the surviving guarantee that the curvilinear
     angular-redistribution term is correct is the anisotropic
     curvilinear MMS (:ref:`sn-mms-curvilinear-aniso-verification`).

   - :math:`(L+C).{\rm solve}(q)` bit-identical pre/post-Wave-T,
     verifying the WDD sweep procedural inverse was NOT touched
     (the :class:`StreamingCollisionOperator.solve` body runs the procedural
     algorithm on the operator's own
     :attr:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.loss_representation`
     since S6.5 — at Wave T it was the free function ``transport_sweep``).

5. **Performance regression gate**. The 1-D slab Krylov benchmark
   measured median 1.04× pre-T.4 baseline (under the 5% threshold).
   (#238 retired the ``M_spatial`` / ``M_angular_redist`` cached
   properties along with the typed-leaf split; the fused matvec carries
   no per-leaf construction cost.)


What :class:`SumOfTensorProductsOperator` was supposed to do — and didn't
-------------------------------------------------------------------------

Grand Report v3 §15.2 (lines 2046-2086) names the canonical scattering
form

.. (vv-status rationale) The canonical SOTP scattering form
   S = Σ_ℓ Σ_{s,ℓ} ⊗ A_ℓ ⊗ G_ℓ (Grand Report v3 §15.2). A
   literature-transcribed form documenting a design that was NOT realized
   (the per-cell material id breaks separability; an OperatorSum of
   bespoke per-ℓ leaves shipped instead) — it names a non-built structure.
.. vv-status: sotp-scattering-form documented

.. math::
   :label: sotp-scattering-form

   S \;=\; \sum_{\ell=0}^{L}
     \Sigma_{s,\ell}\, \otimes\, A_\ell\, \otimes\, G_\ell

with :math:`A_\ell` the angular Pℓ-projection factor,
:math:`\Sigma_{s,\ell}` the per-ℓ group-coupling factor, and
:math:`G_\ell` the per-ℓ spatial factor. Wave T's original T.3 plan
targeted this SOTP form for the scattering kernel.

The design fork (T.3 spec Q6) surfaced that the per-material per-ℓ
einsum — then ``MaterialXSField.apply_legendre_scattering_moments``,
since the CS4c step-3 O-6 move
:meth:`TransferMaterialField.moment_source
<orpheus.transport.material_field.TransferMaterialField.moment_source>`
—
**couples the group axis with the spatial axis** — the per-cell
material id ``cells_by_material[mid]`` selects the per-material
scattering matrix :math:`\Sigma_{s,\ell}^{m(\vec r)}`. There is no
factor design where one factor acts on the group axis alone and
broadcasts on the spatial axis; the per-cell material id breaks the
broadcast contract.

The user-resolved math-honest fallback shipped at commit ``9f85c5d``:
:class:`OperatorSum` over per-ℓ ``_PerLegendreOrderScattering``
bespoke leaves. The §15.2 *form* is preserved at the summation level
(one summand per Legendre order); the per-summand decomposition into
:math:`R_\ell \circ \Lambda_\ell \circ M_\ell` is a procedural
composition, not a tensor product. (This per-ℓ ladder was itself
later retired at commit ``93807aa7`` in favour of the single
Funk–Hecke :math:`R \circ \Lambda_{\ell\ge 1} \circ M` moment-space
kernel now carried by :attr:`ScatteringOperator.kernel`.)

The same master condition applies to T.4-spatial (per-direction WDD
recurrence) and T.4-curvilinear (M-M half-grid recurrence). Two of
the three originally-SOTP-targeted Wave-T substeps fell back to
:class:`OperatorSum`-of-bespoke-leaves; only T.1 (BC realizers) and
T.2 (fission rank-1) cleanly support the TP form.

**Implication for Wave O (Issue #208)**. The operator-role typing
work MUST accommodate non-SOTP :class:`OperatorSum` summands. Any
contract of the form "every BulkOperator summand IS a
:class:`TensorProductOperator`" forecloses scattering, streaming
spatial, and curvilinear angular redistribution. The five-shape
catalogue in :ref:`tensor-network-shape-table` is the constraint the Wave O
typing must respect.


Cross-references
----------------

- **Wave T plan** (canonical reference for substep sequencing,
  architectural decisions, deviations from §15.2):
  ``.claude/plans/wave_t_tensor_network.md``.
- **T.4 verification spec** (Q1-Q5 architectural decisions, risk
  register, test catalogue):
  ``.claude/agent-memory/test-architect/wave_t_t4_streaming_verification_spec.md``.
- **Grand Report v3** §15 (V = X ⊗ Ω ⊗ G), §15.1 (streaming as sum of
  tensor products), §15.2 (scattering as sum of tensor products),
  §16A.10 (BC as tensor network), §35 (commandments), north-star line
  5697.
- **Shipped commits**: ``fa13e78`` (T.1 BC), ``0b2848b`` (T.2
  fission), ``9f85c5d`` (T.3b kernel), ``03bcdba`` (T.3c
  build_aniso_source rewire), ``cb18fdb`` (T.4a snapshots),
  ``c55b505`` (T.4b slab M_spatial), ``90e7d4e`` (T.4c curvilinear
  M_angular_redist).
- **Code anchors**:

  - :class:`orpheus.numerics.operator.TensorProductOperator`,
    :class:`orpheus.numerics.operator.SumOfTensorProductsOperator`,
    :class:`orpheus.numerics.operator.OperatorSum`,
    :class:`orpheus.numerics.operator.RankOneOperator`,
    :class:`orpheus.numerics.operator.IdentityOperator`,
    :class:`orpheus.numerics.operator.ZeroOperator`.
  - :class:`orpheus.sn.boundary.realizer.SNBoundaryRealizer` —
    the BC realizer dispatching the T.1 lifts.
  - :class:`orpheus.transport.operators.isotropic_transfer.IsotropicFission`
    and its
    :attr:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission.kernel`
    property (the dyad's arithmetic home since CS4c step 4);
    :class:`orpheus.transport.operators.fission.FissionOperator`, whose
    same-named property delegates to it.
  - :class:`orpheus.transport.operators.scattering.ScatteringOperator` and its
    :attr:`~orpheus.transport.operators.scattering.ScatteringOperator.kernel`
    property.
  - :class:`orpheus.sn.operators.streaming.StreamingOperator` and its
    :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply` /
    :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply_transpose`
    public matvec surface. (#238 retired the per-direction
    ``M_spatial`` / ``M_angular_redist`` typed-leaf split — the
    ``_SpatialSweepDirection`` / ``_MSpatialOperatorSum`` /
    ``AngularRedistributionOperator`` leaves had no production
    consumer.)
  - :meth:`orpheus.sn.loss_representation._OneDimScanWalk._apply_walk`
    — the fused single-emission 1-D matvec body (``(L+C)ψ``), the
    apply-direction twin of the sweep. #206 Phase C moved the walk off
    the operator INTO the representation; #238 removed the dual-emission
    ``(M_spatial, M_angular_redist)`` arm (no production consumer). The
    public surface is :meth:`StreamingOperator.apply`.
  - :class:`orpheus.sn.angular.closure.AngularClosureBase`
    — the M-M closure data and per-cell algebra primitive.
  - :func:`orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`
    — the three-term cell-balance primitive.
