.. _theory-sn-index-convention:

==========================================================
Indexing and Layout --- ``(N, ng, nx, ny)``
==========================================================

**The storage-layout half of the SN conventions.** This page is the
canonical statement of the storage layout for every SN solver array ---
its derivation, the six-PR migration that installed it, the
load-bearing bit-identity gate that verified it, and the typed-field
vocabulary built on top. The cross-section *conventions* it once also
carried --- the ``[g_from, g_to]`` scattering-matrix law and the
per-material / per-cell split --- now live on the sibling
:doc:`cross_section_conventions` page.

.. contents:: Contents
   :local:
   :depth: 3


Key Facts
=========

**Read this before reading or writing SN array code.**

This page is the canonical statement of the storage layout for every
SN solver array.  It supersedes every per-file docstring that
described a different layout; if a per-file docstring disagrees with
this page, this page is correct.

- **Angular flux** :math:`\psi`: ``(N, ng, nx, ny)`` --- ordinate
  index first, energy second, spatial last.  1-D problems use
  ``ny = 1`` (the trailing axis is preserved as a singleton, NOT
  squeezed).
- **Scalar flux** :math:`\phi` and **cross sections** (:math:`\Sigma_t`,
  :math:`\Sigma_a`, :math:`\Sigma_p`, :math:`\nu\Sigma_f`, :math:`\chi`):
  ``(ng, nx, ny)`` --- energy first, spatial last.  Same trailing
  ``ny = 1`` rule for 1-D.
- **External source** :math:`q`: ``(ng, nx, ny)`` (isotropic) or
  ``(N, ng, nx, ny)`` (anisotropic / :term:`per-ordinate <ordinate>`).
- **Cell-flattening invariant**: the principled storage round-trips
  with the legacy one under transpose:
  ``xs.sig_t.T.reshape(ng, nx, ny)[g, i, j] ==
  xs.sig_t.reshape(nx, ny, ng)[i, j, g]``.  Asserted in ``__debug__``
  at :class:`~orpheus.sn.solver.SNSolver` construction.
- **The six-operator algebra** — the within-group loss composite
  :math:`A = L + C - S - N_{2n} - B`, posed as
  :math:`A\,\psi = \tfrac{1}{k}\,F\,\psi` (eigenvalue) or
  :math:`A\,\psi = q` (fixed source) — consumes and returns :math:`\psi`
  shaped as ``(N, ng, nx, ny)`` at
  every leaf (:class:`~orpheus.sn.operators.streaming.StreamingOperator`,
  the collision multiplier :math:`C = M[\sigma_t]`
  (:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`),
  :class:`~orpheus.transport.operators.scattering.ScatteringOperator`,
  the :math:`(n,2n)` emission
  :class:`~orpheus.transport.operators.n2n.N2NOperator`,
  the boundary law :class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`
  — a first-class **sibling**, *not* folded into :math:`L` —
  :class:`~orpheus.transport.operators.fission.FissionOperator`).
- **Historical note (resolved 2026-05)**: a legacy FD-matvec
  packed-vector helper ``solution_to_angular_flux`` returned
  ``(ng, N, nx, ny)`` internally with a Krylov-side
  :func:`numpy.transpose` adapter.  The entire helper family
  (``EquationMap`` / ``solution_to_angular_flux*`` /
  ``pack_with_traces``) retired in Depth B D-J (commit ``4a53737``)
  alongside the bare-ndarray operator contract; the principled
  ``(N, ng, nx, ny)`` layout is now universal on the typed
  :class:`~orpheus.transport.fields.angular_flux.AngularFlux` carrier.

.. note:: **The spatial tail is rank-generic** (Issue #225 / C5).

   ``(N, ng, nx, ny)`` is the :math:`d \le 2` spelling of the
   rank-generic layout ``(N, ng, *spatial_shape)``.  At :math:`d = 1`,
   ``spatial_shape == (nx,)`` (written ``(N, ng, nx, ny)`` with the
   trailing ``ny = 1`` singleton preserved, **never** a phantom second
   axis — see below).  At :math:`d = 3` an axis-native :class:`SNMesh`
   (:ref:`sn-axis-primary-c5`) produces ``(N, ng, nx, ny, nz)``.
   Energy-first / spatial-last and ordinate-first hold at every rank;
   only the **length** of the spatial tail changes.  Every field /
   cross-section / scattering read since C5.2 keys on the rank-generic
   :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.spatial_shape`, **not** on a
   hard-coded ``(nx, ny)`` pair (an ``(nx, ny)``-keyed read silently
   truncates a 3-D tensor — the live :math:`d = 3` landmine C5.2
   retired; see :ref:`sn-c5-phantom-retirement`).  The :math:`d \le 2`
   bytes are unchanged (affine ``sha256`` goldens identical across the
   carve).

.. admonition:: Authoritative origin

   The derivation of this layout is encoded in
   :ref:`sn-index-convention-derivation`.  The full migration
   narrative --- six commits between ``c21c2ef`` and ``3356cec`` that
   flipped the codebase from ``(N, nx, ny, ng)`` to
   ``(N, ng, nx, ny)`` --- is recorded in
   :ref:`sn-index-convention-history`.  Numerical evidence
   (11/11 regression snapshots bit-identical at ``rtol=1e-12`` via the
   load-bearing Step-1 bit-identity-via-transpose gate) is in
   :ref:`sn-index-convention-numerical-evidence`.


Overview
========

The SN solver discretises four indices:

- :math:`n` --- ordinate (angular direction), :math:`n = 0, \ldots, N-1`.
- :math:`g` --- energy group, :math:`g = 0, \ldots, n_g - 1`.
- :math:`i, j` --- spatial cell, :math:`i = 0, \ldots, n_x - 1`,
  :math:`j = 0, \ldots, n_y - 1` (with :math:`n_y = 1` for 1-D).

A storage decision picks one ordering of these axes for every flux,
source, and cross-section array.  The decision is consequential: it
affects how every operator-leaf ``apply`` body indexes, how the
:term:`sweep`'s hot loop traverses memory, and how a future JAX or GPU port
maps batched dimensions to the device grid.

Before Issue #196 the codebase carried ``(N, nx, ny, ng)`` for
:math:`\psi` (energy trailing) and ``(nx, ny, ng)`` for everything
scalar.  That layout was historical: it descended from a 2-D-Cartesian
prototype in which group iteration was the outermost Python loop, so
``g`` trailing was the most natural place to put it.  The principled
order --- derived in :ref:`sn-index-convention-derivation` from the
block-diagonality of the within-group system plus the
streaming-dependency-only-on-cell-axes --- is the opposite:
:math:`g` belongs *second*, immediately after :math:`n`, with the
must-iterate cell axes last.

The migration from the historical layout to the principled one was
done in six PRs over a single branch (``refactor/sn-operator-algebra``)
in May 2026.  Every intermediate commit kept
``tests/sn/regression/`` green at ``rtol=1e-12``.  The final commit
(``3356cec``) regenerated the regression snapshots under the
principled layout *only after* a bit-identity-via-transpose gate
demonstrated that every one of the 11 snapshots agreed with the
legacy layout to ``rtol=1e-12, atol=1e-13``.


.. _sn-index-convention-derivation:

Derivation --- why ``(N, ng, nx, ny)`` is principled
====================================================

The choice of axis order is dictated by the structure of the
within-group transport system.  For a single energy group :math:`g`,
the discrete ordinates equation is

.. math::
   :label: sn-within-group-system

   (L_g + C_g)\,\psi_g \;=\; S_g\,\phi + N_{2n,g}\,\phi
   + \frac{1}{k}\,\chi_g\,F\,\phi + q_g\,,

.. (vv-status rationale) definition: the governing within-group system stated
.. here to motivate the ``(N, ng, nx, ny)`` storage layout (the block-diagonal
.. -in-g structure that makes energy the joint-batch axis). Definitional /
.. representational — the within-group solve it names is verified by the SN
.. within-group operator gates, not by this layout-derivation restatement.
.. vv-status: sn-within-group-system documented

where :math:`L_g` is the streaming operator on group :math:`g`,
:math:`C_g = \Sigma_{t,g}\,\mathbb{I}` is the diagonal collision
operator, :math:`S_g` accumulates the in-scatter contribution,
:math:`N_{2n,g}` the :math:`(n,2n)` in-transfer — the same binding as
:math:`S_g` with the yield :math:`\nu_{2n} = 2`, first-class since CS4c
step 3 (:ref:`the two collision gains <operator-algebra-two-gains>`) —
and the right-hand side carries the fission source and external source.
The *within-group* system is the per-:math:`g` problem when the
scattering source is held fixed at the current outer iterate ---
exactly what each inner source iteration solves.

Two observations make the storage decision:

1. **No cross-group coupling within a sweep.** The within-group
   problem :eq:`sn-within-group-system` is solved for each :math:`g`
   independently --- the operator :math:`L_g + C_g` is
   *block-diagonal in g*.  Once :math:`\phi^{(k)}` is fixed for the
   inner iteration, the energy group is the obvious
   joint-batch axis: every group-independent quantity (the
   streaming operator, the collision diagonal, the per-ordinate
   intermediate arrays) can be processed in lockstep across all
   groups using a single numpy expression.  This is the same
   structural reason Galerkin spectral methods batch over the
   variational basis.

2. **No cross-ordinate coupling for the within-group P\ :sub:`0`
   problem.** When scattering is :math:`P_0` (isotropic), the
   right-hand side is a :term:`scalar flux` :math:`\phi(r)` that does not
   depend on the outgoing ordinate.  Different ordinates therefore
   solve *the same source* through different streaming directions,
   and an outer Krylov batch can compute their residuals in parallel.
   The block structure is therefore **block-diagonal in both g and
   n** at the within-group P\ :sub:`0` level.  Curvilinear angular
   redistribution and :math:`P_\ell` anisotropic scattering reduce
   this independence but do not destroy it: the redistribution is
   one tight ordinate band (the M--M angular thread per
   :math:`\mu`-level), not the full :math:`N \times N` dense
   coupling.

3. **The spatial axes are the only must-iterate axes.** The
   streaming term :math:`\mu_x \partial_x \psi + \mu_y \partial_y
   \psi` connects every cell to its upwind neighbour.  Sweep order is
   dictated by the DAG of cell dependencies; the sweep
   *fundamentally cannot* be parallelised across cells along the
   streaming direction.  The cell axes are therefore the innermost
   axes --- the ones we *want* to traverse sequentially.

These three observations give the priority ordering for storage axes:

.. list-table:: Axis-priority table (principled storage)
   :header-rows: 1
   :widths: 12 32 56

   * - Index
     - Within-group coupling
     - Storage role
   * - :math:`n` (ordinate)
     - None for within-group P\ :sub:`0`; tight band for
       curvilinear / P\ :sub:`\ell`
     - **Outermost** --- sweep iterates over chains
       outside the per-ordinate body; Krylov batches
       across :math:`n`
   * - :math:`g` (group)
     - None for within-group
     - **Second** --- block-diagonal axis, joint-batched in
       every per-ordinate kernel
   * - :math:`i, j` (cells)
     - Streaming dependency chain
     - **Innermost** --- the only axes that *must* iterate
       sequentially

The principle generalises:

   **In a tensor-product discretisation, axes with no cross-coupling
   for the within-group system belong before the axes that carry a
   sequential dependency.**

:cite:`LewisMiller1984` §4.5 ("Source Iteration") gives the same
block-diagonal structure as the textbook proof that the within-group
problem decouples; :cite:`AdamsLarsen2002` §III confirms the same picture
for the SAILOR preconditioner family.  The block structure is the
mathematical reason every modern transport code (PARTISN, Denovo,
JAGUAR, OpenMOC) carries the same ``(angular, energy, spatial)``
priority --- the storage layout is dictated by the operator algebra,
not by a historical implementation choice.

Algorithmic consequence
-----------------------

Under the principled layout, the per-sweep hot path

.. code-block:: python

   # For SLAB: ordinates within a chain are not coupled (no M-M thread).
   # Joint-batch over (chain_size, ng, nx) — one scan per chain.
   psi_face_chain = ordinate_scan(
       a_atten_chain.T,    # (nx, K, ng) — scan axis leads (Blelloch)
       b_chain.T,          # (nx, K, ng)
       psi_in,
   )

becomes a single
:func:`~orpheus.sn.sweep.scan.ordinate_scan` call per chain (two
chains per slab problem), rather than ``N/2`` per-ordinate calls.

The closed-form scan that
:func:`~orpheus.sn.sweep.scan.ordinate_scan` evaluates is the
**Blelloch §1.5 first-order linear-recurrence form**.  For the
per-ordinate spatial recurrence
:math:`\psi[i+1] = a[i]\,\psi[i] + b[i]` with :math:`\psi[0]
= \psi_0` (forward substitution on the block-triangular streaming +
collision operator), the prefix-product factorisation of the
associated 2×2 lower-triangular affine matrix
:math:`M = [[a,0],[b,1]]` gives

.. math::
   :label: blelloch-1990-eq-1-5

   \psi[n] \;=\; \left(\prod_{i=0}^{n-1} a[i]\right)
                 \left(\psi_0 \;+\; \sum_{i=0}^{n-1}
                       \frac{b[i]}{\prod_{j=0}^{i} a[j]}\right).


.. implements:: blelloch-1990-eq-1-5
   :by: orpheus.sn.sweep.scan.ordinate_scan

   **Implemented by** the one site in the tree that executes this
   equation's arithmetic.

In numpy this is three ops:
``cumprod(a) * (psi_0 + cumsum(b / cumprod(a)))`` —
no Python loop over cells.  The pair-monoid composition
:math:`(\alpha_1,\beta_1) \oplus (\alpha_2,\beta_2) =
(\alpha_1 \alpha_2,\, \alpha_2 \beta_1 + \beta_2)` is associative
(:cite:`Blelloch1990` §1.5; :cite:`Brent1974`), so the same closed form admits
Brent's :math:`O(N/\log N)`-work parallel decomposition if a future
GPU port adopts a parallel-prefix backend.  The
``tests/sn/sweep/core/test_ordinate_scan.py`` algebraic-theorem suite
pins the pair-monoid associativity, identity, linearity in
:math:`\psi_0`, linearity in :math:`b`, and bit-identity to a serial
explicit loop — fifteen foundation-level invariants that justify
the implementation, not the other way around.
For ``N = 16``, this saves 14 Python invocations per sweep ---
roughly 28 % mean speedup on the
``slab_2g_3reg_dd_n40`` regression case (PR-INDEX-1 benchmark; see
:ref:`sn-index-convention-numerical-evidence`).

The same principle would let curvilinear sweeps joint-batch over
groups too, but the M--M angular thread is sequential *across*
ordinates within a :math:`\mu`-level, so the curvilinear sweep keeps
the per-ordinate scan with ``(ng, nx)`` joint batching only.  A
parallel-prefix reformulation of the M--M recurrence (research-level
algorithm work) could lift this restriction; see
:ref:`sn-index-convention-future-work`.

.. _sn-cell-flattening-invariant:

Cell-flattening invariant
-------------------------

The principled storage must agree with the legacy storage under a
pure-transpose round-trip.  The check is

.. math::
   :label: sn-cell-flatten-roundtrip

   \texttt{sig\_t}_{\text{principled}}[g, i, j]
   \;=\;
   \texttt{sig\_t}_{\text{legacy}}[i, j, g]
   \qquad \forall (g, i, j)\,,

.. (vv-status rationale) representational: the pure-transpose bit-identity
.. invariant between the principled and legacy cell-flattened layouts. Its
.. verifiable content is a bit-identity contract, pinned by the in-``__init__``
.. ``np.array_equal`` transpose assert (PR-INDEX-3) and the 11-snapshot
.. bit-identity-via-transpose regression gate — a foundation/bit-identity
.. check, not a physics ``verifies`` claim.
.. vv-status: sn-cell-flatten-roundtrip documented

implemented at :meth:`SNSolver.__init__`:

.. code-block:: python

   xs = assemble_cell_xs(materials, sn_mesh.mat_map)
   self.sig_t = xs.sig_t.T.reshape(self.ng, nx, ny)
   if __debug__:
       _sig_t_old = xs.sig_t.reshape(nx, ny, self.ng)
       assert np.array_equal(
           _sig_t_old, self.sig_t.transpose(1, 2, 0)
       ), "PR-INDEX-3 cell-flattening invariant broke"

The invariant is load-bearing: it detects accidental mat-ids ravel
order changes (Fortran vs C order) that would silently corrupt the
spatial-to-group mapping.  An assertion failure here would surface as
a clean test failure rather than a flux distribution that looks
plausible but is wrong by a permutation of cells.


.. _sn-index-convention-history:

History --- the six-PR migration
================================

The migration unfolded as a six-commit chain on the
``refactor/sn-operator-algebra`` branch between 2026-05-14 and
2026-05-15.  Each PR kept ``tests/sn/regression/`` green at
``rtol=1e-12`` by inserting temporary bridge transposes at the
boundary between flipped (principled) and unflipped (legacy) layers.
The bridges were named ``BRIDGE_*_to_principled`` /
``BRIDGE_*_to_legacy`` so a grep-tag retired them as the migration
progressed.

The proposal that was wrong
---------------------------

The initial typed-field contract memo
(``.claude/agent-memory/explorer/typed_field_contracts_for_phase_g.md``,
committed at ``9d74184``) proposed ``(N, nx, ny, ng)`` as the
canonical storage --- energy trailing.  The memo's argument was
operational: numpy ``block_op @ flux.values`` works with the last
axis as the contraction axis, and many group-block matrices act on
:math:`g` as the inner axis.  That argument is locally correct but
**inverts the coupling priority**: it puts the must-iterate cell axes
``(nx, ny)`` *before* the block-diagonal group axis ``ng``, which
forces every per-:math:`g` numpy operation to reach across the cell
axes to find its group block.  For a strided memory traversal this
wastes cache; for a future GPU port it would map the
block-diagonal-but-stride-:math:`n_x \cdot n_y` axis to the wrong
grid dimension.

The wrong proposal was caught **before implementation** by re-reading
the derivation table in this page's §1 against the memo's §1.1.  The
discovery point is documented in the migration plan (`§1.1 "Why we
paused the typed-field contract plan to do this first"
<../../.claude/plans/principled_index_migration.md>`_) and is the
canonical example of:

   **When a refactor calls for new types AND a layout change, do the
   layout change first on bare arrays.  Types ossify the layout;
   flipping bare arrays is mechanical, flipping a layout that's
   encoded in twelve dataclass ``__init__``\ s + their dunder
   consumers is not.**

This is a concrete instance of the *defer abstraction until you have
evidence* principle from the project's ``coding-elegance`` skill
(Pattern 6): the team had **one** concrete instance of "the layout
we want" (the four-operator algebra acceptance criterion), but the
layout itself was still wrong.  Build the layout first; build the
abstraction on top.

The six PRs
-----------

.. list-table:: Migration commit chain
   :header-rows: 1
   :widths: 14 12 74

   * - PR
     - Commit
     - Scope
   * - PR-INDEX-1
     - ``e09b9f8``
     - ``_run_1d_sweep`` internal layout flipped to principled
       ``(N, ng, nx, ny)``; slab joint-batch ``ordinate_scan`` over
       ``(chain_size, ng, nx)`` --- 2 scan calls per sweep replacing
       ``N/2``.  Public ``transport_sweep`` signature unchanged
       (entry / exit transposes carry the legacy boundary).  11/11
       regression bit-identical at ``rtol=1e-12``; 26/26 L0
       streaming-equilibrium curvilinear; 312/312 spatial.
       ~28 % mean speedup on the slab benchmark.
   * - PR-INDEX-2
     - ``6cfdfd4``
     - :class:`~orpheus.sn.sweep.cache.CollisionCache` field
       layout flipped to ``(N, ng, nx)`` natively (``a_attenuation``,
       ``inverse_denom``, etc.); cumprod axis updated 1→2; slab
       ``np.swapaxes`` and curvilinear ``.T`` bridges at cache-read
       sites retired.  ``StreamingCoefficientCache`` untouched (no group
       axis).  New transient bridge at the
       ``CollisionCache.from_geometry`` callers in
       :class:`SNSolver` (PR-INDEX-3 removes).  Mean
       0.149 ms/sweep on the slab benchmark (down from 0.21 ms at
       PR-INDEX-1); benchmark variance tightened from 2× to 1.06×.
   * - PR-INDEX-3
     - ``313f510``
     - :class:`SNSolver` cross-section storage flipped:
       ``sig_t / sig_a / sig_p / chi`` from ``(nx, ny, ng)`` to
       ``(ng, nx, ny)`` via
       ``xs.<field>.T.reshape(ng, nx, ny)`` at ``__init__``.
       Producer :func:`~orpheus.data.macro_xs.assemble_cell_xs`
       **unchanged** (CP no-regression guaranteed by construction).
       PR-INDEX-2 transient bridges removed; new transients added at
       :meth:`FissionOperator.apply` legacy return contract and
       2-D wavefront ``transport_sweep`` body (PR-INDEX-4 removes).
       ``np.einsum`` rewrites at every reduction site (named-
       intermediate Pattern 3).  ``__debug__`` cell-flattening
       invariant assertion added at ``__init__``.
   * - PR-INDEX-4
     - ``fa41767``
     - Operator-leaf ``apply`` PUBLIC contracts flipped to principled:
       :meth:`FissionOperator.apply` returns ``(ng, nx, ny)``;
       :meth:`ScatteringOperator.apply` returns ``(N, ng, nx, ny)``;
       ``LegendreMomentScattering`` (today
       :class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`)
       consumes/returns the
       principled moment layout
       ``(L+1, 2L+1, ng, nx, ny)``;
       ``DiamondDifference.update_batch`` (the batched DD kernel of the
       day; collapsed into the storage-free
       :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch`
       pair at S6.4(e)) consumes ``(ng, ...)`` slices;
       ``_sweep_jacobi`` body principled.  PR-INDEX-3 bridges
       at ``fission.py:175`` and ``sweep.py:127`` retired.  Fourteen
       new ``BRIDGE_*`` named intermediates at 11 :class:`SNSolver`
       consumption sites + 3 ``sweep.py`` entry/exit points
       (PR-INDEX-5 removes).  ``EquationMap`` packed-vector
       traversal **deferred** to PR-INDEX-7
       (see :ref:`sn-index-convention-future-work`) — PR-INDEX-7 has
       since been obsoleted: the packed-vector path (``EquationMap``
       codec + ``solution_to_angular_flux*``) was deleted at #197
       Depth-B D-J, so this traversal migration never landed (see the
       obsoleted-banner in the PR-INDEX-7 section).
   * - PR-INDEX-5
     - ``3356cec``
     - Public API flip:
       ``SNFixedSourceResult`` / ``SNResult`` (RETIRED in Issue #197
       PR-TYPED-5; now :class:`~orpheus.sn.solution.Solution`)
       storage flipped to ``(N, ng, nx, ny)`` / ``(ng, nx, ny)``;
       ``transport_sweep`` PUBLIC contract principled;
       :func:`~orpheus.sn.solver.solve_sn` /
       :func:`~orpheus.sn.solver.solve_sn_fixed_source` return shapes
       flipped; external-source contract ``(N, ng, nx, ny)``.
       Eleven regression snapshots regenerated under principled
       layout via the load-bearing
       :ref:`Step-1 bit-identity-via-transpose gate
       <sn-index-convention-step1-gate>` (ALL 11 cases PASS at
       ``rtol=1e-12``, max abs diff 1.75 × 10\ :sup:`-14`).  ALL
       fourteen PR-INDEX-4 ``BRIDGE_*`` named intermediates retired.
   * - PR-INDEX-6
     - **this PR**
     - Documentation deliverable: this page; cross-references from
       :ref:`theory-discrete-ordinates` and
       :ref:`operator-algebra`; sweep audit of legacy-shape
       mentions in code docstrings + comments + tests.  No
       production code semantics touched.

What stayed deliberately legacy: the FD-matvec internal contract (historical)
-----------------------------------------------------------------------------

.. note:: **Deleted path (typed-field campaign, #197 / Depth-B
   D-H–D-J).**  This subsection describes the packed-vector internal
   layout of the FD-matvec helpers, which at the time were *kept*
   deliberately legacy pending a PR-INDEX-7 migration.  That whole path
   — the ``transport_operator_matvec_*`` family, the
   ``solution_to_angular_flux*`` unpackers, and the ``EquationMap``
   codec — was subsequently **deleted**: operators now consume the
   single typed :class:`~orpheus.transport.full_field.FullField`
   composite (see :ref:`theory-sn-index-convention`'s principled
   ``(N, ng, nx, ny)`` layout above), so there is no packed vector left
   to migrate and **PR-INDEX-7 is moot**.  The description below is
   retained as a record of the packed-vector convention that once
   existed.

The FD-matvec internal helpers (``solution_to_angular_flux`` and
``transport_operator_matvec_cylindrical``, plus Cartesian / spherical
analogues — all since deleted) carried a separate internal
packed-vector convention: the flat layout
:math:`\texttt{flux}[g + n_g \cdot k_{\text{eq}}]` where
:math:`k_{\text{eq}}` enumerated cells in the order
``for iy: for ix: for n:`` so :math:`n` was the next-fastest axis
after :math:`g`.  The unpacked helper returned
``fi.shape == (ng, N, nx, ny)`` --- a *third* layout, distinct from
both the legacy ``(N, nx, ny, ng)`` and the principled
``(N, ng, nx, ny)``.

Flipping the FD-matvec internal layout would touch 200+ lines of code
across :mod:`orpheus.sn.operators.streaming` (30+ ``fi[:, n, i, j]`` indexing
sites) plus the two ``np.transpose`` axis-swap adapters at
``solver.py:1361`` and ``solver.py:1408``.  The migration plan
defers this to **PR-INDEX-7** for two reasons:

1. The packed vector is **not user-facing**.  Callers see the public
   ``angular_flux`` and ``scalar_flux`` arrays (now principled) ---
   the packed vector lives between the Krylov solver and the matvec
   primitive.  Its convention is an implementation detail of the
   FD-matvec path.
2. Bit-identity at the public boundary is preserved by two
   *zero-copy* :func:`numpy.transpose` adapter calls at the Krylov
   decode sites (``solver.py:1361`` and ``solver.py:1408``).  A
   transpose of leading axes is a stride-only view, NOT a memory
   copy --- the runtime cost is one numpy header allocation per
   GMRES iteration, well below measurement noise.

See :ref:`sn-index-convention-future-work` for the PR-INDEX-7 scope.


.. _sn-index-convention-step1-gate:

The load-bearing Step-1 bit-identity gate
=========================================

The migration's most consequential single step was the
**bit-identity-via-transpose verification** that ran at PR-INDEX-5
*before* the regression snapshots were regenerated.  Without this
gate, the principled-layout snapshots would have been written to disk
without independent verification that they corresponded to the same
flux distribution as the legacy snapshots --- and any subsequent
regression test would have been measuring agreement with a
potentially-wrong reference.

The gate is the following Python:

.. code-block:: python

   for case in CASES:
       snap_file = SNAPSHOT_DIR / f'{case.name}.npz'
       old = np.load(snap_file)
       old_sf = np.asarray(old['scalar_flux'], dtype=np.float64)
       cfg = case.builder()
       result = run_case(cfg)
       new_sf = np.asarray(result.scalar_flux, dtype=np.float64)
       # OLD layout (nx, ny, ng); NEW layout (ng, nx, ny); transpose-check:
       new_sf_legacy = new_sf.transpose(1, 2, 0)
       np.testing.assert_allclose(
           old_sf, new_sf_legacy, rtol=1e-12, atol=1e-13, equal_nan=True,
       )

Every one of the 11 regression cases passed; the maximum absolute
difference observed was **1.75 × 10\ :sup:`-14`**, which is the
FP-non-associativity ULP scale predicted for layout flips by the
project's ``vv-principles`` skill (the
"bit-identity vs principled-equivalence" boundary).  Eigenvalue
agreement was at machine precision (max ``keff`` delta
6.66 × 10\ :sup:`-16` --- one ULP for ``keff ≈ 1``).

Only after every case passed did the migration proceed to step 2
(snapshot regeneration via
``tests.sn.regression._generate_snapshots``).  This sequence ---
**verify first, then regenerate** --- is the canonical pattern for
any future layout flip and is enshrined in the migration plan's
risk register.


.. _sn-index-convention-numerical-evidence:

Numerical evidence
==================

The migration's correctness rests on the following gates, each run
post-merge of the relevant PR.  All numbers below are verbatim from
the PR closeout memos
(``.claude/agent-memory/method-implementer/issue_196_pr_index_*_closeout.md``).

Regression snapshots (rtol=1e-12)
---------------------------------

The 11 ``tests/sn/regression/`` snapshots cover:

- Slab 2-group homogeneous DD (``slab_2g_homogeneous_dd_n20``).
- Slab 2-group 3-region DD (``slab_2g_3reg_dd_n40``).
- Sphere 2-group homogeneous DD (``sphere_2g_homogeneous_dd_n20``).
- Sphere 2-group 3-region DD (``sphere_2g_3reg_dd_n40``).
- Cylinder 1-group homogeneous LS\ :sub:`4` DD (``cyl_1g_homogeneous_LS4_dd_n20``).
- Cylinder 1-group homogeneous product-quadrature DD (``cyl_1g_homogeneous_product_dd_n20``).
- Cylinder 2-group 3-region LS\ :sub:`4` DD (``cyl_2g_3reg_LS4_dd_n40``).
- Slab 2-group P\ :sub:`1` anisotropic DD (``slab_2g_p1_aniso_dd_n20``).
- Sphere 2-group P\ :sub:`1` anisotropic DD (``sphere_2g_p1_aniso_dd_n20``).
- 2-D Cartesian 1-group LS\ :sub:`4` DD 15×15 (``2d_1g_LS4_dd_15x15``).
- Slab fixed-source DD (``slab_fixed_source_dd_n20``).

Step-1 transpose-check residuals across all 11 cases (PR-INDEX-5):

.. list-table:: Bit-identity-via-transpose residuals
   :header-rows: 1
   :widths: 56 22 22

   * - Case
     - max ``rtol``
     - ``keff`` delta
   * - ``slab_2g_homogeneous_dd_n20``
     - ≤ 1e-12
     - 0.00 × 10\ :sup:`+00`
   * - ``slab_2g_3reg_dd_n40``
     - ≤ 1e-12
     - 4.44 × 10\ :sup:`-16`
   * - ``sphere_2g_homogeneous_dd_n20``
     - ≤ 1e-12
     - 0.00 × 10\ :sup:`+00`
   * - ``sphere_2g_3reg_dd_n40``
     - ≤ 1e-12
     - 2.22 × 10\ :sup:`-16`
   * - ``cyl_1g_homogeneous_LS4_dd_n20``
     - ≤ 1e-12
     - 4.44 × 10\ :sup:`-16`
   * - ``cyl_1g_homogeneous_product_dd_n20``
     - ≤ 1e-12
     - 2.22 × 10\ :sup:`-16`
   * - ``cyl_2g_3reg_LS4_dd_n40``
     - ≤ 1e-12
     - 4.44 × 10\ :sup:`-16`
   * - ``slab_2g_p1_aniso_dd_n20``
     - NaN-bit-identity
     - 0.00 × 10\ :sup:`+00`
   * - ``sphere_2g_p1_aniso_dd_n20``
     - NaN-bit-identity
     - 0.00 × 10\ :sup:`+00`
   * - ``2d_1g_LS4_dd_15x15``
     - ≤ 1e-12
     - 6.66 × 10\ :sup:`-16`
   * - ``slab_fixed_source_dd_n20`` (no ``keff``)
     - ≤ 1e-12
     - n/a

Max absolute difference across the 11 cases:
**1.75 × 10\ :sup:`-14`** --- FP-non-associativity ULP scale.

Post-regeneration, the regression suite passes 11/11 at
``rtol=1e-12`` against the new snapshots (``248.63 s`` wall-clock for
the full regression run on the development host).

L0 streaming-equilibrium curvilinear
------------------------------------

The L0 curvilinear gate at
``tests/sn/sweep/curvilinear/test_streaming_equilibrium_curvilinear.py``
asserts the streaming-equilibrium identity
:math:`\phi = q / \Sigma_t` to machine precision under refinement.
It is the strongest L0 test for the sphere and cylinder sweep, and
the canonical detector for the historical curvilinear bugs
(ERR-004, ERR-025, ERR-026, weight-normalisation slips).

Pre-migration (``c21c2ef`` baseline): 26 passed in 1044 s.
Post-PR-INDEX-4 (``fa41767``): 26 passed in 1044 s.
Post-PR-INDEX-5 (``3356cec``): the Step-1 bit-identity gate on the
six curvilinear regression snapshots (4 sphere + 2 cylinder) is the
strong proxy that the curvilinear sweep math is unchanged --- those
snapshots exercise the same per-cell algebra at the same granularity.

Performance benchmark
---------------------

Slab sweep benchmark on the ``slab_2g_3reg_dd_n40`` configuration
(N=16, ng=2, nx=160):

.. list-table:: Wall-clock per sweep
   :header-rows: 1
   :widths: 36 32 32

   * - Step
     - Mean / sweep
     - Variance (max/min)
   * - Pre-migration baseline (``c21c2ef``)
     - ~0.21 ms
     - 2×
   * - PR-INDEX-1 (slab joint-batch)
     - ~0.21 ms (variance ↓)
     - --
   * - PR-INDEX-2 (cache layout flip)
     - **0.149 ms**
     - **1.06×**
   * - PR-INDEX-5 (public API flip)
     - 0.149 ms
     - 1.06×

The ~28 % mean speedup at PR-INDEX-1 came from the joint-batch
reduction (2 ``ordinate_scan`` calls per sweep replacing N/2).  The
variance tightening at PR-INDEX-2 (from 2× to 1.06×) came from the
cache layout flip eliminating per-cell strided reads.  PR-INDEX-3,
PR-INDEX-4, and PR-INDEX-5 are layout view changes (zero-copy
transposes) --- no measurable wall-clock change.

2-D wavefront equivalence
-------------------------

The 2-D Cartesian octant-equivalence suite at
``tests/sn/test_2d_octant_sweep_equivalence.py`` exercises six
bit-identity cases plus one closed-form L1 anchor.  Post-PR-INDEX-5,
all 7 pass.  The six bit-identity cases agree at ``nulp=64``
(~1.4 × 10\ :sup:`-14`), which is principled-equivalence per
``vv-principles`` --- the principled layout produces the same value
as the legacy layout up to FP-non-associativity at the ULP regime.


.. _sn-field-vocabulary:

SN Field Vocabulary
===================

Before the per-array shape table, this section catalogues the
*conceptual* field hierarchy a future maintainer (or LLM session)
will see in the SN codebase.  Each entry pairs the
mathematical role (what does this quantity *mean*?) with the
physical units, the storage shape under the principled layout, and
the existing-code counterpart with file:line pointers where helpful.
The vocabulary descends from the typed-field contract memo at
``.claude/agent-memory/explorer/typed_field_contracts_for_phase_g.md``
(now corrected for the principled layout); the table below is the
durable, build-versioned reference.

The hierarchy is grouped by epistemic role, *not* by storage shape:
two arrays may share the same numpy shape but play different
roles in the operator algebra (``ScalarFlux`` vs ``ScalarSourceSink``,
for instance).  The role distinction is what makes the type-system
discipline of Issue #197 productive — a bare ``np.ndarray`` cannot
distinguish them.

Field hierarchy --- phase-space and reduced flux types
-------------------------------------------------------

These are the structural-flux types: directional, scalar, and
reductions.  The principled layout places ordinate-index :math:`N`
first (joint-batch over the Krylov axis), then energy :math:`n_g`
(joint-batch over the block-diagonal group axis), then the cell
axes :math:`(n_x, n_y)`.

.. list-table:: Field types
   :header-rows: 1
   :widths: 18 28 14 18 22

   * - Type
     - Physical meaning
     - Units
     - Shape
     - Existing counterpart
   * - :class:`AngularFlux`
     - :math:`\psi(r, \Omega, g)` --- phase-space directional flux
     - 1/(cm²·s·sr)
     - ``(N, ng, nx, ny)``
     - :class:`~orpheus.sn.solution.Solution`.\ ``angular_flux``
       (Issue #197 PR-TYPED-5)
   * - :class:`ScalarFlux`
     - :math:`\phi(r, g) = \int_{4\pi}\psi\,d\Omega` --- angle-integrated
     - 1/(cm²·s)
     - ``(ng, nx, ny)``
     - :class:`~orpheus.sn.solution.Solution`.\ ``scalar_flux``
       (Issue #197 PR-TYPED-5)
   * - ``GroupFlux``
     - Slice of :class:`AngularFlux` / :class:`ScalarFlux` at one :math:`g`
     - same as parent
     - ``(N, nx, ny)`` or ``(nx, ny)``
     - Slice expression; not a separate type
   * - :class:`HarmonicMomentFlux`
     - :math:`\phi_{\ell m}(r, g)` --- Pℓ moment coefficients
     - 1/(cm²·s·sr·eV) [inherits from source field]
     - ``(<angular head>, ng, nx, ny)`` — ``(L+1, 2L+1)`` on a rule
       that binds the spherical harmonics, ``(L+1,)`` on a 1-D rule
       (see the note below this table)
     - ``frame.analysis.apply`` output (the SH
       :class:`~orpheus.numerics.frame.GalerkinFrame`'s analysis face;
       :mod:`orpheus.transport.operators.scattering`); typed wrapper at
       :class:`orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
       (Issue #197 PR-TYPED-4)
   * - ``BoundaryField``
     - :math:`\psi` restricted to :math:`\Gamma_-` or :math:`\Gamma_+`
     - 1/(cm²·s·sr)
     - ``(N_inflow, ng)`` per face
     - :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
       face views, keyed by face name (``face_view("xmin")`` …) —
       the pre-#197 ``psi_bc`` dict (the boundary arg of the since-retired
       ``transport_sweep``) is retired

The conversion functions between the two principal types are the
load-bearing primitives of the operator algebra: ``to_scalar()`` is
the :math:`\sum_n w_n \psi_n` reduction (one ``np.einsum`` over the
leading ordinate axis), and ``broadcast_to_ordinates()`` is the
isotropic embedding :math:`\psi_n(r, g) = \phi(r, g)\ \forall\,n`
(a ``np.broadcast_to`` prepending the :math:`N` axis).  Both run in
:math:`O(N \cdot n_g \cdot n_x \cdot n_y)` and live in the
``orpheus/sn/typed_fields.py`` module (planned — the structural
type lands as the typed-field-contract resume).

Source / RHS vocabulary
-----------------------

The six-operator algebra — :math:`A\,\psi = q` for a fixed source, with
the loss composite :math:`A = L + C - S - N_{2n} - B` — has a
typed RHS.  The "source" :math:`q` is a deliberate split into
direction-independent (``ScalarSourceSink``) and per-ordinate
(``AngularSourceSink``) contributions — the within-group sweep (the
resolvent ``solve``) consumes both, and the
internal P₀ accumulation in
:class:`~orpheus.transport.operators.scattering.ScatteringOperator` emits the first
while the P\ :sub:`ℓ≥1` accumulation emits the second.  The
:math:`(n,2n)` gain has the SAME two halves and has had since #426
step 2 (2026-09-04): its :math:`P_0` emission is scalar-driven and its
:math:`P_{\ell\ge1}` emission is per-ordinate, emitted by its own
:class:`~orpheus.transport.operators.n2n.N2NOperator` — a role of the
same :class:`~orpheus.transport.operators.transfer.TransferOperator`
core as :math:`S` (:ref:`n2n-reactions`).  Both gains reach the
per-ordinate carrier through the ONE shared ``(iso / W) + aniso``
combine, which is now single-sourced on that core rather than once per
channel.

⛔ This paragraph read *"both isotropic producers"* until 2026-09-04,
which was true of the tree and not of the physics: the :math:`(n,2n)`
producer was isotropic because the operator tier truncated it to
:math:`P_0`, not because the reaction is (ERR-082;
:ref:`the truncation record <sn-n2n-p0-truncation>`).  The
:math:`\ell = 0` half of each gain is still scalar-driven, and that is
what the shared combine's ``iso`` slot carries.

.. list-table:: Source / RHS field types
   :header-rows: 1
   :widths: 22 28 14 18 18

   * - Type
     - Physical meaning
     - Units
     - Shape
     - Existing counterpart
   * - ``ScalarSourceSink``
     - :math:`q(r, g)` --- direction-independent external source
     - 1/(cm³·s·sr)
     - ``(ng, nx, ny)``
     - ``Q`` arg of the retired ``transport_sweep``;
       ``Q_iso`` in
       :meth:`~orpheus.transport.operators.scattering.ScatteringOperator.apply`
   * - ``AngularSourceSink``
     - :math:`q_n(r, g)` --- per-ordinate (P\ :sub:`ℓ≥1` /
       boundary) source
     - 1/(cm³·s·sr)
     - ``(N, ng, nx, ny)``
     - ``Q_aniso`` arg of the retired ``transport_sweep``; the interior
       leaf of :meth:`ScatteringOperator.apply
       <orpheus.transport.operators.transfer.TransferOperator.apply>`
       (the combined :math:`(\text{iso}/W) + \text{aniso}` emission)
   * - ``ResidualSource``
     - :math:`r = q - A\psi_D` for hybrid corrections (Grand
       Report v3 §25.1)
     - 1/(cm³·s·sr)
     - matches ``q``
     - None — not used in Phase G
   * - ``AngularBoundarySourceSink``
     - Prescribed inflow at :math:`\Gamma_-` (Grand Report v3
       §16A.2)
     - 1/(cm²·s·sr)
     - ``(N_inflow,)`` per face
     - Implicit in BC-applied face buffers consumed at sweep
       entry

The shape distinction between the two source flavours
(``(ng, nx, ny)`` vs ``(N, ng, nx, ny)``) is load-bearing: the
within-group sweep avoids a wasteful
:math:`N`-fold broadcast of the isotropic part by accepting both
splits.

Rates and tallies
-----------------

These are derived scalar / vector observables produced by the
solver as named intermediates (Pattern 3 — "named intermediates"
from the ``coding-elegance`` skill).  Their values fall out of the
operator algebra once a flux is solved; the principled storage is
the natural numpy reduction shape.

.. list-table:: Rate and tally fields
   :header-rows: 1
   :widths: 20 28 14 18 20

   * - Type
     - Physical meaning
     - Units
     - Shape
     - Existing counterpart
   * - ``ReactionRate``
     - :math:`\sigma \cdot \phi` --- per-cell, per-group rate density
     - 1/(cm³·s)
     - ``(ng, nx, ny)``
     - Computed inline in
       :meth:`~orpheus.sn.solver.SNSolver.compute_group_production_rate`
       / ``..._absorption_rate``
   * - ``GroupRate``
     - :math:`\int_V \sigma \cdot \phi\,dV` --- volume-integrated
       per group
     - 1/s
     - ``(ng,)``
     - Return of
       :meth:`~orpheus.sn.solver.SNSolver.compute_group_production_rate`
   * - ``CurrentCochain``
     - Face-summed currents (Grand Report v3 §15A.10)
     - n/(cm²·s)
     - ``(N_faces, ng)``
     - None — future
   * - ``Functional``
     - Map field → scalar response
     - varies
     - scalar
     - :meth:`~orpheus.sn.solver.SNSolver.compute_keff` is a degenerate case

Iteration state
---------------

These are the per-outer / per-inner diagnostic carriers that a
solution wraps.  They live on the
:class:`~orpheus.sn.solution.IterationHistory` dataclass (holding the
per-outer / per-inner trajectory with named fields and method-style
accessors such as :attr:`~orpheus.sn.solution.IterationHistory.dominance_ratio`
/ :attr:`~orpheus.sn.solution.IterationHistory.converged`), which a
:class:`~orpheus.sn.solution.Solution` carries as an optional field
(populated for eigenvalue problems).  This pair (Issue #197 PR-TYPED-5)
replaced the legacy bare-dataclass ``SNResult`` /
``SNFixedSourceResult`` data bags.

.. list-table:: Iteration state
   :header-rows: 1
   :widths: 22 30 18 30

   * - Field
     - Physical meaning
     - Shape
     - Existing counterpart
   * - ``keff``
     - Multiplication eigenvalue
     - scalar
     - :class:`~orpheus.sn.solution.Solution`.\ ``keff: float | None``
       (``None`` for fixed-source problems — Issue #197 PR-TYPED-5)
   * - ``keff_history``
     - Outer iteration trajectory
     - ``tuple[float, ...]``
     - :class:`~orpheus.sn.solution.IterationHistory`.\ ``keff_history``
       (exposed as ``list[float]`` via
       :meth:`SolutionBase.keff_history_list`)
   * - ``Eigenpair``
     - ``(value, right, left, residual_norm)`` tuple
     - varies
     - Implicit — :class:`~orpheus.sn.solution.Solution` is a
       degenerate Eigenpair when :meth:`is_eigenvalue` returns ``True``
   * - ``ResidualHistory``
     - Per-iter relative flux residual
     - ``tuple[float, ...]``
     - :class:`~orpheus.sn.solution.IterationHistory`.\ ``flux_residuals``
   * - ``DominanceRatio``
     - :math:`|k_n - k_{n-1}| / |k_{n-1}|` convergence quotient
     - scalar
     - :meth:`~orpheus.sn.solution.SolutionBase.dominance_ratio`

Solution-class container
------------------------

Issue #197 PR-TYPED-5 lands the typed solution container.  It RETIRED
the legacy bare-array ``SNResult`` / ``SNFixedSourceResult`` data bags
into one typed carrier covering both problem kinds.  Campaign #276 A5
(2026-07-25) then split the carrier along the **role axis**:
:class:`~orpheus.sn.solution.SolutionBase` (the role-agnostic,
non-instantiable carrier) → :class:`~orpheus.sn.solution.Solution`
(forward; :func:`~orpheus.sn.solver.solve_sn` /
``solve_sn_fixed_source``) and
:class:`~orpheus.sn.solution.AdjointSolution` (adjoint;
``solve_sn_adjoint`` / ``solve_sn_adjoint_fixed_source``, whose
``scalar_flux`` is the importance :math:`\varphi^*` — alias
``importance``).  The two discrimination axes use DIFFERENT
mechanisms: the problem KIND (fixed-source vs eigenvalue) is a
property (optional ``keff``), the solution ROLE (forward vs adjoint)
is the type.

The :class:`~orpheus.sn.solution.SolutionBase` carrier holds:

- :class:`~orpheus.transport.fields.angular_flux.AngularFlux` +
  :class:`~orpheus.transport.fields.scalar_flux.ScalarFlux` +
  :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux` typed fields (NOT
  bare ndarrays);
- :class:`~orpheus.sn.solution.IterationHistory` carrying tuple-based
  per-outer / per-inner trajectory diagnostics (NOT list-based);
- ``mesh`` reference shared (by identity) with every typed flux field
  — validated at construction (``coding-elegance`` Pattern 4 —
  illegal states unrepresentable);
- ``keff: float | None`` — ``None`` for fixed-source problems;
  :meth:`SolutionBase.is_eigenvalue` and
  :meth:`SolutionBase.is_fixed_source` are the canonical
  discriminators;
- :meth:`SolutionBase.dominance_ratio` /
  :meth:`SolutionBase.converged` — iteration diagnostics that read as
  math (``coding-elegance`` Pattern 1);
- :meth:`SolutionBase.compare` — field-by-field difference summary
  that returns :class:`~orpheus.sn.solution.SolutionDiff`.
  **Role-closed** (``Self``-typed + runtime guard): comparing a
  forward flux against an importance map is a type error, not a
  number.

The FORWARD leaf alone carries the reaction-rate-preserving physics —
:meth:`Solution.homogenize`, :meth:`Solution.condense`, and
:meth:`Solution.reaction_rate_density` (:math:`\sigma\cdot\phi`
per-cell rate density as a named elementwise product, Pattern 3).
These are structurally ABSENT on ``AdjointSolution``: an importance
map has no reaction rate to preserve; the adjoint enters
homogenization/condensation only as the optional Petrov–Galerkin test
weight of the forward collapse (the #281 P6-B2 parameter).

The Solution evolution is the SN-specific specialisation of the
``Eigenpair`` concept from Grand Report v3 §21.5 (lines 4252–4269).
For a fixed-source problem ``keff`` is ``None`` and the iteration
history records only the relative flux-residual trajectory.

Operator vocabulary --- the six leaves of the algebra
------------------------------------------------------

The six-operator algebra — the within-group loss composite
:math:`A = L + C - S - N_{2n} - B`, posed as
:math:`A\,\psi = \tfrac{1}{k}\,F\,\psi` (eigenvalue) or
:math:`A\,\psi = q` (fixed source) — is
implemented by six leaf operators, each conforming to
:class:`~orpheus.numerics.operator.LinearOperator` with
``apply(psi: AngularFlux) -> AngularFlux`` under the typed contract.
The algebra closes because every operand agrees on the
:class:`AngularFlux` domain — :class:`OperatorSum` distributes
:math:`a.\text{apply}(\psi) + b.\text{apply}(\psi)` element-wise,
and :class:`ScaledOperator` carries the :math:`1/k` (or :math:`-1`)
scale through the same algebra.

.. list-table:: Operator leaves
   :header-rows: 1
   :widths: 14 38 48

   * - Leaf
     - Code class
     - Mathematical role
   * - :math:`L`
     - :class:`~orpheus.sn.operators.streaming.StreamingOperator`
     - Streaming :math:`\Omega \cdot \nabla\psi`; per-ordinate
       sweep over :func:`~orpheus.sn.mesh.augmented_mesh.SNMesh.dag_walk`,
       fold over :meth:`DiscretizationScheme.residual`
   * - :math:`C`
     - :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
       (the collision multiplier :math:`C = M[\Sigma_t]`)
     - Collision :math:`\Sigma_t \psi`; one broadcast multiply
       ``sigma[None, :, :, :] * psi.values``
   * - :math:`S`
     - :class:`~orpheus.transport.operators.scattering.ScatteringOperator`
     - Full Legendre scattering :math:`\sum_\ell \Sigma_{s,\ell}\,
       P_\ell\,\phi_{\ell m}`; foldable P₀ within-group part
       :meth:`~orpheus.transport.operators.scattering.ScatteringOperator.foldable_part`
       absorbs into :math:`\Sigma_r`
   * - :math:`N_{2n}`
     - :class:`~orpheus.transport.operators.n2n.N2NOperator`
     - The :math:`(n,2n)` emission :math:`N_{2n} = R\,\Lambda_{2n}\,M / W`
       — the **same** binding as :math:`S` in a different role, over the
       mixture's ``Sig2`` Legendre stack and with the yield
       :math:`\nu_{2n} = 2` inside :math:`\Lambda`.  First-class since
       CS4c step 3 because the channel's bundling is context-dependent
       (with :math:`S` for anisotropy, with :math:`F` for production
       accounting) and must not be decided at the operator level; see
       :ref:`the two collision gains <operator-algebra-two-gains>`
   * - :math:`B`
     - :class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`
     - The realized boundary law — a first-class **sibling** operator, *not*
       folded into :math:`L`.  Reports the same
       :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace` as
       :math:`L`/:math:`C`/:math:`S`/:math:`N_{2n}`/:math:`F`, so the
       ``OperatorSum``
       composition guard accepts :math:`(L + C - S - N_{2n} - B - F/k)`.
       Acts as the
       :math:`A_{ss}` block: zero on the bulk, non-zero only on the trace,
       where the cosine-weighted :math:`|\Omega\cdot n|\,w` partial-current
       metric lives
   * - :math:`F`
     - :class:`~orpheus.transport.operators.fission.FissionOperator`
       (angular binding) /
       :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`
       (energy binding)
     - Fission :math:`\chi_g \sum_{g'} \nu\Sigma_{f,g'}\,\phi_{g'}`;
       rank-1 in energy, rank-0 in angle.  The energy binding IS that
       dyad on the scalar flux; the angular binding wraps it in
       ``integrate_angular`` + the shared per-ordinate isotropic
       combine (CS4c step 4)

Two derived combinations carry their own names:

- The fusion target :math:`A_{wg} = L + C - S_{\text{foldable}}`
  is the within-group system; its ``solve`` routes through the
  fused sweep (the
  ``L + C`` composition rule mints the sweep-invertible
  :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator`, whose
  own ``solve`` override runs the sweep on its selected representation
  rather than the unfused Krylov outer-iteration — the generic
  ``OperatorSum`` carries no ``solve`` at all since carve P4).
  :meth:`foldable_part
  <orpheus.transport.operators.transfer.TransferOperator.foldable_part>`
  is defined on the **shared** transfer core, so the :math:`(n,2n)` gain
  has one too and enters this target the same way; the single symbol is
  written here because the fold's algebra does not distinguish the two
  gains (:ref:`the two collision gains <operator-algebra-two-gains>`).
- The multiplication operator :math:`K = A_{\text{loss}}^{-1} F`
  carries the k-eigenvalue iteration; it lives implicitly in the outer
  :func:`~orpheus.numerics.eigenvalue.power_iteration` loop, which
  repeatedly calls
  :meth:`~orpheus.sn.solver.SNSolver.solve_fixed_source`.

Boundary-trace vocabulary
-------------------------

Boundary handling has its own type vocabulary
(:mod:`orpheus.geometry.boundary`).  These types are
**orthogonal** to the index convention — they live on
:math:`\Gamma_\pm` faces, not in the volume:

.. list-table:: Boundary-trace types
   :header-rows: 1
   :widths: 28 34 38

   * - Type
     - Code counterpart
     - Role
   * - ``BoundaryTraceLaw``
     - :mod:`orpheus.geometry.boundary` per-BC modules
     - The discrete BC's algebraic action — the affine map
       :math:`\gamma_-\psi = R\,G\,\gamma_+\psi + q`, i.e.
       **outflow trace in, inflow trace out**
   * - ``BoundaryRealizer``
     - :class:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer`
     - SN-side discretisation of the trace law. Since campaign phase
       B3.2 a realized SN law is typed :math:`\Gamma_+ \to \Gamma_-`
       (see :ref:`bc-domain-narrowing`)
   * - ``AngularTraceSpace``
     - :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`
     - **One** space carrying the whole boundary, with inflow /
       outflow as directional *selectors* over it (the per-direction
       ``InflowTraceSpace`` / ``OutflowTraceSpace`` pair was collapsed
       at #205 / #201). The ordinate set at a face is a **three-way**
       partition — inflow ⊔ outflow ⊔ tangential
   * - ``VacuumInflow`` /
       ``ReflectiveBoundary``
     - :mod:`orpheus.geometry.boundary.vacuum` /
       :mod:`orpheus.geometry.boundary.reflective`
     - Physical BC kinds
   * - ``PermutationOperator`` /
       ``TraceRestrictionOperator``
     - :mod:`orpheus.numerics.operator`
     - The discrete-ordinate face-permutation algebra, and the
       gather / scatter pair :math:`(\gamma_S, \iota_S)` that types
       the trace maps. Siblings, never subclasses: a permutation is a
       bijection, a restriction is rank-deficient

Diagnostic and historical state
-------------------------------

Aspirational from Grand Report v3 §32 / §39; not yet implemented
in SN.  Listed here for vocabulary completeness — a future
``Solution`` extension may carry them as named fields.

.. list-table:: Diagnostic types (aspirational)
   :header-rows: 1
   :widths: 28 38 34

   * - Type
     - Meaning
     - Existing counterpart
   * - ``Axis(name, size, coordinate, measure)``
     - Labelled-axis primitive
     - None — bare numpy shapes
   * - ``AxisProduct``
     - Tuple of ``Axis``
     - None
   * - ``DomainMismatchError`` /
       ``CodomainMismatchError``
     - Operator-algebra type errors
     - :class:`~orpheus.numerics.operator.IncompatibleOperatorComposition`
   * - ``ConservationDefect`` /
       ``PositivityViolation``
     - V&V residuals
     - Distributed across :mod:`tests._harness`


Layout-by-array reference table
===============================

Every array a future maintainer encounters in the SN codebase
matches one of these shapes.  The reference table consolidates what
lives in scattered docstrings.

.. list-table:: SN array shapes (post-PR-INDEX-5)
   :header-rows: 1
   :widths: 32 28 40

   * - Array
     - Shape
     - Defined at
   * - :class:`SNSolver`.\ ``sig_t``, ``sig_a``, ``sig_p``, ``chi``
     - ``(ng, nx, ny)``
     - :meth:`SNSolver.__init__`
   * - :class:`~orpheus.sn.solution.Solution`.\ ``scalar_flux`` values
     - ``(ng, nx, ny)``
     - :class:`~orpheus.transport.fields.scalar_flux.ScalarFlux`
   * - :class:`~orpheus.sn.solution.Solution`.\ ``angular_flux.interior``
     - ``(N, ng, nx, ny)``
     - :class:`~orpheus.transport.fields.angular_flux.AngularFlux`
   * - SN sweep input ``Q`` (isotropic source)
     - ``(ng, nx, ny)``
     - :meth:`~orpheus.sn.loss_representation.CumprodScan.sweep`
   * - SN sweep input ``sig_t``
     - ``(ng, nx, ny)``
     - :meth:`~orpheus.sn.loss_representation.CumprodScan.sweep`
   * - SN sweep per-ordinate source (folded into ``Q``)
     - ``(N, ng, nx, ny)``
     - :meth:`~orpheus.sn.loss_representation.CumprodScan.sweep`
   * - SN sweep return ``angular_flux``
     - ``(N, ng, nx, ny)``
     - :meth:`~orpheus.sn.loss_representation.CumprodScan.sweep`
   * - SN sweep return ``scalar_flux``
     - ``(ng, nx, ny)``
     - :meth:`~orpheus.sn.loss_representation.CumprodScan.sweep`
   * - :func:`solve_sn_fixed_source` input ``external_source``
     - ``(N, ng, nx, ny)``
     - :func:`~orpheus.sn.solver.solve_sn_fixed_source`
   * - :class:`CollisionCache` fields (``a_attenuation``,
       ``inverse_denom``, …)
     - ``(N, ng, nx)``
     - :class:`~orpheus.sn.sweep.cache.CollisionCache`
       (1-D; collapses ``ny=1``)
   * - :class:`ScatteringOperator`.\ ``apply`` in/out
     - ``(N, ng, nx, ny)``
     - :meth:`~orpheus.transport.operators.scattering.ScatteringOperator.apply`
   * - :class:`IsotropicFission`.\ ``apply`` in/out
       (the fission ENERGY binding)
     - ``(ng, nx, ny)``
     - :meth:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission.apply`
   * - :class:`FissionOperator`.\ ``apply`` in/out
       (the fission ANGULAR binding; a scalar carrier is REFUSED)
     - ``(N, ng, nx, ny)``
     - :meth:`~orpheus.transport.operators.fission.FissionOperator.apply`
   * - :class:`StreamingOperator`.\ ``apply`` in/out (Resolution A)
     - ``(N, ng, nx, ny)``
     - :class:`~orpheus.sn.operators.streaming.StreamingOperator`
   * - :class:`MultiplicationOperator`.\ ``apply`` in/out (Resolution A;
       the collision multiplier :math:`C = M[\Sigma_t]`)
     - ``(N, ng, nx, ny)``
     - :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
   * - :class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`
       moment field (either channel)
     - ``(<angular head>, ng, nx, ny)`` — see the note below
     - :mod:`orpheus.transport.operators.transfer`

.. note::

   ⚠ **The** ``(L+1, 2L+1, …)`` **head is the REAL-HARMONIC family's,
   not the moment carrier's contract** (2026-09-02, #429 / ERR-080).
   A moment field's space is ``<angular head> ⊗ cells``, and since the
   ERR-080 repair a **1-D** rule binds a FLAT head — `[M]` a
   ``gauss_legendre(8)`` phase space gives
   ``HarmonicMomentFlux.zeros_for_mesh_and_L(sn, 2).values.shape ==
   (3, 1, 4)``, i.e. ``(L+1, ng, *spatial)``, while
   ``level_symmetric(8)`` and ``folded_product(4,8)`` give
   ``(3, 5, 1, 4)``. Read every ``(L+1, 2L+1, …)`` shape on this page as
   *the shape on a rule that binds the spherical harmonics*. The layout
   is now the head's to say —
   :class:`~orpheus.numerics.spaces.moment_head.MomentHead`'s
   ``isotropic_slot`` / ``degree_block`` / ``shape`` — and a consumer
   READS it (:ref:`spaces-moment-head`).

.. note:: **Two rows above were re-homed, one is still stale
   (audit 2026-08-09).** The flux rows used to read
   ``SNSolver.scalar_flux`` / ``SNSolver.angular_flux`` with the
   definition site given as ``SNResult`` — a class RETIRED at Issue
   #197 PR-TYPED-5 (see the PR-INDEX-5 changelog row above). The
   *shapes* were and remain correct; only the carrier moved, to the
   typed fields now shown. The first row, ``SNSolver.sig_t, sig_a,
   sig_p, chi`` at ``(ng, nx, ny)`` defined by ``SNSolver.__init__``,
   is **NOT yet reconciled**: measured 2026-08-09,
   :meth:`SNSolver.__init__` takes ``(sn_mesh, inner_solver,
   scattering_order, keff_tol, flux_tol, max_inner, inner_tol,
   inner_schedule)`` and the solver carries no ``sig_*`` / ``chi``
   attribute at all; the per-cell cross sections live on
   :class:`~orpheus.data.macro_xs.cell_xs.CellXS`, whose arrays are
   **cell-major** ``(nc, ng)`` — a different priority order from the
   row's claim. Reconciling that row needs a trace of how the SN path
   reshapes ``CellXS`` into the sweep's layout, which this audit did
   not do; the row is preserved unmodified rather than guessed at.

.. note:: A former row here recorded the FD-matvec packed-vector
   internal ``fi`` shape ``(ng, N, nx, ny)`` (``ng`` first, then ``N``,
   an out-of-priority-order layout deferred to the never-executed
   PR-INDEX-7).  That whole path — the ``transport_operator_matvec_*``
   family and its ``solution_to_angular_flux*`` unpackers — was deleted
   in the typed-field campaign (#197), so the row no longer describes
   any live array.

One array does **not** follow the priority order:

- :class:`CollisionCache` fields drop ``ny`` (they are 1-D-only and
  consume the cell axis as a single innermost contraction).

Three array shapes that look like exceptions but are not:

- ``ordinate_scan`` consumes ``(nx, K, ng)`` --- the scan axis
  (cell, here ``nx``) leads.  This is a **primitive contract**
  required by Blelloch's parallel-prefix algorithm (the scan axis
  *must* be the outermost iteration).  The principled-storage
  ``(ng, nx)`` slice is transposed to ``(nx, ng)`` at the call
  site (one ``.T`` per ordinate-chain).
- 1-D internal arrays drop the trailing ``ny`` (slice
  ``[:, :, 0]`` at the boundary).  This is layout-consistent: the
  ``ny=1`` singleton is preserved at the public-API surface, but
  internal 1-D primitives work on ``(ng, nx)`` for clarity.
- The 1-D scratch buffer ``Q_p = Q[:, :, 0]`` in
  :meth:`._OneDimScanWalk.sweep` is a zero-copy view of the public
  ``(ng, nx, ny)`` source.  No layout decision is made here ---
  it's a slice of the principled storage.


.. _theory-sn-typed-fields:

Typed field types
=================

The principled-layout migration (Issue #196 PR-INDEX-1..7) flipped the
bare ndarray storage to ``(N, ng, nx, ny)``.  Issue #197 PR-TYPED-2
wraps those arrays in three typed dataclasses so the field semantics
read as the math (``coding-elegance`` Pattern 1) and shape mismatches
fail at construction time (Pattern 4 — illegal states unrepresentable).

.. todo:: Archivist expansion needed.

   The full rich-narrative version of this section — derivation of each
   type's domain semantics, units, the dunder algebra and its
   correspondence to the operator-equation form
   ``(L + C − S − N_2n − F/k) ψ = q``, and a worked walk-through of the
   within-group sweep's
   AngularBoundaryFlux contract —
   should be authored by the **archivist** sub-agent.  This section
   is the stub written by the method-implementer per
   ``algebra-of-record``'s Sphinx stub vs rich narrative discipline.

   Source modules: :mod:`orpheus.transport.fields.angular_flux`,
   :mod:`orpheus.transport.fields.scalar_flux`, :mod:`orpheus.transport.fields.angular_boundary_flux`.
   Foundation tests:
   :file:`tests/sn/test_typed_fields.py` (22 cases, all green).
   Closeout memo:
   :file:`.claude/agent-memory/method-implementer/issue_197_pr_typed_2_closeout.md`.

The three types
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 28 52

   * - Type
     - Storage shape
     - Reads as the math
   * - :class:`~orpheus.transport.fields.angular_flux.AngularFlux`
     - ``(N, ng, nx, ny)``
     - :math:`\psi(\vec r, \hat\Omega_n, g)`.  ``psi.integrate_angular()``
       returns the :class:`ScalarFlux` (the canonical ``Σ_n w_n ψ_n``
       reduction).
   * - :class:`~orpheus.transport.fields.scalar_flux.ScalarFlux`
     - ``(ng, nx, ny)``
     - :math:`\phi_g(\vec r) = \int_{4\pi} \psi\,d\Omega`.  Dunder
       arithmetic: ``a + b``, ``α · phi``, ``phi.at_group(g)``.
   * - :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
     - Per-face: 1-D ``(N, ng)``; 2-D persistent ``(N, ng, nx+1, ny)``
       / ``(N, ng, nx, ny+1)``.
     - Boundary :math:`\psi` at every face plus curvilinear pole state.
       Replaces the stringly-typed ``psi_bc: dict``.

Zero-field allocation is SPACE-keyed: the carrier mints, the leaf allocates
---------------------------------------------------------------------------

.. note:: **Correction (2026-08-10, Issue #346).**  This section was
   headed *"Factory methods (SNMesh)"* and stated that
   :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` carries
   ``zeros_angular_flux`` / ``zeros_scalar_flux`` /
   ``zeros_boundary_flux``.  It does not, and the capability was not
   renamed — it **changed owner**.  ``[M]``
   ``[n for n in dir(SNMesh) if "zero" in n.lower()] == []``.

.. note:: **Second correction (2026-08-24, campaign 1 CS4b S5).**  The
   ownership has moved once more, and this section carried the middle
   step of a two-step history.  #346 moved the allocator **off the mesh
   and onto the leaf**, where it was spelled ``Leaf.zeros_on(mesh)``
   (with siblings ``from_mesh(values, mesh)`` and ``from_ndarray``).
   CS4b S5 retired that whole *mesh-keyed sugar tier* in turn: a field is
   an ELEMENT of a function space, so its constructor and its allocator
   are keyed by the **space**, not by the carrier that happens to mint
   one.  The leaf is still the owner; only the key changed.  ``[M]``
   ``hasattr(AngularField, "zeros_on") is False`` and likewise for
   ``from_mesh`` / ``from_ndarray`` on every transport field leaf.

A field leaf has exactly **two** construction spellings, and both name a
:class:`~orpheus.numerics.space.FunctionSpace`:

.. code-block:: python

   psi = AngularFlux(values=arr, space=sn.angular_bulk_space)   # primary
   psi = AngularFlux.zeros(sn.angular_bulk_space)               # allocator

The primary constructor is the dataclass itself (``values``/``space`` are
its only two fields on the angular and scalar families), and the
allocator is the single shared
:meth:`Field.zeros <orpheus.numerics.field.Field.zeros>` on the base —
one body, ``np.zeros(space.shape)``, inherited by every leaf.  Nothing
between the caller and that body has to know a storage layout, because
the space already *is* the layout.

The carrier's cached mints
~~~~~~~~~~~~~~~~~~~~~~~~~~

What a call site reads instead of passing a mesh is one of the carrier's
**cached** space properties.  Each is minted once per carrier and shared
by reference, so every field on one carrier holds the *same* space
instance (a cheap identity; the operator-admission guards themselves
compare space CONTENT — the F2 doctrine — so equal twin-carrier mints
mix and only a genuinely different space refuses):

.. list-table:: Which mint keys which family
   :header-rows: 1
   :widths: 34 24 42

   * - Carrier mint
     - Families that live there
     - What it is
   * - :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_bulk_space`
     - :class:`AngularFlux`, :class:`~orpheus.transport.source_sinks.AngularSourceSink`,
       the angular residual
     - the per-ordinate bulk phase space, the ordered axis product
       :math:`V_\Omega \otimes V_E \otimes V_r` with the quadrature
       measure :math:`w_n` on the ordinate axis and the cell volumes on
       the spatial axis
   * - :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_trial_space`
     - the same angular leaves, when the caller wants the **scheme's**
       within-cell basis
     - :attr:`angular_bulk_space` extended by the scheme's MODAL moment
       axis (below).  Byte-identical to it for the slopeless closures
   * - :attr:`~orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space`
     - :class:`ScalarFlux`, :class:`~orpheus.transport.source_sinks.ScalarSourceSink`,
       the scalar residual
     - the scalar bulk :math:`V_E \otimes V_r` — literally
       :attr:`angular_bulk_space` minus axis 0, which is why the angular
       retract cannot disagree with it on the shared factors
   * - :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_trace`
     - :class:`AngularBoundaryFlux`,
       :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink`
     - the FLAT per-face trace buffer under the
       :math:`\lvert\hat\Omega\cdot\hat n\rvert\,w_n` partial-current
       metric
   * - :attr:`~orpheus.diffusion.augmented_mesh.DiffusionMesh.scalar_trace`
     - the scalar :math:`(J^+, J^-)` boundary leaves
     - the diffusion partial-current trace, under the face-AREA metric.
       A bare ``MaterialMesh`` has none — a scalar trace is diffusion
       *behaviour*, not method-agnostic *data* (#290 P7a)
   * - :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.radial_characteristic_interior_space`
       / :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.radial_characteristic_boundary_space`
     - the :math:`\psi_{1/2}` starting-direction leaves (System B)
     - the split ``cells`` and ``corner`` ray spaces, keyed by
       ``(level, sign)``.  Both are ``None`` on a carrier whose
       :math:`\mu`-levels consume no independent starting direction
       (R12a — Cartesian and the production cylinder rules)

The second row's parenthetical — *the scalar bulk is literally the
angular bulk minus axis 0* — is not a coincidence to be maintained by
hand: since CS4b S6 it is realized by an **arrow between the two
mints**. :meth:`FunctionSpace.retraction
<orpheus.numerics.space.FunctionSpace.retraction>` over the
``"angular"`` axis contracts the angular bulk with the quadrature
measure and its bound codomain *is* the scalar mint, content-equal by
construction; its section
(:meth:`~orpheus.numerics.space.FunctionSpace.section`) is the
normalized lift back. Both are memoized on the space, so
:meth:`AngularFlux.integrate_angular
<orpheus.transport.fields.angular_flux.AngularFlux.integrate_angular>`
and
:meth:`AngularSourceSink.from_isotropic
<orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink.from_isotropic>`
are one ``apply`` each rather than two hand-written kernels that could
drift from the mints they connect. ⚠ They are DIFFERENT arrows — the
adjoint of the retraction is the *plain* broadcast, the section divides
by :math:`\Sigma w` first — see :ref:`spaces-collapse-pair` on
:doc:`/theory/foundations/spaces`.

Measured allocations
~~~~~~~~~~~~~~~~~~~~

``[M]`` on a vacuum slab with ``N = 4`` (Gauss–Legendre), ``ng = 2``,
``nx = 4``, diamond difference:

.. list-table::
   :header-rows: 1
   :widths: 46 26 28

   * - Allocation
     - Space read
     - Measured shape
   * - ``AngularFlux.zeros(sn.angular_bulk_space)``
     - :attr:`angular_bulk_space`
     - ``(4, 2, 4)`` — rank-honest ``(N, ng, *spatial_shape)``, **not**
       a phantom-``ny`` ``(N, ng, nx, 1)``
   * - ``ScalarFlux.zeros(sn.bulk_space)``
     - :attr:`bulk_space`
     - ``(2, 4)`` — ``(ng, *spatial_shape)``
   * - ``AngularBoundaryFlux.zeros(sn.angular_trace)``
     - :attr:`angular_trace`
     - ``(16,)`` — the FLAT trace buffer over the faces the geometry
       actually carries (here :math:`N \times n_g \times 2` faces)
   * - ``AngularSourceSink.zeros(sn.angular_trial_space)``
     - :attr:`angular_trial_space`
     - ``(4, 2, 4)`` under DD — the trial mint *is* the bulk mint here

Switch the same slab to linear discontinuous and only the trial mint
moves: ``[M]`` :attr:`angular_bulk_space` still reads ``(4, 2, 4)``
while :attr:`angular_trial_space` reads ``(4, 2, 4, 2)``, its axes being
``angular(4) × energy(2) × spatial(4) × spatial_moment(2)`` with the
moment factor MODAL and carrying the scheme's mass
:math:`(1, \tfrac13)` as its measure.

The trial mint — construct general, select narrow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_trial_space` is
the scheme-widened sibling of :attr:`angular_bulk_space`: the same
product with the bound scheme's within-cell spatial-moment factor
(:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis`,
the tensor-Legendre cell basis) appended,

.. math::

   V_{\rm trial} \;=\;
   V_\Omega \otimes V_E \otimes V_r \otimes V_{\rm moment}({\rm scheme}).

For a slopeless closure
(:attr:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.is_multi_moment`
``False`` — diamond difference and step) the trial basis *is* the cell
average, and the property returns :attr:`angular_bulk_space` **itself**
— the same cached instance, byte-identical, so slopeless consumers pay
nothing and the two mints cannot drift.

This is the whole of the construct-general / select-narrow seam of #240
D5b-S3-A0: **which of the two properties a call site reads IS the
widening decision.**  The seams that FILL the moment axis (the SI cold
starts, the LD emissions, the coupled-system assembly) allocate on the
trial mint; width-1 cell-average consumers read the bulk mint.  It
replaces the retired ``spatial_moments=`` factory parameter, whose
integer was a lossy proxy for the scheme's basis and had to be threaded
identically at every call site; the composition rule *"widen* :math:`\iff`
*append the scheme's moment axis"* is now spelled once, on the carrier.

The metric is what makes this a correctness mint rather than sugar
(#310 C2 ruling 3): the moment axis carries the scheme's mass
:math:`M_{ii}/V = \prod_a \theta^{o_a}` as its measure, so the trial
Gram is :math:`G = V\,w_n \otimes \mathrm{diag}(1, \theta, \dots)`.
Broadcasting an average-only :math:`V\,w_n` across the moment axis would
mis-weight the slope degrees of freedom: ``.H`` becomes a WRONG adjoint
on the slope rows, and reciprocity goes Mode-12 blind to a slope-row
transpose.

Why the key is the space and not the mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original argument for moving the allocator off the mesh
(:ref:`sn-c5-phantom-retirement`, the C5.2 rank-honesty carve) was that
a mesh-side ``zeros_angular_flux`` has to know the *storage layout of
every leaf type*, so each new leaf grows another method on the mesh.
Keying on the space finishes that inversion rather than reversing it,
and it removes the residual duplication the mesh-keyed sugar still
carried:

* **The derivation ran twice.**  ``Leaf.zeros_on(mesh)`` re-derived the
  leaf's space from the mesh on every call, while the carrier was
  already caching exactly that space for the operators to bind against.
  Two routes to one object is the Pattern-2 smell; now there is one, and
  a field's space is ``is``-identical to the operator's domain instead of
  merely equal to it.
* **The widening knob was a proxy.**  ``spatial_moments=`` was an ``int``
  that every call site had to re-read off the scheme and thread through
  a factory.  A property choice cannot be threaded wrong.
* **Absence became spellable.**  A carrier that does not host a family
  mints ``None`` for it, so a leaf that cannot exist on a given carrier
  is refused at the composite seam with a diagnosis (below) rather than
  materialising as a zero-DOF field.

Composites are allocated one level up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A composite is an ELEMENT of its direct-sum space (CS4b S4), so its
allocator is space-keyed exactly like a leaf's.
:meth:`FullField.zeros <orpheus.transport.full_field.FullField.zeros>`
and
:meth:`TimedFullField.zeros <orpheus.transport.timed_full_field.TimedFullField.zeros>`
take the two leaf **classes** plus the carrier's cached
:class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace` (and,
for the timed carrier, a ``history_depth``), and zero-allocate each
block on the matching block space through the one
:meth:`Field.zeros <orpheus.numerics.field.Field.zeros>` primitive::

    q = FullField.zeros(
        interior=AngularFlux,
        boundary=AngularBoundaryFlux,
        space=sn.full_field_space,
    )

``[M]`` on the slab above this yields blocks ``(4, 2, 4)`` and ``(16,)``.
Naming the classes rather than calling each leaf's allocator is what
keeps the container cross-method generic: diffusion and CP pass their
own scalar leaves through the same body.

System B's :math:`\psi_{1/2}` composite is the presence-gated case.
:meth:`RadialCharacteristicField.flux_zeros <orpheus.transport.radial_characteristic_field.RadialCharacteristicField.flux_zeros>`
and its source-role sibling
:meth:`~orpheus.transport.radial_characteristic_field.RadialCharacteristicField.source_zeros`
take the carrier's
:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.radial_characteristic_field_space`,
which is ``None`` exactly where no :math:`\mu`-level consumes an
independent starting direction — and ``None`` is REFUSED with the R12a
diagnosis (*"System B is absent on this carrier…"*) rather than
silently allocating an empty block.  ``[M]`` on a 4-cell reflective
sphere with the same quadrature the blocks read ``(16,)`` (cells) and
``(4,)`` (the :math:`r = R` corner); on the slab the same call raises.
Presence is block existence, and the diagnosis now lives at the
composite seam instead of inside a leaf factory.

Mutability discipline
---------------------

:class:`AngularFlux` and :class:`ScalarFlux` are **frozen**
(``@dataclass(frozen=True)``) — every dunder operation returns a fresh
instance.  This matches the typical algebra-of-record usage where
intermediates carry meaning (Pattern 3 — named intermediates) and
in-place mutation would obscure provenance.

:class:`AngularBoundaryFlux` is **mutable** by design — the sweep's
persistent-BC contract is a write-through cache, and reflective-BC
partners read the previous-sweep outgoing-face writes.  Forcing a
fresh allocation per sweep would force memory churn the production
hot path cannot afford.

Cross-references
----------------

* :ref:`scattering-matrix-convention` for the ``SigS[g_from, g_to]``
  convention these types' arithmetic respects.
* :doc:`/theory/foundations/operator_algebra` for the six-operator algebra
  ``A ψ = (1/k) F ψ``, ``A = L + C − S − N_2n − B``, that the typed fields
  read as.


.. _theory-sn-typed-sources:

Typed source types
==================

Issue #197 PR-TYPED-3 introduces two typed source-density carriers
that wrap the right-hand side of the transport equation
:math:`A\,\psi = q`, with the loss composite
:math:`A = L + C - S - N_{2n} - B`:

* :class:`~orpheus.transport.source_sinks.ScalarSourceSink` — the isotropic
  volumetric source :math:`Q(\vec r, g)`, shape ``(ng, nx, ny)``.
  Aggregates per-group P0 in-scatter, (n,2n), and fission
  contributions that every ordinate sees identically.
* :class:`~orpheus.transport.source_sinks.AngularSourceSink` — the per-ordinate
  source :math:`Q^{\rm aniso}(\vec r, \hat\Omega_n, g)`, shape
  ``(N, ng, nx, ny)``.  Carries the :math:`P_\ell \ge 1` Galerkin
  reconstruction contribution plus any MMS-style external source.

.. todo:: Archivist expansion needed.

   The rich-narrative version should walk through:

   * The math of why iso + aniso = per-ordinate (broadcast across the
     N axis is the algebraic content of the dunder).
   * How the typed dunder dissolves the procedural
     ``np.broadcast_to(Q_iso[None, :, :, :], psi.shape).copy(); Q +=
     Q_aniso`` pattern that historically lived inside
     :meth:`~orpheus.transport.operators.scattering.ScatteringOperator.apply`.
   * Why source and flux types stay distinct (same storage shape;
     different algebraic role; cross-type addition undefined).
   * The decision to keep ``sig_t`` as bare ndarray (static-parameter
     quantities don't need typed wrappers — Issue #197 plan).

   Source module: :mod:`orpheus.transport.source_sinks`.
   Foundation tests:
   :file:`tests/sn/primitives/test_typed_source_sinks.py` (37 cases).
   Closeout memo:
   :file:`.claude/agent-memory/method-implementer/issue_197_pr_typed_3_closeout.md`.

The load-bearing dunder
-----------------------

The cross-type
:meth:`~orpheus.transport.source_sinks.scalar_source_sink.ScalarSourceSink.__add__`
accepting an
:class:`~orpheus.transport.source_sinks.AngularSourceSink` partner is
the load-bearing pattern of PR-TYPED-3.  It replaces the procedural
broadcast-and-copy pattern with a single algebraic line::

    # Before PR-TYPED-3 (scattering.py ``apply`` body):
    Q = np.broadcast_to(Q_iso[None, :, :, :], psi.shape).copy()
    Q += Q_aniso

    # After PR-TYPED-3:
    combined: AngularSourceSink = iso_source + aniso_source

The new path reads as the math: the iso → per-ordinate broadcast is
internal to the dunder (the canonical subspace-containment injection
``iso → 1 ⊗ iso`` — :class:`ScalarSourceSink` lives in the subspace of
:class:`AngularSourceSink` where every ordinate carries the same
value); the caller spells the algebra.  The cross-class dunder is
**preserved** under the refined Issue #207 principle (cross-class
arithmetic is permitted when there is a canonical subspace-containment
relation between the operands' spaces) — see the
:mod:`orpheus.transport.source_sinks` module docstrings.

Zero-source allocation
----------------------

The source leaves allocate exactly like the flux leaves — same base
:meth:`Field.zeros <orpheus.numerics.field.Field.zeros>`, same carrier
mints (:ref:`theory-sn-typed-fields`), role being class identity rather
than a different construction path:

* ``ScalarSourceSink.zeros(mesh.bulk_space)`` → ``(ng, *spatial)`` zeros.
* ``AngularSourceSink.zeros(sn.angular_bulk_space)`` →
  ``(N, ng, *spatial)`` zeros; read
  :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_trial_space`
  instead when the source carries the scheme's within-cell moments (the
  LD slope source :math:`\hat Q`).
* ``AngularBoundarySourceSink.zeros(sn.angular_trace)`` → the flat
  trace buffer, and the bottom rung of the prescribed-inflow ladder
  (:ref:`bc-affine-channel-where-q-travels`).

The ownership of this allocator has moved **twice** — off
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` onto the leaf at #346
(as ``zeros_on(mesh)``), then off the mesh key onto the space key at
CS4b S5.  The reasoning for both steps, and the measured shapes, are in
:ref:`theory-sn-typed-fields`; only the *key* changed at the second
step, never the owner.

Cross-type ``__add__`` table
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Left operand
     - Right operand
     - Result type
   * - :class:`ScalarSourceSink`
     - :class:`ScalarSourceSink`
     - :class:`ScalarSourceSink` (within type)
   * - :class:`ScalarSourceSink`
     - :class:`AngularSourceSink`
     - :class:`AngularSourceSink` (broadcast across N)
   * - :class:`AngularSourceSink`
     - :class:`ScalarSourceSink`
     - :class:`AngularSourceSink` (commutative — delegates to the
       :class:`ScalarSourceSink` side)
   * - :class:`AngularSourceSink`
     - :class:`AngularSourceSink`
     - :class:`AngularSourceSink` (within type)

The cross-type with :class:`~orpheus.transport.fields.scalar_flux.ScalarFlux` /
:class:`~orpheus.transport.fields.angular_flux.AngularFlux` is **not** defined.
Source density and flux carry the same numpy storage shape but are
different physical quantities — keeping the types distinct enforces
the algebraic distinction at the dunder layer.

Typed sweep input
-----------------

The production within-group sweep is a **typed operator surface**: the
loss operator's ``solve``
(:meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve`) consumes
a typed source field and returns a typed angular-flux field; the isotropic
and per-ordinate source contributions ride the one typed source object,
not a separate ``Q_aniso=`` keyword.  The bare
``(Q, sig_t, boundary_flux)`` ndarray contract survives one layer down, in
the representation ``sweep`` methods
(:meth:`~orpheus.sn.loss_representation.CumprodScan.sweep` /
:func:`~orpheus.sn.loss_representation._sweep_jacobi`), where the internal
hot path consumes bare ndarray throughout.

(The pre-#197 ``transport_sweep``-era typed/bare deprecation alias — the
one-cycle ``Q_aniso=`` keyword and the bare/typed dual entry — retired with
the operator-free ``transport_sweep`` entry at step 6, R-6.1.)

The collision gain's emission — which type comes out of which tier
------------------------------------------------------------------

There are three tiers, and each has exactly one output convention.  ⛔
This section described two *public verbs*, ``add_iso_source`` and
``build_aniso_source``, until 2026-09-06 — including a "return-new under
typed input" arm that ``add_iso_source`` never had (its signature was
``(Q: ndarray, phi: ndarray) -> None`` for its whole life).  Both verbs
retired at #448 with the hand-built eigenvalue-finalize source that was
their only production caller
(:doc:`ERR-083 </theory/verification/error_catalog>`); what ships is the
tier ladder below.

* **The array verb** —
  :meth:`~orpheus.transport.material_field.TransferMaterialField.add_p0_source`
  (with its transpose sibling and the per-\ :math:`\ell`
  ``moment_source`` pair): raw ``np.ndarray`` in, **mutates in place**,
  returns ``None``.  This is the per-material dispatch — the loop, the
  gathered ``einsum``, the yield — and since #426 step 2 it is the SAME
  verb for both channels, scaled by the datum's multiplicity.
* **The energy binding** —
  :meth:`IsotropicTransfer.apply
  <orpheus.transport.operators.isotropic_transfer.IsotropicTransfer.apply>`:
  bare ndarray of the domain's shape in, bare ndarray out (the
  model-portable contract; ``admit_array`` is the admission).  It
  allocates and calls the array verb, so nothing mutates a caller's
  buffer.
* **The lifted gain** — :meth:`TransferOperator.apply
  <orpheus.transport.operators.transfer.TransferOperator.apply>`: the
  composite ``FullField`` in, composite out, whose **interior leaf is an**
  :class:`~orpheus.transport.source_sinks.AngularSourceSink` and whose
  trace leaf is the zero source/sink (a collision gain is volumetric).
  The interior is the producer-side combine
  :math:`(\text{iso}/W) + \text{aniso}`: the :math:`\ell = 0` part as a
  :class:`~orpheus.transport.source_sinks.ScalarSourceSink` from the
  energy binding, the :math:`\ell \ge 1` part from the
  construction-selected redistribution body.  The type chain is preserved
  for :math:`N_{2n}` exactly as for :math:`S`, since #426 step 2 made them
  two instances of one binding.

Cross-references
----------------

* :ref:`theory-sn-typed-fields` for the PR-TYPED-2 flux carriers
  that share storage shape with the source types.


Gotchas and subtleties
======================

ny=1 singleton --- do NOT squeeze
---------------------------------

1-D problems are stored with the trailing ``ny = 1`` axis preserved:
:math:`\psi.\text{shape} == (N, n_g, n_x, 1)`, NOT ``(N, ng, nx)``.

The reason is uniform broadcasting: most SN operations broadcast
across the trailing axis (``ng \cdot V[:, None]``, ``Σ_t \cdot \phi``,
etc.).  With ``ny`` preserved as a singleton, every 1-D operation
uses the same numpy expression as its 2-D counterpart.  Squeezing
``ny`` would force per-dimension branching in every consumer.

**The single exception** is the per-cell cross-section slice in
:meth:`._OneDimScanWalk.sweep`: ``sig_t_1d = sig_t[:, :, 0]`` strips the
trailing axis for the cache primitive's ``(ng, nx)`` contract.
This is a localised slice, not a layout decision; the result is
re-broadcast to ``ny=1`` at the sweep's public-API exit.

Test fixture construction order
-------------------------------

Test fixtures construct sources, cross sections, and fluxes in the
principled order directly:

.. code-block:: python

   Q = np.zeros((ng, nx, ny))                  # principled
   sig_t = np.full((ng, nx, ny), sigma_t)      # principled
   psi = np.empty((N, ng, nx, ny))             # principled
   external = rng.standard_normal((N, ng, nx, ny))

No test should construct in legacy order and then transpose.  The two
remaining transposes in
:file:`tests/sn/test_2d_octant_sweep_equivalence.py` (cases 4--5) are
documented adapters that build sources via a broadcast against
``np.array([...])[None, None, :]`` (per-group profile times spatial
profile) and then transpose to principled --- this is a readability
choice, not a layout slip.


.. _sn-index-convention-future-work:

Future work
===========

PR-INDEX-7 --- EquationMap packed-vector traversal flip (obsoleted)
-------------------------------------------------------------------

.. note:: **Obsoleted by deletion (typed-field campaign, #197 /
   Depth-B D-H–D-J).**  This planned flip targeted the packed-vector
   internal layout of the FD-matvec path.  That path — the
   ``transport_operator_matvec_*`` family, the
   ``solution_to_angular_flux*`` unpackers, and the ``EquationMap``
   codec — was **deleted** in favour of the single typed
   :class:`~orpheus.transport.full_field.FullField` composite.  With no
   packed vector left, the flip below is moot; it is retained only as a
   record of the migration that was superseded by the deletion.

At the time, the FD-matvec internal helpers
(``solution_to_angular_flux`` and the
``transport_operator_matvec_*`` family, since deleted) carried a
Fortran-flatten layout for the packed vector
:math:`\texttt{solution}[g + n_g \cdot k_{\text{eq}}]` with
:math:`k_{\text{eq}}` enumerating cells via
``for iy: for ix: for n:``.  The unpacked result was
``fi.shape == (ng, N, nx, ny)``.

The planned PR-INDEX-7 flip would have moved this to the principled
``(N, ng, nx, ny)`` traversal:

- Reverse the ``EquationMap`` enumeration order to put :math:`n`
  outermost.
- Update the Fortran-reshape pair at every
  ``solution.reshape(ng, n_eq, order='F')`` /
  ``lhs.ravel(order='F')`` site.
- Update the ``fi[:, n, i, j]`` indexing sites in
  ``transport_operator_matvec_*``.
- Retire the two ``np.transpose(fi, (1, 0, 2, 3))`` axis-swap
  adapters at the Krylov decode sites.

The deletion of the whole packed-vector path resolved all of the above
at once — no separate PR-INDEX-7 was needed.

Typed-field contract resume
---------------------------

The typed-field contract is the natural successor to the index
migration.  The field catalog landed at
:ref:`sn-field-vocabulary` above is the durable Sphinx-side
vocabulary; the design-side detail is in the corrected memo at
``.claude/agent-memory/explorer/typed_field_contracts_for_phase_g.md``
(the memo predated the layout discovery; every shape mention there
has been corrected to the principled layout — see the memo's
correction banner).  The resume plan with PR boundaries and
acceptance criteria lives in
``.claude/plans/principled_index_migration.md`` §10.  The public
API reference at :doc:`/api/discrete_ordinates` will gain typed
signatures as the resume PRs land.

The dataclasses on first landing are:

.. code-block:: python

   @dataclass(frozen=True, slots=True)
   class AngularFlux:
       values: np.ndarray   # (N, ng, nx, ny) — principled
       sn_mesh: "SNMesh"    # by-reference

   @dataclass(frozen=True, slots=True)
   class ScalarFlux:
       values: np.ndarray   # (ng, nx, ny) — principled
       sn_mesh: "SNMesh"

The dataclasses land on the principled foundation laid by
PR-INDEX-5; the principled-layout :ref:`sn-field-vocabulary`
section names every flux / source / rate / trace type the resume
will eventually surface as a typed field.  Every operator-leaf's
``apply`` signature becomes
``apply(psi: AngularFlux) -> AngularFlux``, with the six-operator
algebra :math:`(L + C - S - N_{2n} - B - F/k).\texttt{apply}(\psi)`
distributing through :class:`~orpheus.numerics.operator.OperatorSum`
unchanged.
This closes the Issue #197 Wave 1 partial as documented in the memo's
§6.

Joint-batch ordinate_scan for curvilinear
-----------------------------------------

The M--M angular thread in curvilinear sweeps is sequential across
ordinates within a :math:`\mu`-level, which forces the curvilinear
sweep to keep a per-ordinate scan.  A research-level reformulation
of the M--M recurrence as a parallel-prefix scan over ordinates
would unlock joint-batching for curvilinear at the same scale
PR-INDEX-1 unlocked for slab.  The estimated win is 3--10× sweep
speedup on cylindrical pin-cell problems.  See the migration plan's
§7 deferred-work register.

JAX / GPU port
--------------

Under the principled layout, the ``(N, ng)`` leading batch maps
cleanly to a GPU grid dimension or to ``jax.vmap(scan, axes=(0,
0))``.  The cell axes :math:`(n_x, n_y)` map to the block dimension.
The migration plan's §7 lists this as the natural follow-up to the
typed-field contract resume.


Cross-references
================

- :ref:`theory-discrete-ordinates` --- SN method theory page; the
  Key Facts header references this convention.
- :ref:`operator-algebra` --- the six-operator algebra; every leaf's
  ``apply`` consumes / returns arrays in the convention defined
  here.
- :ref:`scattering-matrix-convention` --- the cross-section matrix
  convention, unchanged by the migration.
- :ref:`synthetic-xs-library` --- the verification cross sections
  used by the regression snapshots.
- ``.claude/plans/principled_index_migration.md`` --- the migration
  plan with the per-PR scope and the deferred-work register.
- ``.claude/agent-memory/method-implementer/issue_196_pr_index_*_closeout.md``
  --- the per-PR closeout memos with verbatim test paste-back.


References
==========

The layout derivation is grounded in the standard SN textbook
treatment of the within-group source iteration.

- :cite:`LewisMiller1984` §4.5 ("Source Iteration") --- block-diagonality
  of the within-group system.
- :cite:`AdamsLarsen2002` §III --- the SAILOR / Larsen-Adams
  preconditioned-Krylov framework that motivates the
  ``(N, ng)`` joint-batch storage.
- :cite:`BaileyMorelChang2010` (the curvilinear :math:`\alpha` recursion) and
  Eq. 43 (Morel--Montry weights) --- the curvilinear M--M angular
  thread that obstructs joint-batch over ordinates and motivates
  the principled :math:`n` leading layout.
- :cite:`Blelloch1990` §1.5 ("First-Order Linear Recurrences") --- the
  closed-form scan factorisation in :eq:`blelloch-1990-eq-1-5` that
  underlies :func:`~orpheus.sn.sweep.scan.ordinate_scan` and the
  slab joint-batch hot path.
- :cite:`Brent1974` --- Brent's theorem on work-efficient associative-scan
  reduction.  The pair-monoid associativity test in
  ``tests/sn/sweep/core/test_ordinate_scan.py`` is the algebraic
  justification for the closed form.
