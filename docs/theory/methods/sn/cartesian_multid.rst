.. _sn-cartesian-multid:

Cartesian multi-D: space enters the walk
========================================

This chapter broadens exactly one axis of the slab chapters:
**space**. Streaming becomes a true directional gradient — two (in 3-D,
three) sink terms instead of one — and with it the :term:`sweep` stops being a
chain and becomes a **wavefront over a causal dependency graph**. What
does *not* change is the algebra: Cartesian geometry keeps a neutron's
direction constant in flight, so there is still no angular coupling
outside the sources, the group axis of :doc:`slab_multigroup` rides
along untouched, and the within-group operator keeps its honest shape
:math:`A = L + C - S - N_{2n} - B` (:eq:`sn-within-group-with-n2n`) with
:math:`L+C` invertible in one pass.
The chain of the book repeats on the new axis:

1. **the invariant** — sinks = sources, now on a rectangular cell with
   faces on every axis → *pose* the 2-D balance;
2. **the operator** — :math:`L+C` remains **lower-triangular in sweep
   order**, but "sweep order" is now a *partial* order: each octant
   induces a causal DAG whose anti-diagonal levels are mutually
   independent;
3. **the matrix picture** — the loss inverse factors as a **direct sum
   over octants** (:eq:`streaming-inverse-direct-sum`), each block a
   forward substitution over its DAG;
4. **the strategy-encoding operators** — the wavefront walk realized as
   frozen primitives (graph, storage walks, level operations, kernel
   pair), and the representation layer that selects among schedules
   (:doc:`loss_representation`).

The angular :term:`quadrature` now genuinely uses both direction cosines
(:doc:`angular_quadrature`); the generic discretization machinery is
still :doc:`/theory/foundations/discretization`, cross-linked never
re-derived.

.. admonition:: Key Facts
   :class: tip

   * The 2-D transport equation is :eq:`transport-cartesian-2d` — two
     streaming sinks, and **still no angular coupling between
     ordinates**: space broadens the walk, not the angle algebra.
   * The 2-D DD balance :eq:`dd-cartesian-2d` applies the diamond
     closure on **both** axes simultaneously; the streaming
     coefficients are :math:`s_x = 2|\mu_x|/\Delta x`,
     :math:`s_y = 2|\mu_y|/\Delta y`.
   * Cells on an anti-diagonal :math:`i + j = k` share no faces —
     they are solved **simultaneously**. The sweep is a per-octant
     batched forward substitution over a precomputed causal DAG, and
     the loss inverse is the **direct sum over octants**
     :eq:`streaming-inverse-direct-sum`.
   * The walk factors into three layers — **storage walk × level
     operation × kernel pair** — so a closure supplies only its
     storage-free cell algebra and inherits both storage policies
     (full cochain / rolling frontier) and both directions
     (solve / apply) for free.
   * The multi-D LD closure is the **tensor-product bilinear (UBLD)**
     :math:`\{1, x, y, xy\}` cell system :eq:`ld-ubld-cell-system` —
     the :math:`xy` cross moment is diffusion-limit-load-bearing
     (simplex-P1 fails the thick limit on quadrilaterals), and sweep
     and matvec share one :math:`d`-generic kernel through the octant
     moment-frame involution :eq:`ld-ubld-octant-moment-frame-signs`
     (ERR-061).
   * The SI layer rides the walk: the persistent iterate is
     **windowed to harmonic moments** (:math:`N \to (L{+}1)(2L{+}1)`,
     2-D Cartesian only — the curvilinear Carlson seed and Krylov
     stay full-angular), and the reflective coupling runs the
     **boundary Gauss-Seidel** splitting
     :math:`M = L+C-B_{\rm lower}`, each shared reflective face
     reduced after its LAST outflowing octant group (ERR-056) — a
     boundary-transient accelerator, NOT a scattering accelerator.
   * That splitting is **not a** *regular* **splitting** in Varga's
     sense, so **no comparison theorem bounds it**: the multi-D DD
     face-to-face transmission has :math:`d-1` eigenvalues of exactly
     :math:`-1` (:eq:`dd-face-transmission-spectrum`), and boundary G-S
     is measured *slower* than Jacobi on some configurations
     (:ref:`sn-boundary-gs-not-regular`). The rate is not bounded
     either way; the **bulk** fixed point is schedule-invariant, and
     the **trace** is not — close :math:`\ge 2` reflective axis pairs
     and :math:`A` is *exactly singular*, so the schedule selects a
     member of a solution manifold rather than a point, and the solver
     returns the canonical one (:ref:`sn-loss-kernel-gauge`).
   * Boundary conditions apply **once per octant per axis**, never per
     ordinate — the L7 trap (ERR-003): :term:`per-ordinate <ordinate>` BC application is
     redundant in cost and order-sensitive in correctness.
   * The per-cell operation order is **bit-identity-load-bearing**:
     different *schedules* of the same operator are compared by
     principled-equivalence gates; different *storage policies* of one
     schedule are bit-identical (``array_equal``).


The posing: a second streaming sink
===================================

The invariant is unchanged — **sinks = sources** on every region of
phase space (:doc:`slab_one_group`, The Posing). Space enters through
the streaming term alone: on a 2-D Cartesian phase space the beam at
:math:`(x, y, \hat\Omega_n)` leaks through faces on both axes.

In two Cartesian dimensions the :term:`angular flux` depends on two direction
cosines :math:`\mu_x` and :math:`\mu_y`:

.. math::
   :label: transport-cartesian-2d

   \mu_x \frac{\partial \psi}{\partial x}
   + \mu_y \frac{\partial \psi}{\partial y}
   + \Sigt{} \, \psi
   = \frac{Q}{W}

There is no angular coupling between ordinates --- each direction is
solved independently.  The two streaming terms are the only difference
from the 1D case.


Nothing else moved. Collision is the same multiplication operator,
scattering and fission the same group-coupling operators of
:doc:`slab_multigroup` — the sources see a longer spatial index, not a
new structure. This is the payoff of the Cartesian structural fact:
**the spatial axis broadens the walk, and only the walk.**


.. _balance-cartesian-2d:

The discrete balance on a rectangular cell
==========================================

The cell balance now carries face terms on every axis — the
one-equation-*three*-unknowns shape (cell average + two downstream
faces) that the closure must reduce per axis
(:doc:`/theory/foundations/discretization` §3):

Integrating :eq:`transport-cartesian-2d` over a rectangular cell
:math:`\Delta x_i \times \Delta y_j`:

.. math::
   :label: balance-cartesian-2d-eq

   \mu_{x,n}\bigl[\psi_{i+\frac12,j} - \psi_{i-\frac12,j}\bigr] \Delta y_j
   + \mu_{y,n}\bigl[\psi_{i,j+\frac12} - \psi_{i,j-\frac12}\bigr] \Delta x_i
   + \Sigt{} \Delta x_i \Delta y_j\, \psi_{n,i,j}
   = S_{i,j}\, \Delta x_i \Delta y_j

.. (vv-status rationale) Derivation step: the integrated rectangular-cell
   balance. Its terminal DD form (dd-cartesian-2d) drives the tested 2-D
   solve; definitional, not a solver claim.
.. vv-status: balance-cartesian-2d-eq documented

Dividing through by :math:`\Delta x_i \Delta y_j` and applying
:term:`diamond-difference <diamond difference>` closures in **both** directions simultaneously:

.. math::

   \psi_{n,i} &= \tfrac{1}{2}(\psi^x_{\rm in} + \psi^x_{\rm out})
   \qquad\text{(x-closure)} \\
   \psi_{n,i} &= \tfrac{1}{2}(\psi^y_{\rm in} + \psi^y_{\rm out})
   \qquad\text{(y-closure)}

yields the 2D DD equation:

.. math::
   :label: dd-cartesian-2d

   \psi_{n,i,j}
   = \frac{S_{i,j}
     + s_x\, \psi^x_{\rm in}
     + s_y\, \psi^y_{\rm in}}
     {\Sigt{} + s_x + s_y}

where the streaming coefficients are:

.. math::
   :label: dd-cartesian-2d-streaming-coeffs

   s_x = \frac{2|\mu_{x,n}|}{\Delta x_i}, \qquad
   s_y = \frac{2|\mu_{y,n}|}{\Delta y_j}

.. (vv-status rationale) Notation: the per-axis streaming coefficients
   s_x, s_y. Definitional; the DD update they feed is exercised by the
   2-D transport gates.
.. vv-status: dd-cartesian-2d-streaming-coeffs documented

Both outgoing face fluxes are then updated from the DD closure:

.. math::

   \psi^x_{\rm out} = 2\psi_{n,i,j} - \psi^x_{\rm in}, \qquad
   \psi^y_{\rm out} = 2\psi_{n,i,j} - \psi^y_{\rm in}

These are precomputed by :class:`SNMesh` as ``streaming(0)[n, i]`` and
``streaming(1)[n, j]``, so the inner loop in
:func:`_sweep_jacobi` reduces to a single vectorised division per
diagonal.


.. _sweep-wavefront:

Cartesian 2D: Anti-Diagonal Wavefront Sweep
=============================================

In 2D, the DD equation :eq:`dd-cartesian-2d` creates a data dependency:
cell :math:`(i, j)` requires incoming face fluxes from its upwind
neighbours in both :math:`x` and :math:`y`.  Cells along an
**anti-diagonal** :math:`i + j = k` are mutually independent because
they share no incoming faces, so they can be solved simultaneously.

The wavefront sweep is implemented as a **per-octant batched
forward-substitution** over a precomputed causal cell DAG (Wave 2 of
the SN performance plan, closing Issue #4).  This subsection states
the algebraic framing; the primitives that realise it
(:class:`~orpheus.sn.loss_representation.sweep_graph.OctantLabel`,
:class:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph` and its two
storage walks, the level-operation pair ``_CellSolve`` /
``_CellResidual``, and the discretization's kernel pair
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.cell_kernel_batch`
/ :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.residual_kernel_batch`)
are documented in detail at
:ref:`sweep-octant-dependency-graph` immediately below.

The §15A.2 sum-of-tensor-products framing
-----------------------------------------

Following Grand Report v3 §15A.2 (lines 2137–2171), the loss inverse
(the sweep) :math:`(L+C)^{-1}` on the 2-D Cartesian SN field
space decomposes as a **direct sum over angular octants** — the block
structure is streaming-induced, since each octant sweeps in a fixed
direction (in this section's equations :math:`A` abbreviates the loss
composite :math:`L+C`, the invertible sub-composite of the chapter's
:math:`A = L+C-S-N_{2n}-B`):

.. math::
   :label: streaming-inverse-direct-sum

   A^{-1} \;=\; \bigoplus_{\sigma \in \mathcal{O}} A^{-1}_{\sigma},
   \qquad
   \mathcal{O} \;=\; \{\sigma = (\mathrm{sgn}\,\mu_x,\,
                                  \mathrm{sgn}\,\mu_y) :
                       \sigma \neq (0,0)\}
                  \,\cup\, \{(0,0)\},

.. vv-status: streaming-inverse-direct-sum documented

acting on the octant-restricted tensor space :math:`(N_\sigma,\,n_x,\,n_y,\,n_g)`.
The direction-cosine partition (Eq. :eq:`octant-sign-predicate`) is
the predicate the
:class:`~orpheus.numerics.quadrature.Quadrature` class exposes as
its cached :attr:`~orpheus.numerics.quadrature.Quadrature.octants`
property — a tuple of
:class:`~orpheus.numerics.measure.DiscreteMeasurePartition`
entries realised by
:meth:`~orpheus.numerics.measure.DiscreteMeasure.partition_by`
(see :ref:`tensorial-framing` and the
:doc:`/theory/foundations/discrete_measures` consumer table).

For each non-degenerate octant :math:`\sigma`, the action of
:math:`A^{-1}_\sigma` is a **forward substitution along a per-octant
causal cell DAG** — the topological order is structural (anti-diagonal
sweep on the Cartesian grid), the per-level cell update is one
vectorised einsum.  The pure-:math:`z` degenerate octant
:math:`\sigma = (0,0)` (ordinates with :math:`\mu_x = \mu_y = 0`,
which appear in 3-D angular cubatures projected to the in-plane
2-D problem) has no spatial streaming and reduces to a per-cell
balance :math:`\psi = Q / \Sigma_t` — the wavefront sweep handles
it via a short-circuit and skips the dependency graph.

**The four quadrant sweeps.**  Each non-degenerate octant
:math:`\sigma = (\mathrm{sgn}\,\mu_x, \mathrm{sgn}\,\mu_y) \in \{-1,+1\}^2`
determines a sweep direction:

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - :math:`\mu_x`
     - :math:`\mu_y`
     - *x*-direction
     - *y*-direction
   * - :math:`+`
     - :math:`+`
     - left :math:`\to` right
     - bottom :math:`\to` top
   * - :math:`-`
     - :math:`+`
     - right :math:`\to` left
     - bottom :math:`\to` top
   * - :math:`+`
     - :math:`-`
     - left :math:`\to` right
     - top :math:`\to` bottom
   * - :math:`-`
     - :math:`-`
     - right :math:`\to` left
     - top :math:`\to` bottom

For each octant, the sweep visits topological levels
(anti-diagonals) :math:`k = 0, 1, \ldots, n_x + n_y - 2`.  On level
:math:`k`, the cells :math:`(i, j)` satisfying :math:`i + j = k`
(in the per-octant traversal index space) are gathered into a numpy
batch and solved with a single vectorised evaluation of
:eq:`dd-cartesian-2d` — vectorised across the **ordinate axis**
(:math:`N_\sigma` — every ordinate in the octant), the
**anti-diagonal axis** (:math:`n_{\rm diag}` — number of cells on
this level), and the **group axis** (:math:`n_g`) simultaneously.

**Vectorisation within each level.**  Each level contains up to
:math:`\min(n_x, n_y)` cells.  The incoming face fluxes
``psi_in_x`` and ``psi_in_y`` are gathered by advanced indexing;
the DD equation is evaluated as one numpy operation; and the
outgoing face fluxes are scattered back into the persistent face-
flux buffers.  There is **no Python-level cell loop within a level**
and **no Python-level ordinate loop within an octant** — both axes
are internal to the einsum.

**Reflective BCs in 2D.**  At each boundary face, the incoming flux
for ordinate :math:`n` is set to the outgoing flux of its reflected
partner.  For the left/right boundaries (*x*-reflection), the partner
is ``ref_x[n]`` (negating :math:`\mu_x`); for the top/bottom boundaries
(*y*-reflection), the partner is ``ref_y[n]`` (negating :math:`\mu_y`).
The pairings are derived from the mirror motions by the quadrature's
:meth:`~orpheus.numerics.quadrature.Quadrature.ordinate_permutation`
at realization.  Crucially, the BC apply happens
**once per octant per axis** (not once per ordinate per axis) —
see :ref:`sweep-octant-dependency-graph-l7-trap` for the rationale.

Implemented in :func:`~orpheus.sn.loss_representation._sweep_jacobi`, which
is a thin orchestrator over the
:class:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph` primitives
described next.


.. _sweep-octant-dependency-graph:

Cartesian 2D: Octant Dependency Graph
=====================================

This section documents the **§15A.2 "upwind trace complex / causal
transport DAG / direction sweep ordering" primitive** as it lives in
:mod:`orpheus.sn.loss_representation.sweep_graph` after Wave 2 of the SN performance plan
(branch ``feature/sn-octant-sweep-graph``, closes Issue #4).  The
shipped architecture replaces the legacy per-ordinate ``for n in
range(N)`` loop in :func:`~orpheus.sn.loss_representation._sweep_jacobi` with
a per-octant batched dispatch, lifting the per-call ``_diag_cache``
build to mesh-time work, and isolating the per-cell DD algebra in the
discretization's pure kernel pair
(:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch`
/ :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch`)
that LD / EC / Step closures can override later.

.. note::

   **Architecture history — the dispatch surface re-layered twice.**
   Wave 2 (the original closure of Issue #4) routed the sweep through
   a per-level *packet* (the ``SweepCellSlice`` dataclass) consumed by
   four direction×storage methods — ``update_batch`` / ``residual_batch``
   on the strategy (full-field) plus their ``apply_windowed`` /
   ``residual_windowed`` siblings on the graph.  S6.4(e) **collapsed
   that surface**: the four walk methods became TWO storage walks
   (:meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_full`,
   :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_windowed`)
   each parameterised by a level-operation OBJECT (``_CellSolve`` for
   the solve direction, ``_CellResidual`` for the apply direction —
   direction is never a boolean flag); the per-level ``SweepCellSlice``
   packet was retired (it existed only to feed the now-deleted storage
   adapters); and the strategy's ``update_batch`` / ``residual_batch``
   were replaced by the **storage-free kernel pair**
   ``cell_kernel_batch`` / ``residual_kernel_batch`` (pure cell
   algebra — no gather/scatter).  The historical names ``update_batch``
   / ``residual_batch`` / ``SweepCellSlice`` appear below only as
   *history*; the current contract is the kernel pair + the level
   operations.  See :ref:`sweep-dispatch-relayering` for the WHY.

The primitives
--------------

The architecture is a small set of frozen, individually unit-tested
primitives plus a mesh-time precompute step.

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Primitive
     - Lives in
     - Role
   * - :class:`~orpheus.sn.loss_representation.sweep_graph.OctantLabel`
     - :mod:`orpheus.sn.loss_representation.sweep_graph`
     - Frozen + slotted dataclass carrying one direction sign per
       spatial axis (``signs[axis] ∈ {-1, 0, +1}``) — a single type
       labels a 1-D (``(±1,)``), 2-D (``(±1, ±1)``), or 3-D octant.
       Hashable; used as the key in the per-shape graph family
       :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.for_shape`
       (owned by the ``_DAGWavefront`` representation family since
       S6.4(c) — historically a mesh attribute).  An all-zero
       signature denotes the pure-:math:`z` degenerate octant — no
       graph is built for it
       (:attr:`~orpheus.sn.loss_representation.sweep_graph.OctantLabel.streams` is
       ``False``).  The 3-D ``sign_z`` is dropped by the 2-D Cartesian
       orchestration: the in-plane sweep is invariant under the
       out-of-plane axis, so multiple ordinates with the same in-plane
       ``signs`` but different ``sign_z`` share a single graph instance.
   * - :class:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph` (+ its
       two storage walks)
     - :mod:`orpheus.sn.loss_representation.sweep_graph`
     - Frozen dataclass holding the per-octant topological levels
       (anti-diagonals) and the per-axis face-index offsets.  Built
       once per ``(shape, octant)`` pair in the
       :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.for_shape`
       cache (S6.4(c); historically at mesh construction); reused
       across every source iteration / Krylov matvec / outer
       iteration.  Exposes TWO storage walks (S6.4(e)):
       :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_full`
       carries the COMPLETE per-axis interior face cochain (the
       verification-oracle policy);
       :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_windowed`
       advances a rolling :math:`(d{-}1)`-frontier window (the
       production policy, ``O(N·n_g·∏ n_a)`` shrunk to
       ``O(N·n_g·∏_{a<d−1} n_a)`` backing).  The walk owns the level
       loop, the storage, and the per-level operand extraction; it
       dispatches the cell algebra to a level operation (next two rows).
   * - The level-operation pair ``_CellSolve`` / ``_CellResidual``
     - :mod:`orpheus.sn.loss_representation.sweep_graph`
     - The **direction fork, as OBJECTS** (S6.4(e); direction is never
       a boolean flag).  Exactly ONE is constructed per octant walk; the
       storage walk calls ``level_op.cell(...)`` per topological level.
       ``_CellSolve`` runs the solve direction — calls the strategy's
       ``cell_kernel_batch`` then performs the Phase-5c angular-XOR-
       moment per-level emit (write the angular flux + accumulate the
       scalar flux, OR accumulate the harmonic-moment tensor, never
       both).  ``_CellResidual`` runs the apply direction — calls
       ``residual_kernel_batch`` then writes the per-level residual.
       The per-level *emit* expressions and their order are
       bit-identity-load-bearing — relocated verbatim from the four
       retired walk methods.
   * - The kernel pair
       :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.cell_kernel_batch`
       / :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.residual_kernel_batch`
     - :mod:`orpheus.transport.spatial.scheme`
     - The **storage-free extension point** on the
       :class:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase` ABC
       (S6.4(e); historically the ``SweepCellSlice``-packeted
       ``update_batch`` / ``residual_batch``).  Each takes the per-axis
       incoming face fluxes + streaming coefficients + the level's cross
       section and source and returns ``(psi_avg, psi_out)`` (solve) or
       ``(residual, psi_out)`` (apply) — PURE cell algebra, no
       gather/scatter (that is the walk's job).  Default raises
       :exc:`NotImplementedError` — additive capability, not a contract
       change.  :class:`~orpheus.transport.spatial.diamond.DiamondDifference`
       overrides the pair; LD / EC / Step closures override it later to
       join the batched wavefront walks (their per-cell
       :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.update`
       stays the canonical reference contract).

Per-shape precompute pattern (family-owned since S6.4(c))
---------------------------------------------------------

The dependency graph is a **derived object** — the
``(shape × octant)`` joint property.  It depends only on cell topology
and the octant sign convention; it does **not** depend on fluxes,
sources, BCs, quadrature, cross sections, or iteration state.  So the
graph build is paid once **per spatial shape** in the cached accessor
:meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.for_shape`, owned
by the DAG-consuming ``_DAGWavefront`` representation family:

.. code-block:: python

   class _DAGWavefront(_LossRepresentation):
       @property
       def sweep_graphs(self) -> dict[OctantLabel, SweepDependencyGraph]:
           # cached per shape; same-shape meshes share byte-identical graphs
           return SweepDependencyGraph.for_shape(self.mesh.spatial_shape)

**Ownership history** (two relocations, each a pure refactor):

#. *Wave 2 / C2.4* lifted the per-call ``_diag_cache`` build that
   previously lived inside the 2-D wavefront sweep (rebuilt once per
   sweep call) to **mesh-construction** time — a measurable but
   second-order saving on the 421-group benchmark; the structurally
   important effect was making the graphs named, inspectable state.
#. *S6.4(c)* moved ownership **off the mesh onto the representation
   family**: the mesh is pure geometry, and only the two DAG-walking
   representations (the window + the full-field oracle) ever mention
   the substrate.  This retired the curvilinear
   ``mesh.sweep_graphs = None`` slot — an illegal state (a mesh
   carrying a "no DAG here" marker for a structure it never owned) —
   and replaced mesh-lifetime caching with per-SHAPE caching, so
   same-shape meshes share one graph family (the graphs carry no
   mesh-identity information).  DAG-free representations
   (``CumprodScan``, ``ScanMarch``) and curvilinear meshes simply
   never touch the accessor; curvilinear sweeps walk the cell graph
   differently (per-ordinate march; see
   :meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.dag_walk`).

The closed-form precompute lives in
:meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.from_cartesian`
and never appears in the sweep loop.  This is structural, not
hand-rolled — the "library version" (a generic topological-sort over
an explicit DAG) would be over-engineering for a regular pattern that
collapses to ~5 lines of ``arange`` + anti-hyperplane extraction.  The
builder is dimension-generic (``d = len(shape) ∈ {1, 2, 3}``): a d=1
chain, the d=2 anti-diagonal, a d=3 anti-hyperplane.

The §15A.2 invariant set
-------------------------

The Grand Report v3 §15A.2 (lines 2165–2171) prescribes a fixed set
of L0 invariants every ``SweepDependencyGraph`` instance must satisfy.
These are pinned by ``tests/sn/test_sweep_graph.py`` (63 L0 tests):

* **Upwind orientation** — for each octant
  :math:`\sigma = (\mathrm{sgn}\,\mu_x, \mathrm{sgn}\,\mu_y)`, the
  ``face_in_x`` and ``face_out_x`` offsets satisfy
  ``face_in_x + face_out_x == 1`` and
  ``face_in_x = 0`` iff :math:`\mathrm{sgn}\,\mu_x \ge 0` (and
  analogously on :math:`y`).  Asserted by
  ``test_face_pairing_consistent`` and ``test_upwind_orientation``.
* **Topological sort** — every level's cells depend only on cells in
  strictly earlier levels (under the per-octant orientation).  No
  intra-level dependencies; no back-edges.  Asserted by
  ``test_topologically_sorted``.
* **Cell coverage** — every cell :math:`(i, j) \in [0, n_x) \times
  [0, n_y)` appears in **exactly one** level.  Disjoint union over
  the topological levels reconstructs the full grid.  Asserted by
  ``test_cell_coverage``.
* **Face-pairing consistency** — the incoming-face index of cell
  :math:`(i, j)` on level :math:`k` matches the outgoing-face index
  of its upwind neighbour on level :math:`k - 1` (under the per-
  octant orientation).  Asserted by
  ``test_face_pairing_consistent``.

These four invariants are the **load-bearing correctness floor** of
the wavefront sweep.  Any future closure (LD, EC, Step) plugged in
via the kernel pair consumes the same invariants — they describe
the topology, not the algebra, so the strategy contract is orthogonal
to the graph correctness.

.. _sweep-dispatch-relayering:

The dispatch boundary: walk (scheduler) vs cell update (closure)
-------------------------------------------------------------------

A central architectural decision is the **separation between the
scheduler and the closure**.  Three layers stack from storage outward
to algebra (S6.4(e)):

#. **The storage walk** —
   :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_full` (full
   cochain) or
   :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_windowed`
   (rolling frontier).  Owns the topological-level loop and the
   per-axis face gather/scatter (full cochain) or the frontier
   seed/incoming/emit/shed cochain trace algebra (window).  Storage is
   the walk's concern — the SAME two walks serve every closure and
   both directions.

#. **The level operation** — ``_CellSolve`` or ``_CellResidual``,
   constructed once per octant walk and called as ``level_op.cell(...)``
   per level.  Owns the direction fork (solve vs apply) and the
   per-level *emit* (angular/moment write, or residual write).
   Direction is an OBJECT, never a boolean flag passed down the walk.

#. **The kernel pair** —
   :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch`
   /
   :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch`.
   Owns the **pure cell algebra** and nothing else — no gather, no
   scatter, no storage. This is the ONLY direction-aware math left in
   the SN spatial stack.

**Why this layering (the WHY behind S6.4(e)).**  Wave 2 carried the
storage concern *inside* the strategy: the DD ``update_batch`` /
``residual_batch`` methods gathered the cell's face inputs from the
``SweepCellSlice`` packet, ran the algebra, and scattered the outgoing
faces back — a four-method direction×storage product
(``update_batch`` / ``residual_batch`` full-field +
``apply_windowed`` / ``residual_windowed`` windowed).  That entangled
two orthogonal concerns: a NEW closure (LD / EC / Step) would have had
to re-implement the gather/scatter (storage) plumbing four times just
to supply its cell math.  S6.4(e) lifts storage to the walk layer
**once, above every strategy**, so a closure supplies ONLY its
storage-free kernel pair (pure algebra over the per-axis incoming face
fluxes) and inherits both storage policies (full + window) and both
directions (solve + apply) for free.  The ``SweepCellSlice`` packet —
which existed only to feed the retired storage adapters — is gone with
them.  This is the Cardinal-Rule-2 "build primitives, not products"
discipline: the four-method product collapses to a 2 (walks) × 1
(level-op pair, direction-by-object) × 1 (kernel pair) factoring where
each factor varies independently.

This means: **DD is the only shipping closure today**, but Step / LD
/ EC override the kernel pair later without touching the walk driver
or the level operations.  The Wave C-extension rollout (Issues #157 /
#158) ships the per-cell :meth:`update` method first as the canonical
reference contract; the batched kernel pair is the parallel
level-vectorised capability for closures whose per-cell algebra
vectorises across an ``(N_oct, n_diag, ng)`` slice without per-cell
branching.

The DD ``cell_kernel_batch`` reproduces the legacy 2-D wavefront DD
math **bit-identically** (operation order matters; see
:class:`~orpheus.transport.spatial.diamond.DiamondDifference` docstring on
bit-identity).  The math is the **balance form** of WDD on a 2-D
Cartesian cell:

.. math::
   :label: dd-2d-balance-form

   \overline{\psi}_{i,j}
   \;=\; \frac{Q_{i,j}
               + s_{x,i}\,\psi^{\rm in}_{x,i,j}
               + s_{y,j}\,\psi^{\rm in}_{y,i,j}}
              {\Sigma_{t,i,j} + s_{x,i} + s_{y,j}},
   \qquad
   s_{x,i} = \frac{2|\mu_x|}{\Delta x_i},
   \quad s_{y,j} = \frac{2|\mu_y|}{\Delta y_j},

.. vv-status: dd-2d-balance-form documented

with the spatial closure
:math:`\psi^{\rm out}_x = 2\overline{\psi} - \psi^{\rm in}_x`
(and analogously on :math:`y`).  The operation order is fixed at:

.. code-block:: text

   denom    = sig_t + sx + sy
   psi_avg  = (Q + sx * psi_in_x + sy * psi_in_y) / denom
   psi_out  = 2 * psi_avg - psi_in

Algebraically equivalent rearrangements (e.g., reordering
``sig_t + sx + sy`` to ``sx + sy + sig_t``) break the 1-ULP regression
contract even though the math is identical.  This is the canonical
**bit-identity vs principled-equivalence** instance from the
``vv-principles`` skill — the regression contract is bit-identity
gated on the existing snapshots; deviations are admissible only when
backed by a structurally-independent reference (e.g., :math:`k_\infty`
analytical limit on a homogeneous reflective problem).

.. _sweep-octant-dependency-graph-l7-trap:

The L7-trap fix: BC apply once per octant
------------------------------------------

Wave 2 closes a class of bugs that the test-architect dispatch
identified as the **L7 trap** — the design pattern where a sweep
driver re-applies a boundary operator at each ordinate iteration.
The legacy ``_sweep_jacobi`` had this shape:

.. code-block:: python

   # legacy — the L7 trap
   for n in range(N):
       psi_x = sn_mesh.bc_xmin.apply(psi_x, quad)[n]   # per-ordinate apply
       psi_y = sn_mesh.bc_ymin.apply(psi_y, quad)[n]   # per-ordinate apply
       # ... walk cells, sweep, etc. ...

Each ``bc.apply`` call sees the FULL ``(N, ny, ng)`` face buffer (so
reflective partners can read across rows) and returns an updated full
buffer.  Calling this :math:`N` times per sweep is wrong on two
counts:

1. **Cost** — :math:`N` redundant invocations of the same boundary
   operator on the same buffer.  For a 2-D ``LS-N`` quadrature with
   :math:`N \sim 30`–:math:`80`, this is the dominant per-sweep cost
   on small meshes.
2. **Correctness** — when reflective BCs interact with mid-sweep
   reflective-buffer state, **the order of BC apply vs ordinate
   sweep matters**.  The legacy code's behaviour is sensitive to
   ordinate iteration order; reorderings that algebraically should
   be no-ops (e.g., octant batching, parallel ordinate evaluation)
   silently change the converged solution.

The Wave-2 form applies each boundary operator **once per octant per
axis** — :math:`O(\text{octants}) = 4` calls, not :math:`O(N)`:

.. code-block:: python

   # Wave 2 — L7-trap closed by construction
   for octant in quad.octants:                    # 4 iterations, structural
       sx, sy = octant.label
       ...
       # Apply BC once for this octant on each axis
       if sx_eff >= 0:
           full_face_x = sn_mesh.bc_xmin.apply(psi_x[:, 0, :, :], quad)
           psi_x[oct_idx, 0, :, :] = full_face_x[oct_idx]
       else:
           full_face_x = sn_mesh.bc_xmax.apply(psi_x[:, nx, :, :], quad)
           psi_x[oct_idx, nx, :, :] = full_face_x[oct_idx]
       # ... analogously on y ...
       sweep_graph.walk_windowed(level_op=_CellSolve(...), ...)  # all N_oct batched

The architectural argument: the boundary operator's *semantics* are
"map outgoing partner-octant fluxes to incoming this-octant fluxes".
That mapping is per-octant by construction — applying it once per
ordinate within an octant is redundant; applying it once per octant
is structurally correct.

.. note::

   The ``sn_mesh.bc_xmin.apply(..., quad)`` spellings in the two
   code blocks above are **historical** (the Wave-2 era 2-arg
   ``apply`` on a per-attribute BC surface). Both spellings are
   retired: the 2-arg ``apply`` by Issue #186 (the law is now a pure
   descriptor; the realizer produces a strict 1-arg operator — see
   :ref:`bc-trace-law-descriptor-model`), and the per-attribute
   ``bc_<face>`` surface by C4 / #220 in favour of the
   face-name-keyed :attr:`SNMesh.bc` dict
   (``sn_mesh.bc["xmin"].apply(psi)`` — see
   :ref:`bc-face-name-carve`). The blocks are preserved verbatim
   because they document the *L7-trap structure* the Wave-2 carve
   closed, which is independent of the storage spelling.

The L7-trap detector test
``tests/sn/test_2d_octant_sweep_equivalence.py::case-3`` is the
load-bearing regression gate — a TESTS-FIRST harness (case 3 with
mixed reflective + vacuum BCs, 2G heterogeneous, ``n_sweeps=2``)
designed to fail if any future refactor reintroduces the per-ordinate
BC apply pattern.  The case-3 design uses the post-sweep ``psi_x`` /
``psi_y`` buffer state as bit-identity oracles (rather than the
converged scalar flux), because the L7-trap is invisible in
single-iteration tests: the FIRST iteration's reflective-buffer state
is zero, so per-ordinate vs once-per-octant give the same answer; the
trap surfaces only on the SECOND iteration when the first iteration's
outgoing-face writes feed the second iteration's BC apply.  The case
also explicitly tags ``@pytest.mark.catches("ERR-003")``: ERR-003 is
the catalogued instance where reflective-BC ordering coupled with
ordinate batching produced a converged-but-wrong solution.

Bit-identity to the legacy implementation
-----------------------------------------

For LS-family quadratures (``LevelSymmetricSN``,
``ProductQuadrature``) whose ordinate ordering is octant-grouped in
lexicographic order, the Wave-2 implementation is **bit-identical**
to the legacy per-ordinate loop on every regression snapshot — the
existing
``tests/sn/regression/snapshots/2d_1g_LS4_dd_15x15.npz``,
``test_apply_2d_cartesian_bit_identical_to_legacy``, and
``test_unified_sweep_dispatch`` snapshots all pass with
``np.array_equal``.  The argument has three parts:

1. **BC apply equivalence.**  The boundary operator for octant
   :math:`\sigma` reads partner-octant rows of the persistent
   ``psi_x`` / ``psi_y`` buffer.  For LS, the lex order of
   ``quad.octants`` matches the legacy n-order at the
   partner-state granularity, so the same iteration's value is
   observed at the same point.  Per-octant BC apply produces the
   same ``psi_x`` / ``psi_y`` octant-row contents as :math:`N_\sigma`
   copies of the legacy per-ordinate apply.
2. **Per-cell sweep equivalence.**  Within an octant, per-ordinate
   cell sweeps are independent — different rows of ``psi_x`` /
   ``psi_y``, different rows of ``angular_flux``.  Batching is
   therefore bit-identical to any per-ordinate sequencing of the
   same set, modulo the per-cell DD operation order which is
   pinned (see :ref:`sweep-octant-dependency-graph` dispatch
   boundary above).
3. **Lebedev (octant ordering not lex).**  For Lebedev quadrature
   (case 6 in the C2.5 harness) the converged scalar flux matches
   the legacy code, but the iter-to-iter values differ on the
   inner iteration (different traversal order ⇒ different
   Gauss-Seidel updates).  Case 6 uses **vacuum BCs**, where the
   partner-state semantics don't matter; this is a deliberate
   choice in the harness — Lebedev with reflective BCs would
   require redesign of the bit-identity gate, but the converged
   answer is still verified at the snapshot level via
   ``test_unified_sweep_dispatch``.

This taxonomy follows the ``vv-principles`` skill's
**bit-identity vs principled-equivalence** discipline: bit-identity
where structurally trivial (LS-family, octant-grouped lex order);
principled equivalence (closed-form L1 anchor + MMS regression suite)
where bit-identity would require more work than the engineering value
returns.

.. _sweep-octant-architecture-cardinal-rule-2:

Architectural framing (Cardinal Rule 2)
---------------------------------------

Per the project memory note ``project_moc_structure.md`` and the
:ref:`cell-update-strategies` discussion,
:class:`SweepDependencyGraph` is **SN-specific by design**.  MoC
will define its own analog (per-ray traversal) — different DAG
shape, different mathematical structure (fiber bundles + solution
sheaves over characteristic curves rather than a topological sort
over a cell graph).  There is **no shared SweepGraph Protocol**
because there is no shared mathematical structure.  Cardinal Rule 2
(architecture) prefers **late unification** ("unify after two
instances" — see ``feedback_unify_after_two_instances`` in agent
memory) to premature abstraction; the sweep DAG lives in
:mod:`orpheus.sn` and stays there until a second mathematically-
similar consumer arrives, which by current understanding is **never**
for MoC and only conjectural for any other deterministic transport
solver.

By contrast, the **angular octant partition** primitive
(:meth:`~orpheus.numerics.measure.DiscreteMeasure.partition_by`) is
genuinely shared infrastructure — see the cross-method consumer
table at :doc:`/theory/foundations/discrete_measures` (octant partition consumed by SN
2-D, MoC track-bundle direction grouping, MC boundary-current
hemisphere scoring, future SN boundary realiser).  The split is
**measure-level primitives are shared, sweep-level orchestration
is SN-specific**.

Performance
-----------

The Wave-2 plan target for Issue #4 closure was 3–10× speedup on
the 421-group benchmark (the canonical ``test_profile_421g``
smoking-gun probe).  The shipped speedups:

.. list-table::
   :header-rows: 1
   :widths: 35 20 15 30

   * - Configuration
     - Speedup
     - Target?
     - Comment
   * - 421-group LS4 ``31×31``
     - 1.7×
     - Below
     - numpy-dispatch-overhead-dominated regime
   * - 2-group LS4 ``31×31``
     - 2.78×
     - At lower bound
     - per-octant batching wins more for small ``ng``
   * - 1-group LS4 ``15×15`` (regression snapshot)
     - bit-identical
     - n/a
     - regression contract preserved

The headline 421-group speedup is below the 3-10× target.  The
honest analysis: the Wave-2 implementation eliminates the
:math:`N`-fold ordinate loop overhead but the per-octant per-level
kernel calls still number :math:`O(\text{levels} \times
\text{octants}) \approx (n_x + n_y - 1) \times 4 \approx 88` per
sweep on a ``31 × 31`` mesh, each carrying its own numpy dispatch
cost.  At 421 groups, the per-call work scales linearly so the
ratio of useful work to dispatch overhead remains modest.  The
**follow-up direction** noted at Wave 2 was to carry full-:math:`N`
buffers plus an ``octant_indices`` field so the kernel calls become
level-only (~ 60 calls / sweep) rather than ``levels × octants``
(~ 240 calls / sweep), eliminating the per-octant copy round-trip.
The subsequent Phase 5 / S6.4 work took a different route to the
same end: the rolling-frontier window
(:meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_windowed`)
holds the interior cochain on a contiguous :math:`(d{-}1)`-frontier
slab, turning the per-level gather into a basic-slice zero-copy view
(a measured ``~0.77×`` contiguity speedup AND a ``~3×`` peak-memory
win at d=2) — see :ref:`wavefront-flux-cochain`.

Closing the smoking gun by construction is itself a load-bearing
result: the legacy ``for n in range(N)`` is gone, the metric
(angular-flux tensor; see :ref:`theory-sn-index-convention` for the
canonical ``(N, ng, nx, ny)`` storage) now knows its iterative
structure, and any future numpy-dispatch-cost reduction benefits
all closures uniformly through the strategy contract.

Verification
------------

The Wave-2 verification chain (per the ``algebra-of-record`` skill
discipline):

* **L0 unit tests** — on the primitives:

  - ``tests/sn/test_octants_property.py`` (across 8
    quadrature factories) — disjoint union, weight conservation,
    sign-signature correctness, pure-axis ordinates labelled
    ``sign=0``.
  - ``tests/sn/test_cell_kernel_batch.py`` (S6.4(e) successor of
    ``test_cell_update_batch.py``) — term-level L0 on the storage-free
    kernel pair (``cell_kernel_batch`` / ``residual_kernel_batch``):
    bit-identity against per-cell :meth:`update` on a
    single-cell-per-batch reduction; standalone tests against
    analytical DD recurrence on a 1×3 strip; 4-octant bit-identity vs
    the per-ordinate Python loop; plus a ``sha256`` source-of-record
    pin on the two kernel bodies (the explicit left-fold order is
    bit-identity-load-bearing).
  - ``tests/sn/test_sweep_graph.py`` — the §15A.2 invariant set above;
    anti-diagonal cell coverage; topo-order acyclicity per octant sign;
    BC face conventions; and the ``walk_full`` / ``walk_windowed`` ×
    level-operation walks (with ``window ≡ full`` bit-identity oracles).
  - ``tests/sn/primitives/test_dag_ownership.py`` (S6.4(c) successor of ``test_snmesh_sweep_graphs.py``) — graph
    contents agree with hand-derived schedule on a 3×3 mesh; dict
    keys equal ``quad.octants`` labels; cache invalidates when mesh
    changes.

* **L1 closed-form anchor + L7-trap detector** — the C2.5 TESTS-
  FIRST harness ``tests/sn/test_2d_octant_sweep_equivalence.py``
  (7/7 pass), tagged ``@pytest.mark.l1`` and
  ``@pytest.mark.catches("ERR-003")``.  Includes:

  - **case 3 (L7 trap)** — mixed BC + 2G heterogeneous +
    ``n_sweeps=2``, the primary L7-trap detector.
  - **case 7 (closed-form)** — 1G homogeneous reflective with
    :math:`k_\infty = \nu\Sigma_f / \Sigma_a`, the structural-
    independence anchor.
  - cases 1–6 covering BC mixes, ordinate batching corners, and
    Lebedev (vacuum-BC variant).

* **L2 regression** — existing ``tests/sn/verification/mms/test_mms_2d.py``,
  ``test_discrete_ordinates_2d.py``, ``test_streaming_operator.py``,
  ``test_streaming_operator_decomposition.py``,
  ``test_unified_sweep_dispatch.py``, ``tests/sn/regression/``: 56/56
  pass, 6 slow-marked skipped.

The verification chain is the canonical
**L0 (primitive invariants) → L1 (closed-form anchor + bug catcher)
→ L2 (integration regression)** ladder from the ``vv-principles``
skill.

References and pointers
-----------------------

* Grand Report v3 §15A.2 (lines 2137–2171) — the "upwind trace
  complex / causal transport DAG / direction sweep ordering"
  primitive description with the ``assert_*`` invariant set this
  module's tests pin.  Plan file at
  ``.claude/plans/neutron_transport_grand_report_v3.md``.
* Wave 2 plan at ``.claude/plans/transient-giggling-cake.md`` (C2.1
  through C2.8) — the architectural primitives plan, sequencing,
  verification-first harness design.
* Wave 0 :meth:`~orpheus.numerics.measure.DiscreteMeasure.partition_by`
  primitive — the measure-level partition that the SN ``octants``
  property delegates to.  See :doc:`/theory/foundations/discrete_measures`.
* Wave 1 :math:`R \circ \Lambda \circ M` Galerkin scattering
  composition — the parallel "metric knows its iterative structure"
  refactor for the scattering source build.  See
  :ref:`sn-scattering-fission-operators`.
* :class:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase` — the
  strategy ABC carrying the per-cell :meth:`update` reference contract
  and the storage-free batched kernel pair
  :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.cell_kernel_batch`
  / :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.residual_kernel_batch`
  (S6.4(e); was the ``SweepCellSlice``-packeted ``update_batch`` /
  ``residual_batch``).
* :class:`~orpheus.transport.spatial.diamond.DiamondDifference` — the only
  shipping closure that overrides the kernel pair; the reference for
  the bit-identity contract (pure cell algebra — the ONLY
  direction-aware math in the SN spatial stack since S6.4(e) lifted
  storage to the walk layer).
* :mod:`orpheus.sn.loss_representation.sweep_graph` — the two storage walks
  (:meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_full`,
  :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_windowed`)
  and the ``_CellSolve`` / ``_CellResidual`` level operations.
* C2.5 TESTS-FIRST harness:
  ``tests/sn/test_2d_octant_sweep_equivalence.py``.


Choosing the schedule: the representation layer
===============================================

Everything above is ONE schedule — the DAG wavefront — for one
operator. The full selection story is the capstone architecture page
:doc:`loss_representation` (the representation layer of
:math:`(L+C)`); this chapter states only what the multi-D walk adds to
it:

* :math:`(L+C)` is one lower-triangular object with two actions
  (``solve`` = forward substitution, ``apply`` = the row action); a
  :class:`~orpheus.sn.loss_representation.LossRepresentation` is a
  **schedule** for traversing that triangular structure, not a
  different operator.
* Four schedules ship:
  :class:`~orpheus.sn.loss_representation.CumprodScan` (the 1-D
  parallel-prefix scan of :doc:`slab_one_group`),
  :class:`~orpheus.sn.loss_representation.ScanMarch` (scan the
  contiguous axis, march the others — the **multi-D Cartesian
  production default**; the Fork-B2 rationale lives on the capstone),
  :class:`~orpheus.sn.loss_representation.MovingFrontierWindow` (the
  anti-diagonal wavefront above with a rolling
  :math:`(d{-}1)`-frontier — a selectable peer), and
  :class:`~orpheus.sn.loss_representation.FullFieldWavefront` (the
  same DAG schedule retaining the whole interior cochain — the
  verification oracle, explicit-select only).
* Selection is a single source of truth:
  :func:`~orpheus.sn.loss_representation.default_for` returns the
  first compatible entry of the ordered registry, and an illegal
  ``(representation, mesh)`` pairing is unrepresentable — the
  constructor re-checks
  :meth:`~orpheus.sn.loss_representation.LossRepresentation.supports`
  and raises.
* Whatever the schedule, **one d-generic walk frame serves sweep AND
  matvec** (the L21 invariant), forked only by a kernel object and an
  emit policy — never a boolean. The dependency graph and the kernel
  pair documented above are the shared substrate every multi-D
  schedule rides.

The historical Wave-D ``transport_sweep`` consolidation that first
unified the sweep paths — since retired in favour of this
representation polymorphism — is preserved as origin history in
:doc:`index` (the superseded "Unified sweep dispatch" section).


.. _ld-ubld-multidim:

Multi-dimensional LD: the tensor-product bilinear (UBLD) cell system
=====================================================================

The multi-dimensional analog of Linear Discontinuous on a **Cartesian**
cell is **NOT** the simplex-P1 :math:`\{1, x, y\}` object
(:math:`1+d` moments).  Adams (2001) proved simplex-LD *fails* the thick
diffusion limit on quadrilaterals, while the **bilinear / trilinear
DG-P1** (UBLD) — basis :math:`\{1, x, y, xy\}` (:math:`2^d` moments) —
*passes*.  The :math:`xy` cross moment is diffusion-limit-load-bearing.

The :math:`d`-generic per-cell Galerkin system is assembled as Kronecker
products of the verified 1-D LD factor operators (the streaming
:math:`\Omega\cdot\nabla = \sum_a \mu_a \partial_a` is a sum over axes;
the tensor-product basis separates):

.. math::
   :label: ld-ubld-cell-system

   A_{\rm cell}\,\vec\psi = \vec R, \qquad
   A_{\rm cell} = G + F_{\rm out} + \Sigma_t M, \qquad
   \vec R = M\,\vec S + F_{\rm in}\,\psi_{\rm in}^{\rm traces},

.. (vv-status rationale) Algebra-of-record: the assembled per-cell UBLD
   Galerkin system. Foundation-gated by the SymPy oracle
   (tests.transport.spatial.test_ld_ubld_symbolic / test_ld_ubld_primitive),
   not a solver claim.
.. vv-status: ld-ubld-cell-system documented

a :math:`2^d \times 2^d` dense non-symmetric solve, with
:math:`M = M_1 \otimes \cdots \otimes M_d` (mass),
:math:`G = \sum_a \mu_a\,(M_1 \otimes \cdots \otimes G_{1d} \otimes
\cdots \otimes M_d)` (streaming: gradient on the active axis, mass on the
transverse axes), and :math:`F_{\rm out}` likewise from the per-axis
downstream-face trace.

.. math::
   :label: ld-ubld-d1-reduction

   A_{\rm cell}\big|_{d=1} =
   \begin{bmatrix} \Sigma_t h + |\mu| & |\mu| \\
                   -|\mu| & \Sigma_t \theta h + |\mu| \end{bmatrix}

The :math:`d=1` reduction (Kronecker-with-one-factor identity) recovers
the production slab 2×2 :eq:`dd-cartesian-1d`-sibling exactly; the
:math:`xy` coupling falls out of the algebra for :math:`d \ge 2`.

.. math::
   :label: ld-ubld-exact-on-bilinear

   \psi(x,y) = a + bx + cy + dxy
   \;\Longrightarrow\;
   \vec\psi_{\rm solved} = \vec\psi_{\rm exact-projections}

.. (vv-status rationale) Correctness identity of the d>=2 closure (exact on
   any bilinear flux). The ERR-060 catcher test_d2_exact_on_bilinear
   (foundation, both the symbolic oracle and the numpy primitive) is the
   verifiable content.
.. vv-status: ld-ubld-exact-on-bilinear documented

The UBLD is **exact on any bilinear flux** (the multi-D analog of the
1-D "exact on linear-in-x" oracle), the :math:`xy` cross moment
exercised — the structurally-independent correctness gate for the
:math:`d \ge 2` closure.

The Branch-1 algebra-of-record (the UBLD weak form)
------------------------------------------------------

The canonical symbolic reference for the :math:`d`-generic UBLD system is the
SymPy module :mod:`orpheus.derivations.discrete.sn.ld_ubld` (the
algebra-of-record, State 1A closed-form): the Kronecker assembler
``assemble_ubld`` plus five ``derive_*`` verification functions, each proven by
``sympy.simplify(diff) == 0``.  It is the discrete-SN sibling of
``orpheus.derivations.discrete.sn.balance`` — a *symbolic discretization the
production solver must satisfy*, NOT a continuous reference.

The per-cell system descends from the Galerkin weak form of the within-group
streaming–collision operator (Maginot, Ragusa & Morel 2016, "Non-negative
Methods for Bilinear Discontinuous Differencing of the :math:`S_N` Equations on
Quadrilaterals", NSE 185(1):17–42, Eqs. 1–12).  Multiplying the transport
equation :math:`\Omega\cdot\nabla\psi + \Sigma_t\psi = S` by each basis function
:math:`B_i` and integrating over the cell :math:`K`, then integrating the
streaming term by parts (MRM-2016 Eq. 6),

.. math::
   :label: ld-ubld-weak-form

   \underbrace{(\Omega\cdot)\!\oint_{\partial K} \hat n\,B_i\,\psi\,d\ell}_{\text{surface (upwind)}}
   \;-\; \int_K \psi\,(\Omega\cdot\nabla B_i)\,dV
   \;+\; \Sigma_t\!\int_K B_i\,\psi\,dV
   \;=\; \int_K B_i\,S\,dV,

.. (vv-status rationale) Literature-transcribed Galerkin weak form
   (MRM-2016 Eqs. 1-12). Its assembled terminal (ld-ubld-cell-system) is
   foundation-gated; the derivation itself is definitional.
.. vv-status: ld-ubld-weak-form documented

gives three operators per cell — the **mass** :math:`M_{ij} = \int_K B_i B_j`
(the collision term), the **gradient/stiffness** :math:`G_{ij} = \int_K B_i\,(\Omega\cdot\nabla B_j)`
(the volumetric streaming term, coupling all :math:`2^d` moments), and the
**surface** matrix split per face into an OUTFLOW block (:math:`\Omega\cdot\hat n > 0`,
implicit, the cell's own unknowns) and an INFLOW block (:math:`\Omega\cdot\hat n < 0`,
**upwind**: the incoming face value is the upstream neighbour's outflow trace,
moved to the RHS).  Assembling gives exactly the dense system
:eq:`ld-ubld-cell-system`, :math:`A_{\rm cell} = G + F_{\rm out} + \Sigma_t M`,
:math:`\vec R = M\vec S + F_{\rm in}\,\psi_{\rm in}^{\rm traces}`.

Why bilinear, not simplex-P1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The naïve multi-D analog of 1-D LD — "cell average plus one slope per axis",
the simplex-P1 basis :math:`\{1, x, y\}` of :math:`1+d` moments — is the
**wrong object on a Cartesian cell**.  Adams (2001, "Discontinuous Finite
Element Transport Solutions in Thick Diffusive Problems", NSE 137(3):298–333)
proved that simplex-LD *fails* the thick diffusion limit on quadrilaterals,
while the **bilinear / trilinear DG-P1** (UBLD) — the tensor-product basis
:math:`\{1, x, y, xy\}` of :math:`2^d` moments — *passes* it.  The reason is the
:math:`xy` **cross moment**: it is exactly what the simplex basis lacks, and it
is the term the leading-order asymptotic diffusion balance needs (Börgers,
Larsen & Adams 1992, "The asymptotic diffusion limit of a linear discontinuous
discretization of a two-dimensional linear transport equation", JCP
98(2):285–300, give the 2-D rectangular analysis explicitly).  The simplex-P1
basis *does* preserve the limit on a genuine simplex (triangle/tetra) mesh
(Wareing, McGhee, Morel & Pautz 2001, NSE 138(3):256–268) — but that is a
different cell topology, not a quadrilateral.  ORPHEUS builds Cartesian cells,
so the :math:`2^d` tensor-product object is the diffusion-limit-consistent
choice; the choice is **load-bearing**, not a convenience (see
:ref:`ld-ubld-scattering-moment-lift` for the companion half of the same
asymptotic argument).

The Kronecker single-source build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The three matrices are NOT hand-transcribed entry-by-entry (that would be a
:math:`4\times4` / :math:`8\times8` transcription waiting for a sign error).
They are assembled as **Kronecker products of the verified 1-D LD factor
operators** in the Legendre moment basis :math:`\{1, P_1\}` on width :math:`h`:

.. math::
   :label: ld-ubld-kronecker-factors

   M_{1d} = \mathrm{diag}(h,\ \theta h),
   \qquad
   G_{1d} = |\mu|\begin{bmatrix} 0 & 0 \\ -2 & 0 \end{bmatrix},
   \qquad
   F_{\rm out}^{1d} = |\mu|\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix},

.. (vv-status rationale) Definition of the 1-D LD factor operators
   M_1d / G_1d / F_out. Foundation-gated by the Kronecker-assembly oracle
   (test_ld_ubld_symbolic / test_ld_ubld_primitive).
.. vv-status: ld-ubld-kronecker-factors documented

with the streaming :math:`\Omega\cdot\nabla = \sum_a \mu_a\,\partial_a` a sum
over axes (the tensor-product basis separates), so

.. math::
   :label: ld-ubld-kronecker-assembly

   M = M_1 \otimes \cdots \otimes M_d,
   \qquad
   G = \sum_a \mu_a\,(M_1 \otimes \cdots \otimes G_{1d}^{(a)} \otimes \cdots \otimes M_d),

.. (vv-status rationale) Structural assembly identity (the Kronecker
   single-source build). Foundation-gated (test_d3_assembles_8x8_with_theta_cubed,
   test_d1/d2 assembled-matrices-match-symbolic).
.. vv-status: ld-ubld-kronecker-assembly documented

i.e. the gradient acts on the active axis and **the mass on every transverse
axis** (the volume-integral factorization — this is the load-bearing build
choice).  The :math:`F_{\rm out}` surface matrix is assembled likewise; the
inflow is a :math:`B(-1) = [1, -1]` test-weighting on the active axis (mass on
transverse axes) times :math:`|\mu_{\rm axis}|`.  The diagonal mass weights are
then the Kronecker product of the per-axis diagonals — a power of
:math:`\theta = \tfrac13` equal to the **number of active (slope) axes** of each
moment:

.. math::
   :label: ld-ubld-mass-weights

   M_{ii} = \theta^{|i|},
   \qquad
   |i| = \#\{a : o_a = 1\}
   \;\Longrightarrow\;
   \begin{cases}
     1        & \bar\psi \quad (\text{no slope axis}) \\
     \theta   & \hat\psi_x,\ \hat\psi_y \quad (\text{one slope axis}) \\
     \theta^2 & \hat\psi_{xy} \quad (\text{two slope axes})
   \end{cases}

.. (vv-status rationale) Definitional diagonal Legendre mass M_ii = theta^|i|.
   Foundation-gated by the theta^|i| primitive-assembly tests
   (test_d3_assembles_8x8_with_theta_cubed).
.. vv-status: ld-ubld-mass-weights documented

so the 2-D weights are :math:`(1, \theta, \theta, \theta^2)` and the 3-D
:math:`xyz` cross moment carries :math:`\theta^3`.  These weights re-appear in
the matvec mass-normalization (:eq:`ld-ubld-unified-moment-residual`) — they are
the SAME diagonal Legendre mass.  The :math:`d=1` case is a Kronecker product
with a single factor (an identity), so it reduces EXACTLY to the production
slab 2×2 :eq:`ld-ubld-d1-reduction`; the :math:`xy` coupling *emerges* from the
algebra for :math:`d \ge 2` — no entry is hand-written.

The two oracles
~~~~~~~~~~~~~~~

The module proves the construction with two structurally distinct oracles
(both ``sympy.simplify(diff) == 0``):

* **Oracle (i) — the :math:`d=1` reduction.**
  ``derive_d1_reduction_to_production`` shows the assembled :math:`d=1` system
  equals the production
  :mod:`orpheus.transport.spatial.linear_discontinuous` 2×2 entry-for-entry, with the
  Schur complement :math:`S` and the effective slope denominator
  :math:`D_2' = \Sigma_t h\theta + |\mu|` recovered as the production closed
  forms.  Two further reductions
  (``derive_d1_kernel_view_equals`` / ``derive_d1_scan_view_equals``) prove the
  same :math:`d=1` reduces to BOTH the ÷V DAG kernel ``_kernel_terms`` and the
  ×V scan ``affine_scan_coefficients`` views — the "single-source the math"
  proof that Branch 2's three production views are the SAME algebra
  (:eq:`ld-ubld-rule-of-three-collapse`).

* **Oracle (ii) — exact-on-bilinear at :math:`d=2`.**
  ``derive_d2_exact_on_bilinear`` feeds an exactly-bilinear flux
  :math:`\psi = a + bx + cy + dxy` through the DG-exact upstream face moments and
  the projected source moments, and shows the 4 solved moments equal the exact
  projections (:eq:`ld-ubld-exact-on-bilinear`).  The :math:`xy` cross moment is
  genuinely exercised (:math:`d \ne 0` symbolically) — this is the multi-D
  analog of the 1-D "exact on linear-in-x" oracle and the structurally
  independent correctness gate for the :math:`d \ge 2` closure.

The foundation gate is :mod:`tests.transport.spatial.test_ld_ubld_symbolic` (6
``@pytest.mark.foundation`` tests, one per ``derive_*`` plus an anchor to the
live production ``LinearDiscontinuous.update``); the literature contract is
recorded in
``.claude/agent-memory/literature-researcher/multi_d_ld_closure.md`` (MRM-2016
Eqs. 1–12; the Adams-2001 thick-diffusion verdict; BLA-1992); the closeout is
``.claude/agent-memory/method-implementer/issue_240_d5b_s1_ld_ubld_branch1_closeout.md``.

.. admonition:: ERR-060 — the oracle that earned its keep
   :class: tip

   The first draft of ``assemble_inflow_axis`` dropped the :math:`|\mu_{\rm axis}|`
   streaming factor on the inflow RHS (failure Mode 3, a missing factor).  The
   bug was INVISIBLE to all three :math:`d=1` oracles — the :math:`d=1` RHS is
   built inline, never routed through the multi-axis inflow assembler — and was
   caught by Oracle (ii), the :math:`d=2` exact-on-bilinear gate, which is the
   first consumer of ``assemble_inflow_axis``.  Mutation-verified
   :math:`-O`-safe: re-dropping the factor turns the :math:`d=2` test red while
   the :math:`d=1` tests stay green (proving they are blind to the bug class).
   This is the algebra-of-record discipline working as designed — the bug never
   reached production.  (The ERR-060 marker belongs on the *exact-on-bilinear*
   gates, NOT on the cell-matrix ``A == A`` pin (the code binds
   :math:`A_{\rm cell}` as ``A``), which checks ``assemble_ubld``'s
   ``A``/``M``/``G``/``F_out`` and is structurally blind to the
   dropped inflow factor — see ``error_catalog.rst`` ERR-060.)

.. _ld-ubld-branch2-primitive:

The Branch-2 production primitive + the single-sourced d=1 fast path
------------------------------------------------------------------------

The numpy production counterpart of the symbolic algebra-of-record above
is :mod:`orpheus.transport.spatial._ubld`, in two layers that share ONE source
of truth.  Layer 1 is the :math:`d`-generic dense primitive
(``assemble_ubld`` / ``per_cell_solve``): a batched-over-cells Kronecker
build of the :math:`2^d \times 2^d` system :eq:`ld-ubld-cell-system`,
solved with a batched :func:`numpy.linalg.solve`.  It is the CANONICAL
:math:`d`-generic source for both :math:`d=1` (today) and :math:`d \ge 2`
(S2 wires the bilinear cell-batch kernel onto it); in production
:math:`d=1` does **not** route through this dense solve (that would be
the per-cell-solve performance regression).

Layer 2 is the shared :math:`d=1` closed form ``d1_closed_form`` — the
analytic Schur complement of the primitive's :math:`d=1` 2×2, VECTORIZED
over the cell / ordinate / group stack (no dense solve), so the
production :math:`d=1` path stays on the fast path.  The entire closure
rides two SCALE-FREE invariants:

.. math::
   :label: ld-ubld-scale-free-invariants

   k = \frac{g/\theta}{g/\theta + \Sigma_t}, \qquad
   w = \frac{1}{1 + k}, \qquad g = \frac{|\mu|\,A_{\rm down}}{V},

.. (vv-status rationale) Definition of the scale-free closure invariants
   (k, w, g). Foundation-gated by the d1_closed_form primitive tests.
.. vv-status: ld-ubld-scale-free-invariants documented

with ``w`` the cell-average blend weight
(:math:`\bar\psi = (1-w)\psi_{\rm in} + w\,\psi_{\rm out}`).  Every
production view's coefficients are an algebraic function of
:math:`(g, \Sigma_t, k, w)` times a power of the cell volume :math:`V`
(the ×V vs ÷V choice applied at the call site).

.. math::
   :label: ld-ubld-rule-of-three-collapse

   \texttt{\_schur\_terms}\;(\times V), \quad
   \texttt{\_kernel\_terms}\;(\div V), \quad
   \texttt{affine\_scan\_coefficients}\;(\times V\ \text{scan})
   \;\longleftarrow\; \texttt{d1\_closed\_form}

.. (vv-status rationale) Single-source structural identity (the three
   production views all derive from d1_closed_form). Foundation-gated by
   the view-equals-dense primitive tests (test_divV_kernel_view_equals_dense,
   test_timesV_scan_view_equals_dense, test_xV_schur_view_equals_dense).
.. vv-status: ld-ubld-rule-of-three-collapse documented

The three production 1-D views in
:mod:`orpheus.transport.spatial.linear_discontinuous` — the ×V per-cell Schur
(``_schur_terms``), the ÷V DAG kernel (``_kernel_terms``), and the ×V
scan (``affine_scan_coefficients``) — now ALL derive their coefficients
from ``d1_closed_form``, applying their ×V / ÷V / ×V-scan scaling at the
call site.  The LD 2×2 algebra (the Rule-of-Three) collapses to ONE
place, proven ``==`` the dense primitive's :math:`d=1` reduction
(symbolically by the Branch-1 oracles, numerically end-to-end by the
Branch-2 gate).

The numpy production counterpart descends from the SAME specialized SymPy
ancestor as the Branch-1 algebra-of-record above; only the evaluation strategy
differs (Branch 1 closes the algebra symbolically, Branch 2 evaluates it on
arrays).  The discipline is **construct general, select narrow, specialize only
on measured cost**:

* **Construct general — the dense primitive.**  ``assemble_ubld`` /
  ``per_cell_solve`` build and solve the :math:`2^d \times 2^d` system
  :eq:`ld-ubld-cell-system` for every :math:`d`, batched over the cell /
  ordinate / group stack with a single :func:`numpy.linalg.solve`.  This is the
  canonical :math:`d`-generic source — :math:`d=1` (today), :math:`d \ge 2`
  (S2 wires the bilinear cell batch onto it), :math:`d = 3` (trilinear).

* **Select narrow — the :math:`d=1` closed form.**  ``d1_closed_form`` /
  :class:`~orpheus.transport.spatial._ubld.D1ClosedForm` is the analytic Schur
  complement of the primitive's :math:`d=1` 2×2, VECTORIZED over the stack with
  no dense solve.  Both scale-free invariants of :eq:`ld-ubld-scale-free-invariants`
  drive it.

* **Specialize on measured cost.**  In production :math:`d=1` does **not**
  route through the dense solve — that would be the per-cell-solve performance
  regression (the L16 constraint).  The closed form keeps the production
  :math:`d=1` sweep on the vectorized fast path
  (:class:`~orpheus.sn.loss_representation.CumprodScan` rides the ×V scan view's
  :math:`(a, \mathrm{inverse\_denom}, w)`; the DAG kernel rides the ÷V arrays).

The Rule-of-Three collapse
~~~~~~~~~~~~~~~~~~~~~~~~~~

Before the carve, the LD 2×2 algebra was transcribed in three production views
that had drifted into three independent copies.  All three now derive their
coefficients from the single ``d1_closed_form`` source
(:eq:`ld-ubld-rule-of-three-collapse`), applying only their scaling at the call
site — the Cardinal-Rule-2 / `coding-elegance` Pattern-2 single-source collapse:

.. list-table:: The three production 1-D views — one algebra, three scalings
   :header-rows: 1
   :widths: 30 16 54

   * - Production view (in :mod:`orpheus.transport.spatial.linear_discontinuous`)
     - Scaling
     - Consumer
   * - ``_schur_terms``
     - :math:`\times V`
     - the per-cell Schur (the matvec / ``update`` / ``residual`` path)
   * - ``_kernel_terms``
     - :math:`\div V`
     - the scale-free DAG wavefront kernel (the :math:`d \ge 2` arm rides the
       ÷V system, :eq:`ld-ubld-divv-scale-free-kernel`)
   * - ``affine_scan_coefficients``
     - :math:`\times V` scan
     - the Blelloch parallel-prefix scan (the production :math:`d=1` sweep)

The ×V / ÷V / ×V-scan choice is the volume scaling applied to the same
coefficients: dividing the Galerkin balance by the cell volume :math:`V` leaves
a scale-free system in the per-axis streaming :math:`g_a = |\mu_a| A_{\rm down}/V`
and :math:`\Sigma_t` alone (the form the :math:`d \ge 2` kernel consumes — fed
unit widths and :math:`\mathrm{mus} = (g_0, \ldots)`, it reduces EXACTLY to
``d1_closed_form``); multiplying restores the volume-weighted per-cell Schur
(:math:`D_2' = \theta V\,d_2`, :math:`S_{\times V} = V\cdot\mathrm{eff\_denom}`).
Each view is proven ``==`` the dense primitive's :math:`d=1` reduction —
symbolically by the Branch-1 oracles above (``derive_d1_kernel_view_equals`` /
``derive_d1_scan_view_equals``), numerically end-to-end by the Branch-2 gate.

.. note::

   **A principled ~1-ULP re-baseline, not a bit-identity break.**  Routing the
   three LD views through the shared helper changes the floating-point
   *reduction tree* relative to the legacy inline associations: the helper
   computes :math:`(g, \Sigma_t, k, w)` once and forms each coefficient as an
   algebraic function of them, whereas the old inline code interleaved the
   multiplies and adds differently.  In exact arithmetic the values are
   identical; in IEEE-754 they differ at ~1 ULP because addition is not
   associative.  This satisfies all three `vv-principles` criteria for
   accepting a non-bit-exact change: every intermediate
   (:math:`g`, :math:`k`, :math:`w`) is a *named, inspectable* physics
   quantity; the value is verified against the structurally-independent
   Branch-1 symbolic oracle; and the drift is FP-non-associativity bounded by
   the reduction depth.  The LD gates carry ``rtol = 1e-12`` (far above the
   ULP-scale drift); the DD-only strict gate remains the **bit-identical
   negative control** (DD never reaches the LD helper — its :math:`w=\tfrac12`
   reconstruction is the exact power-of-two doubling that commutes with
   round-to-nearest).

The verification is :mod:`tests.transport.spatial.test_ld_ubld_primitive`:
the numpy primitive :math:`==` the SymPy oracle at :math:`d=1` (matrices +
moments) and exact-on-bilinear at :math:`d=2`; the shared closed form
:math:`==` the dense :math:`d=1` solve in all three views; and the LIVE
production scheme (``update`` / ``cell_kernel_batch`` /
``affine_scan_coefficients``) :math:`==` the dense primitive (the link proof).
The closeout is
``.claude/agent-memory/method-implementer/issue_240_d5b_s1_ld_ubld_branch2_closeout.md``.
The :math:`d \ge 2` hand-off (the bilinear cell-batch kernel + the
:math:`2^{d-1}`-moment face cochain wiring onto this primitive) is S2, the next
subsection.

.. _ld-ubld-d2-wavefront-wiring:

Wiring the d≥2 UBLD kernel onto the DAG wavefront (S2)
--------------------------------------------------------

Sub-step **D5b-S2** closes the :math:`d = 1`-only kernel raise so
Linear-Discontinuous runs in :math:`d \ge 2` on the DAG wavefront,
consuming the verified dense primitive.  Three contract widenings, all
GATED on a single scheme trait so Diamond Difference / Step stay
byte-identical:

.. math::
   :label: ld-ubld-n-spatial-moments

   \text{per-cell} = (\text{per\_axis})^{d}, \qquad
   \text{per-face} = (\text{per\_axis})^{d-1}, \qquad
   \text{per\_axis} =
   \begin{cases} 1 & \text{DD / Step} \\ 2 & \text{LD} \end{cases}

.. (vv-status rationale) Named-field-typing contract-sizing identity keyed
   on spatial_basis_per_axis. Foundation-gated by the field-space widening
   tests (test_spatial_moment_field_space).
.. vv-status: ld-ubld-n-spatial-moments documented

The class-level trait
:attr:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.spatial_basis_per_axis`
(the 1-D moment-basis size) indexes the whole contract via the
tensor-product structure: the per-cell unknown is
:math:`(\text{per\_axis})^d` (LD-2D: 4) and each downstream face carries
:math:`(\text{per\_axis})^{d-1}` transverse moments (LD-2D: 2).  The
boolean ``per_axis > 1`` gates the multi-moment face-cochain trailing
axis and the moment-reducing emit; DD/Step at ``per_axis == 1`` keep the
rank-:math:`r` scalar face and rank-3 ``psi_avg`` EXACTLY.

.. math::
   :label: ld-ubld-divv-scale-free-kernel

   A_{\div V}\,\vec\psi = M_{\div V}\,\vec S + \sum_a F_{\rm in}^{(a)},
   \qquad
   \psi_{\rm out}^{(a)}[t] = \psi[o_a{=}0,\,t] + \psi[o_a{=}1,\,t]

.. (vv-status rationale) The scale-free ÷V kernel form fed to the dense
   primitive. Foundation-gated (test_divV_kernel_view_equals_dense,
   test_production_kernel_equals_dense).
.. vv-status: ld-ubld-divv-scale-free-kernel documented

The :math:`d \ge 2` arm rides the **scale-free ÷V** form of the dense
system: dividing the Galerkin balance by the cell volume leaves a system
depending only on the per-axis ÷V streaming :math:`g_a = |\mu_a|/\Delta_a`
(the ``s_axes`` the kernel already receives) and :math:`\Sigma_t` — so the
dense assembler is fed unit widths and ``mus = (g_0, \ldots)``, reducing
EXACTLY to ``d1_closed_form`` at :math:`d=1`.  The :math:`d` downstream
faces are the trace of the tensor-Legendre solution at the downstream node
(:math:`P_0(+1) = P_1(+1) = 1` sums the :math:`o_a{=}0` and
:math:`o_a{=}1` blocks), in the :math:`2^{d-1}` transverse-Kronecker order
the next cell's upwind inflow consumes (out-of-cell == in-of-next-cell —
the closure consistency the matvec twin verifies).

The scale-free ÷V system fed to the dense primitive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :math:`d \ge 2` arm rides the **scale-free ÷V** form of the dense system
:eq:`ld-ubld-divv-scale-free-kernel`.  Dividing the Galerkin balance by the
cell volume leaves a system depending only on the per-axis ÷V streaming
:math:`g_a = |\mu_a|/\Delta_a` (the ``s_axes`` the kernel already receives) and
:math:`\Sigma_t`.  In code this means the dense assembler ``_ubld_system`` /
``per_cell_solve`` is fed **unit widths** (``hs = [1, …]``) and
``mus = (g_0, …, g_{d-1})`` — so at :math:`d = 1` it reduces EXACTLY to
``d1_closed_form`` (the ÷V view of the Rule-of-Three above), and at
:math:`d \ge 2` it is the same dense object the Branch-1 oracle proves
exact-on-bilinear.  The kernel dispatch lives in
:mod:`orpheus.transport.spatial.linear_discontinuous` (``cell_kernel_batch`` /
``residual_kernel_batch``): the :math:`d=1` closed-form fast path vs the
:math:`d \ge 2` dense ``_ubld_system`` / ``per_cell_solve``.

The downstream faces are the trace of the tensor-Legendre cell solution at the
downstream node: since :math:`P_0(+1) = P_1(+1) = 1`, the outgoing face on axis
:math:`a` sums the :math:`o_a = 0` and :math:`o_a = 1` blocks of the cell moment
vector, producing a :math:`2^{d-1}`-moment face object (average + transverse
slopes) in the transverse-Kronecker order the next cell's upwind inflow
consumes.  *Out-of-cell == in-of-next-cell* is the closure consistency the
matvec twin verifies.

The moment-ordering crosswalk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cell moment vector is the tensor (Kronecker) product of the per-axis 1-D
Legendre basis, ordered **x-outer / y-inner** so the all-:math:`P_0` cell
average is always slot 0 (the same convention the
:ref:`within-cell moment factor <spatial-moment-space>` carries,
:eq:`spatial-moment-kronecker-order`).  The Kronecker layout in
2-D is :math:`[\bar\psi,\ \hat\psi_y,\ \hat\psi_x,\ \hat\psi_{xy}]` (indexing
:math:`[o_x, o_y]` with :math:`o_x` outer); each downstream face carries its
:math:`2^{d-1}` transverse moments in the matching per-axis order.  The
crosswalk between the cell-moment order, the per-face transverse order, and the
downstream-node trace reconstruction is the design record's load-bearing detail
(``.claude/plans/issue_240_d5b_s2_crosswalk.md``; recovery anchor
``.claude/plans/issue_240_phase2_step_d5b_ubld.md``).

The DD bit-identity backward-compat invariant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three contract widenings — the dense kernel arm, the multi-moment
face-cochain trailing axis (:mod:`orpheus.sn.loss_representation.sweep_graph` ``_MovingFrontier``;
the ``_CellSolve`` / ``_CellResidual`` moment-reducing emit), and the window
zero-pad (:mod:`orpheus.sn.loss_representation`
``FullFieldWavefront._octant_face_cochain``, the ``_inflow_to_moments`` pad) —
are GATED on the single scheme trait
:attr:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.spatial_basis_per_axis`
of :eq:`ld-ubld-n-spatial-moments`.  At ``per_axis == 1`` (DD / Step) the tail
is the EMPTY tuple (:eq:`spatial-moment-append-policy`), so the scalar face and
the rank-3 ``psi_avg`` emit are kept byte-identical — DD backward-compatibility
falls out of the ``face_moment_tail`` formula, NOT an ``if scheme is DD`` branch.
This is the negative control: the DD/Step gate stays at its exact pre-S2 golden.

S2 scope boundaries
~~~~~~~~~~~~~~~~~~~

S2 wires the :math:`d \ge 2` UBLD kernel so 2-D LD *runs* on the DAG wavefront,
but it is deliberately scoped to the **average-moment iterate** only.  Three
things remain owed to the later sub-steps, and naming them here is the honest
boundary:

* The full ``loss_action`` / Krylov 2-D LD needs the spatial-moment iterate
  :math:`\hat\phi` to travel between sweeps so the scattering slope source
  :math:`\Sigma_s\hat\phi` couples the slopes globally — **S3**
  (:ref:`ld-ubld-unified-moment-matvec`).  Without it the S2 closure is
  :math:`O(h^2)` but diffusion-limit-inconsistent (the flat-source signature).

* The non-vanishing domain-inflow moment trace (the ``AngularBoundaryFlux`` /
  ``mesh.angular_trace`` widening to a :math:`2^{d-1}` transverse face moment) — **S4**
  (and its honest-scope caveat, :ref:`ld-cartesian-2d`).

* The strengthened vv Mode-7 stress-ansatz MMS and the thick-diffusive
  tripwire — **S4** and **S3** respectively.

The verification is the kernel round-trip + matvec-twin face reconstruction
(:mod:`tests.transport.spatial.test_linear_discontinuous` ``TestLDKernel``), the
end-to-end two-paths FFW :math:`\equiv` MFW, the DD :math:`\ne` LD routing-flip,
and the :math:`O(h^2)` convergence smoke
(:mod:`tests.sn.verification.mms.test_mms_ld_2d`), plus the :math:`d=2`
numpy↔symbolic entry-wise ``A == A`` cell-assembly pin and the
``test_d2_exact_on_bilinear`` ERR-060 catcher
(:mod:`tests.transport.spatial.test_ld_ubld_primitive`).

.. _two-moment-axes:

Two kinds of "moment": angular vs spatial
-------------------------------------------

.. warning::

   The word **moment** denotes two ORTHOGONAL things in this solver, and
   the collision is the single most common source of confusion when reading
   the multi-dimensional Linear-Discontinuous (LD) code.  An **angular
   moment** reduces the *direction* dependence; a **spatial moment** resolves
   the *within-cell position* dependence.  They are independent tensor
   factors of the flux, NOT two names for the same axis.

The discrete-ordinates flux is a function of three independent kinds of
variable: direction :math:`\Omega`, position :math:`\vec r` (which the mesh
splits into a cell index plus a *within-cell* coordinate), and energy
group :math:`g`.  Each admits its own moment expansion, and the LD scheme
in :math:`d \ge 2` carries two of them simultaneously.  Distinguishing them
is the prerequisite for reading the next two subsections.

**Angular moment** :math:`\phi_\ell^m` — *how the flux varies with
direction.*  Projecting the per-ordinate angular flux
:math:`\psi(\Omega_n)` onto the real spherical harmonics
:math:`\{Y_\ell^m\}` collapses the :math:`N` discrete directions into the
:math:`(\ell, m)` harmonic coefficients,

.. math::
   :label: two-moment-angular

   \phi_\ell^m(\vec r, g)
   \;=\;
   \sum_{n=1}^{N} w_n\, Y_\ell^m(\Omega_n)\, \psi_n(\vec r, g),

.. (vv-status rationale) Notation/definition of the angular moment
   projection phi_l^m (the flux-moments quadrature contraction); the SH
   frame's analysis face is separately verified.
.. vv-status: two-moment-angular documented

the typed home of which is
:class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
on the space
:math:`\mathrm{SphericalHarmonicSpace}(L) \otimes \mathrm{CellGroupSpace}`.
The angular moment is a **replacement representation** of the angular flux:
a calculation holds EITHER the per-ordinate field
:math:`\psi` of shape :math:`(N, ng, *\text{spatial})` OR its harmonic
moments :math:`\phi_\ell^m` of shape
:math:`(L{+}1, 2L{+}1, ng, *\text{spatial})`, bridged by the
spherical-harmonic :class:`~orpheus.numerics.frame.GalerkinFrame`'s two faces —
its ``analysis`` face (:math:`M`, the
:math:`\psi \to \phi` reduction :eq:`two-moment-angular`) and its
``reconstruction`` face
(:math:`R`, the :math:`\phi \to \psi` lift).  You never carry both;
windowing the 2-D Cartesian iterate as :math:`\phi_\ell^m` instead of
:math:`\psi` is the harmonic-moment-projection memory win
(the :math:`N \to (L{+}1)(2L{+}1)` collapse, :eq:`harmonic-moment-projection`).
The :math:`\ell = 0` moment IS the scalar flux exactly.

**Spatial moment** :math:`\hat\psi` — *how the flux varies in space inside
one cell.*  A finite-volume / Diamond-Difference closure carries a single
number per cell (the cell average :math:`\bar\psi`).  The
Linear-Discontinuous closure additionally resolves the **sub-cell slope**:
on a Cartesian cell it expands :math:`\psi` in the tensor product of a 1-D
Legendre basis :math:`\{1, P_1\}` per axis,

.. math::
   :label: two-moment-spatial

   \psi(x, y)\big|_{\rm cell}
   \;=\;
   \bar\psi\,
   + \hat\psi_x\, P_1(\xi_x)
   + \hat\psi_y\, P_1(\xi_y)
   + \hat\psi_{xy}\, P_1(\xi_x) P_1(\xi_y),
   \qquad \xi_a \in [-1, 1],

.. (vv-status rationale) Notation/definition of the within-cell bilinear
   (UBLD) spatial-moment expansion. Definitional.
.. vv-status: two-moment-spatial documented

the four within-cell coefficients of the bilinear (UBLD) basis
:math:`\{1, x, y, xy\}` of :eq:`ld-ubld-cell-system`.  Unlike the angular
moment, the spatial moment is an **additional axis** that rides on whatever
angular representation is in play — it does NOT replace anything.  Its typed
home is the discretization scheme's own
:meth:`moment_axis
<orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis>`
(:ref:`the account below <spatial-moment-space>`), a ``MODAL``
:class:`~orpheus.numerics.axis.Axis` of length
:math:`(\text{per\_axis})^d` carrying the scheme's own cell mass, which
composes onto a field's space alongside the cell/group/angular factors.

The two notions are summarised in the contrast table:

.. list-table:: The two "moment" axes — orthogonal tensor factors
   :header-rows: 1
   :widths: 18 26 28 28

   * - Property
     - Angular moment :math:`\phi_\ell^m`
     - Spatial moment :math:`\hat\psi`
     - Shared
   * - Resolves
     - direction :math:`\Omega`
     - within-cell position :math:`x`
     - —
   * - Basis
     - real spherical harmonics :math:`\{Y_\ell^m\}`
     - tensor-Legendre :math:`\{1, P_1\}` per axis
     - both orthogonal polynomial families
   * - Truncation knob
     - :math:`L` (max Legendre order)
     - ``per_axis`` (1 = DD/Step, 2 = LD)
     - —
   * - Count
     - :math:`(L{+}1)(2L{+}1)`
     - :math:`(\text{per\_axis})^d`
     - —
   * - Typed home
     - the head space's ``MODAL``
       :class:`~orpheus.numerics.axis.HarmonicAxis` (on a folded / 1-D
       rule, :class:`~orpheus.numerics.axis.LegendreAxis`)
     - the scheme's ``MODAL`` ``spatial_moment``
       :class:`~orpheus.numerics.axis.Axis`
     - both ``MODAL`` axes of one axis-built space (both were
       ``FunctionSpace`` subclasses until 2026-09-08 — items 6.2c-ii
       and 6.2c-iii)
   * - Role on the flux
     - **replacement** (hold :math:`\psi` OR :math:`\phi_\ell^m`)
     - **additional** (rides on either)
     - both tensor factors of one field
   * - Set by
     - the angular Pℓ order requested
     - the spatial discretization scheme
     - —

Because they are independent factors, a fully-resolved LD-Pℓ angular flux
carries BOTH indices simultaneously — an angular index :math:`(\ell, m)`
and a spatial-moment index :math:`p`:

.. math::
   :label: two-moment-tensor-product

   \phi_\ell^{m, p}(\vec r_{\rm cell}, g),
   \qquad
   (\ell, m) \in \text{angular harmonics}, \quad
   p \in \{\bar{\,}, \hat x, \hat y, \widehat{xy}\}
   \ \text{(spatial moments)}.

.. (vv-status rationale) Notation: the combined angular-tensor-spatial
   moment index phi_l^{m,p}. Definitional.
.. vv-status: two-moment-tensor-product documented

The carrier space is the tensor product of the two moment spaces with the
cell/group space,

.. math::
   :label: two-moment-carrier-space

   \mathrm{SphericalHarmonicSpace}(L)
   \;\otimes\;
   \mathrm{CellGroupSpace}(ng, *\text{spatial})
   \;\otimes\;
   \mathrm{SpatialMomentAxis}(\text{per\_axis}, d),

.. (vv-status rationale) Named-field-typing identity (the carrier-space
   tensor product). Foundation-gated by test_spatial_moment_field_space
   and, for the tail factor, test_spatial_moment_tail_is_the_schemes_axis.
.. vv-status: two-moment-carrier-space documented

.. note::

   **The first factor is named for this chapter's setting, not by
   construction.** A :math:`d \ge 2` Cartesian mesh carries a full-sphere
   angular rule, so the angular head genuinely *is* a
   :class:`~orpheus.numerics.spaces.SphericalHarmonicSpace`. In general
   the head is *the coefficient space of the basis the mesh's quadrature
   bound at* :math:`L`, READ off the frame
   (:eq:`moment-space-read-off-the-frame`,
   :ref:`frame-moment-space-single-home`) — the σ-even restriction on a
   folded rule, and the Legendre basis on :math:`S^2/O(2)_a` on a 1-D
   chart, which #429 tracker 3.4 landed on 2026-09-02. Since then the
   field factory
   reads it rather than minting it from :math:`L`, so the leading
   :math:`(L{+}1, 2L{+}1)` rectangle below is the *harmonic* head's
   layout and not a property of moment fields as such.

so the stored ndarray gains a trailing :math:`(\text{per\_axis})^d`
spatial-moment axis after the :math:`(\ell, m, g, *\text{spatial})` prefix.
The orthogonality is what makes the architecture clean: the scattering
operator :math:`\Sigma_s` couples energy groups and (for anisotropic
scattering) angular moments, but is a **spectator** to the spatial-moment
axis — it scatters every spatial moment independently
(:eq:`ld-ubld-scattering-moment-lift`, next subsection).  Conversely the
spatial discretization (the sweep / cell solve) acts on the spatial moments
but is a spectator to the angular index.  Two operators, two axes, no
cross-talk except through the physics each is responsible for.

.. note::

   **Why an LD-P3 calculation needs both.**  Anisotropic scattering up to
   :math:`P_3` is an *angular*-resolution choice — it carries
   :math:`\phi_\ell^m` for :math:`\ell \le 3`.  The Linear-Discontinuous
   spatial closure is a *spatial*-resolution choice — it carries
   :math:`\hat\psi` for the within-cell slope.  An LD-P3 transport
   calculation makes both choices at once and so carries the full
   :math:`\phi_\ell^{m, p}` object of :eq:`two-moment-tensor-product`.
   Collapsing either axis to its average (P0 angular, or DD spatial)
   degrades a *different* physical fidelity: the angular collapse loses the
   flux's directional anisotropy; the spatial collapse loses the
   diffusion-limit accuracy that the :math:`xy` cross-moment provides
   (:eq:`ld-ubld-cell-system`, the load-bearing moment).

.. _ld-ubld-scattering-moment-lift:

The Σ_s ⊗ I spatial-moment scattering lift (S3-A, partial)
-------------------------------------------------------------

Sub-step **D5b-S3** completes the *physics* of the multi-dimensional UBLD
Linear-Discontinuous scheme.  Now that the two moment axes are clearly
distinguished (:ref:`two-moment-axes`), the completion is statable in one
sentence: the scattering source must scatter EVERY spatial moment, not just
the cell average.  Where S2 ships an O(h²) but diffusion-limit-INCONSISTENT
closure (it scatters only the spatial-AVERAGE moment — the slope rows of the
source are zero), S3 threads the canonical slope source so the converged
operator becomes the diffusion-limit-CONSISTENT one.

The load-bearing bridge is the scattering operator's
:math:`\Sigma_s \otimes I_{\rm spatial}` lift: :math:`\Sigma_s` carries no
spatial-moment index (it is an energy-group :math:`\to` energy-group matrix
per Legendre order), so it is applied to EVERY spatial moment of the scalar
flux INDEPENDENTLY,

.. math::
   :label: ld-ubld-scattering-moment-lift

   \bigl(\Sigma_s \otimes I_{\rm spatial}\bigr)\,
   (\bar\phi,\ \hat\phi_x,\ \hat\phi_y,\ \hat\phi_{xy})
   \;=\;
   (\Sigma_s\,\bar\phi,\ \Sigma_s\,\hat\phi_x,\
    \Sigma_s\,\hat\phi_y,\ \Sigma_s\,\hat\phi_{xy}),

.. (vv-status rationale) Structural Sigma_s (x) I_spatial lift identity;
   the byte-identity-at-single-moment negative control is foundation-verified
   (rank-2-exact). The physics completion is verified downstream via the
   thick-diffusion tripwire (ld-ubld-slope-angular-reduction).
.. vv-status: ld-ubld-scattering-moment-lift documented

so the spatial-moment axis is a SPECTATOR to the energy-group matmul,
exactly as the cell axis is.  In code this is the per-material group
contraction with a trailing ``...`` spectator
(``"fg,fc...->gc..."``): at the single-moment closures (Diamond
Difference / Step, ``per_axis == 1``) the trailing axis is ABSENT and the
``...`` matches nothing, so the lift is BYTE-IDENTICAL to the pre-S3
scattering (the negative-control bit-identity, verified rank-2-exact).
At ``per_axis == 1`` :math:`S_{\rm full} \equiv S_{\rm flat}`; only an LD
multi-moment closure activates the slope rows.

.. admonition:: Status — S3-A is PARTIAL
   :class: caution

   The :math:`\Sigma_s \otimes I_{\rm spatial}` lift documented here is the
   LANDED half of S3-A.  The :math:`\hat\phi` spatial-moment **iterate
   carrier** that FILLS the slope rows the lift now accepts is OWED (it was
   blocked on the typed-field space widening — the
   :ref:`within-cell moment factor <spatial-moment-space>` subsection, the
   prerequisite that was minted
   next).  The lift therefore scatters a slope source that, in the production
   path, is still zero (no field carries :math:`\hat\phi` yet); the converged
   fixed point does not change UNTIL the iterate carrier lands.  This page
   marks what is wired (the lift) versus what is owed (the iterate, the
   cell-emit accumulation, the source seams) so a future reader knows the
   S3-A wiring is mid-flight, not complete.  (**Since completed** — the
   iterate carrier, the cell-emit accumulation, and both source seams
   landed with the unified moment matvec and the :math:`d{=}1` moment
   scan: :ref:`ld-ubld-unified-moment-matvec` and
   :ref:`ld-ubld-moment-scan` below.  This admonition is preserved as
   the campaign-time boundary record.)

Physics-completion, not an iteration-only change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The distinction matters for verification, so it is worth stating precisely.
Most changes to the iteration machinery — a Gauss-Seidel splitting, a σ\
:sub:`r`-removal, a synthetic accelerator (DSA), a preconditioner — MUST NOT
change the converged fixed point; they change only the *rate* at which the
iteration reaches it.  The correctness gate for such a change is
**FP-invariance**: the accelerated solve and the plain solve converge to the
same flux (`vv-principles` Mode 9).  (⚠ That presupposes a fixed *point*
exists.  When :math:`A` is singular the invariant is the solution **set**
and two correct splittings may return different members — the exception,
and the three checks that separate it from a genuine splitting bug, are at
:ref:`sn-loss-kernel-gauge`.)

The :math:`\Sigma_s \otimes I_{\rm spatial}` slope source is **not** that
kind of change.  S2 and S3 solve DIFFERENT operators — the displays below
compare two *representations of the scattering gain* on the LD moment
space, so they carry neither the boundary law :math:`B` nor the
:math:`(n,2n)` gain :math:`N_{2n}`; the shipped within-group member list
is :eq:`sn-within-group-with-n2n`, and :math:`N_{2n}` lifts to the
moment space exactly as :math:`S` does (it is the same
:class:`~orpheus.transport.operators.transfer.TransferOperator` binding
with yield :math:`y = 2`):

.. math::
   :label: ld-ubld-s2-s3-operators

   \text{S2:}\quad (L + C - S_{\rm flat})\,\psi = Q_{\rm ext},
   \qquad
   S_{\rm flat} = \Sigma_s \otimes e_0 e_0^{\mathsf T},
   \\[4pt]
   \text{S3:}\quad (L + C - S_{\rm full})\,\psi = Q_{\rm ext},
   \qquad
   S_{\rm full} = \Sigma_s \otimes I_{\rm spatial}.

.. (vv-status rationale) Definition of the S2 (S_flat) vs S3 (S_full)
   operators. The physics completion (which fixed point is correct) is
   verified by the thick-diffusion tripwire; the operators themselves are
   definitional.
.. vv-status: ld-ubld-s2-s3-operators documented

:math:`S_{\rm flat}` (the rank-1 projector :math:`e_0 e_0^{\mathsf T}` onto
the cell-average moment) scatters ONLY the spatial average — the slope rows
:math:`\hat\phi_x, \hat\phi_y, \hat\phi_{xy}` of the scattering source are
identically zero.  :math:`S_{\rm full}` (the identity on the spatial-moment
axis) scatters all of them.  The two operators have DIFFERENT spectra, hence
DIFFERENT fixed points.  The converged flux CHANGES — and that is the POINT:
the thick-diffusion-limit tripwire (the ``test_ld_thick_diffusive_limit``
xfail) flips xfail :math:`\to` pass *because* the limit becomes correct, not
because the iteration was accelerated.  S3 is therefore **NOT** verified
against the S2 fixed point; verifying it that way would be the Mode-9
mis-application (asserting FP-invariance of a change that legitimately moves
the FP).  The genuine Mode-9 invariant for S3 is the within-group analog:
source-iteration with a lagged moment iterate :math:`\equiv` direct / Krylov
solve of the **same** :math:`(L + C - S_{\rm full})` operator.

Why the slope rows are diffusion-limit-load-bearing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the optically-thick, scattering-dominated (:math:`c \to 1`,
:math:`\Sigma_t h \gg 1`) limit, the transport solution must collapse to the
diffusion solution.  Adams (2001) and Larsen, Morel & Miller (1987) showed
that a spatial discretization passes this asymptotic limit only if its
discrete diffusion limit is a valid (consistent and stable) diffusion
discretization.  For the bilinear UBLD cell the diffusion limit couples the
slope moments :math:`\hat\phi` — the leading-order asymptotic balance is a
relation between the cell-average and the slopes, not the average alone
(Border, Lewis & Adams 1992 give the 2-D asymptotic analysis explicitly).
If the *scattering source* feeds only the cell average (S2's
:math:`S_{\rm flat}`), the slope rows of the within-cell balance see no
scattering re-supply, the discrete diffusion limit is the WRONG diffusion
operator, and the thick-limit error does not vanish under refinement — the
xfail tripwire stays red.  Threading :math:`\Sigma_s \hat\phi` into the slope
rows (:math:`S_{\rm full}`) restores the correct discrete diffusion limit.
This is why the completion is *physics* (the converged answer becomes right
in a regime where it was wrong), not iteration bookkeeping.

.. note::

   This is the same asymptotic reasoning that selects the bilinear
   :math:`\{1, x, y, xy\}` basis over the simplex :math:`\{1, x, y\}` in the
   first place (:eq:`ld-ubld-cell-system` and the parent section): Adams
   (2001) proved the simplex-LD discrete diffusion limit is invalid on
   quadrilaterals.  The :math:`xy` cross-moment carries the limit; the
   :math:`\Sigma_s \otimes I_{\rm spatial}` lift makes sure that cross-moment
   (and the axis slopes) actually receive scattering.  The basis choice and
   the scattering lift are two halves of the *same* diffusion-limit argument.

The producer-side spectator-broadcast (Pattern 7)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lift is implemented as a one-character change to the einsum subscripts of
the three scattering producers in
:class:`~orpheus.transport.mesh.material_xs_field.MaterialXSField`:

.. list-table:: The :math:`\Sigma_s \otimes I_{\rm spatial}` lift in code
   :header-rows: 1
   :widths: 34 30 36

   * - Producer
     - Subscript (pre-S3 :math:`\to` S3)
     - What it scatters
   * - :meth:`~orpheus.transport.material_field.TransferMaterialField.add_p0_source`
       (né ``MaterialXSField.apply_p0_in_scatter`` until CS4c step 3's O-6 move)
     - ``"fg,fc->gc"`` :math:`\to` ``"fg,fc...->gc..."``
     - the P0 in-scatter :math:`\Sigma_{s,0}^{\mathsf T}\phi`
   * - the SAME
       :meth:`~orpheus.transport.material_field.TransferMaterialField.add_p0_source`
       over the :math:`(n,2n)` field (né ``apply_n2n``, then
       ``N2NMaterialField.add_emission`` until #426 step 2)
     - ``"fg,fc->gc"`` :math:`\to` ``"fg,fc...->gc..."``
     - the :math:`(n,2n)` :math:`P_0` source
       :math:`\nu_{2n}\Sigma_{2n,0}^{\mathsf T}\phi` — the SAME einsum
       with ``scale = y``; its :math:`\ell \ge 1` half rides the
       ``moment_source`` row below, which is what made the two verbs
       one
   * - :meth:`~orpheus.transport.material_field.TransferMaterialField.moment_source`
       (né ``apply_legendre_scattering_moments``)
     - ``"mfc,fg->mgc"`` :math:`\to` ``"mfc...,fg->mgc..."``
     - the per-:math:`\ell` block-diagonal :math:`\Lambda\phi`. ⚠ Since
       2026-09-02 the ``m`` in that spec is the **real-harmonic head's**
       axis, and the verb selects the spec by the head's RANK — a 1-D
       rule's FLAT head takes ``"fc...,fg->gc..."`` (:ref:`spaces-moment-head`).
       The harmonic spec is this row's, verbatim

The trailing ``...`` is the **spectator broadcast**: it matches the
spatial-moment axis (if present) and contracts nothing over it — exactly the
:math:`\otimes I_{\rm spatial}` of :eq:`ld-ubld-scattering-moment-lift`.  This
is the `coding-elegance` Pattern-7 producer-side lift: the convention
(scatter each spatial moment independently) is normalised at the producer, so
no consumer special-cases the axis.  Two properties follow by construction:

* **Byte-identical at the single-moment closures.**  When ``phi`` is rank-2
  (``(ng, n_cells)``, the DD/Step shape) the trailing axis is ABSENT, ``...``
  matches nothing, and ``"fg,fc...->gc..."`` is the SAME contraction as
  ``"fg,fc->gc"`` — verified rank-2-exact
  (``np.array_equal`` of the two einsums when no trailing axis is present).
  No re-baseline of the DD/Step path; this is the negative-control
  bit-identity.

* **The projection pair needed no change.**  The spherical-harmonic
  :class:`~orpheus.numerics.frame.GalerkinFrame`'s ``analysis`` and
  ``reconstruction`` faces already carry ``...`` for their trailing
  axes, so :math:`M` and :math:`R` are spatial-moment-agnostic out of
  the box.  The angular reduction
  :eq:`two-moment-angular` and its inverse ride a spatial-moment axis as a
  spectator, which is the architectural payoff of the orthogonal-factor
  framing — the two moment axes never need to know about each other.

.. warning::

   The crosswalk and the original brief ASSUMED the P0 arm
   already broadcast over a trailing axis.  It did NOT: the bare ``"fg,fc->gc"``
   hard-codes the cell axis as a single index ``c``, so a rank-3
   ``phi (ng, n_cells, 2^d)`` RAISES (``operand has more dimensions than
   subscripts``).  The fix is the explicit ``...`` spectator, not a reshape.
   Documented so a future reader does not re-derive the (false) assumption.

What is still owed (the iterate carrier and the source seams)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lift accepts a slope source, but nothing in the production path yet
PRODUCES one.  Filling the slope rows requires (all S3-A proper, owed):

* The :math:`\hat\phi` **iterate carrier** — the between-sweep flux must
  carry the spatial-moment axis.  This was the make-or-break design decision:
  the typed-field spaces validate ``shape == (ng, *spatial)`` with no slot for
  a trailing :math:`(\text{per\_axis})^d` axis, so a slope-carrying field was
  an *illegal state* (Pattern 4 firing correctly).  The resolution — minting
  the :ref:`first-class within-cell moment factor <spatial-moment-space>` —
  is the subject of the
  next subsection.

* The **cell-emit moment accumulation** — the wavefront cell solve already
  computes a :math:`(\text{per\_axis})^d`-moment ``psi_avg``, but the
  between-sweep emit currently drops to slot 0 (the cell average); it must
  accumulate the full moment vector.

* The **two source seams** — the :math:`d \ge 2` wavefront genuine
  :math:`(2^d, ng)` moment source through the dense ``_ubld_system``, and the
  :math:`d = 1` scan slope source threaded via
  :meth:`~orpheus.transport.spatial._ubld.D1ClosedForm.kernel_rhs` and the Schur
  ``schur_xV`` term.

The verification chain for the completed S3 is the thick-diffusion-limit
VALUE anchor (the continuous diffusion solution at :math:`\varepsilon \to 0`,
structurally independent of the LD kernel — Adams 2001 / Border-Lewis-Adams
1992 / Larsen-Morel-Miller 1987), the convergence-order MMS smoke (the slope
source exercised), and the genuine Mode-9 SI :math:`\equiv` Krylov on
:math:`(L + C - S_{\rm full})`.  The design record is
``.claude/plans/issue_240_d5b_s3_crosswalk.md``; the verification spec is
``.claude/agent-memory/test-architect/d5b_s3_inc_c_moment_iterate_verification.md``;
the lift's landed-half closeout is
``.claude/agent-memory/method-implementer/issue_240_d5b_s3_a_inc_c_closeout.md``.

All three owed items have SINCE LANDED in the subsections that follow:
the :math:`\hat\phi` iterate carrier and the cell-emit moment
accumulation with the unified moment matvec
(:ref:`ld-ubld-unified-moment-matvec`), and the two source seams with
that matvec's :math:`d \ge 2` dense path and the :math:`d{=}1` moment
scan (:ref:`ld-ubld-moment-scan`).  The owed-list above is preserved as
the campaign-time boundary record.

.. _spatial-moment-space:

The within-cell spatial-moment factor: a first-class DG moment carrier (S3-A0)
------------------------------------------------------------------------------

.. important::

   **This question has been answered twice, and only the REALIZATION
   changed.** The design ruling — *the within-cell moment is a
   first-class typed factor, never a bare trailing integer axis* — was
   taken at **#240 D5b-S3-A0** and has never been overturned; the
   argument for it is preserved verbatim below and is still the reason
   the factor is typed. What changed is what carries the type:

   * **#240 D5b-S3-A0** (the account below, in its original tense) minted
     the factor as its own :class:`~orpheus.numerics.space.FunctionSpace`
     subclass, ``SpatialMomentSpace``, composed on with the tensor
     product ``*`` and recovered by ``find_factor``.
   * **CS4c step 6 item 6.2c-iii** (2026-09-08) retired that class. The
     factor is now the discretization scheme's OWN ``MODAL``
     :class:`~orpheus.numerics.axis.Axis`, minted by
     :meth:`DiscretizationSchemeBase.moment_axis
     <orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis>`
     — label ``"spatial_moment"``, shape :math:`(2^d,)`, and
     ``weights`` the scheme's own :meth:`moment_mass_diagonal
     <orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_mass_diagonal>`
     — composed on by :meth:`BulkField.compose_spatial_moments
     <orpheus.transport.fields._bases.BulkField.compose_spatial_moments>`.

   ⛔ **Why the class had to go: it was a SECOND spelling of a factor
   that already existed.** Since CS4b the widened *angular* and *scalar*
   spaces carried the scheme's mass-weighted axis, while the widened
   harmonic-*moment* product appended the Euclidean class beside it —
   one factor, two spellings, with four consequences the corpus records:
   a widened moment product was **axes-less** (an axes-less factor takes
   ``*``'s non-axis arm), its tail carried **no mass** (so the moment
   field's norm was not its energy on that factor), the frame's
   derivation had to **drop** the angular space's own tail axis and
   re-append the class to stay ``(name, shape)``-equal to the hub's, and
   the widened :meth:`HarmonicMomentFlux.scalar_flux
   <orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux.scalar_flux>`
   self-derive was **refused by contract** because the product's own
   cell-group factor lacked the axis the target needed. All four are
   closed; the closing evidence is the changelog entry in
   :ref:`sn-development-history` and the "What the retirement moved"
   note at the end of this section, with the space-layer and frame-layer
   statements at :ref:`spaces-moment-head-axis-built` and
   :ref:`frame-the-one-moment-space`.

   **What is unchanged and still normative:** the Kronecker ordering and
   the slot-0 cell-average convention, the ``(per_axis)^d`` count law,
   and the "append iff > 1" byte-identity policy. That doctrine was
   never the class's — it lives in
   :mod:`orpheus.numerics.moment_layout`, and it did not move.

The typed-field-space half of S3-A (the half the scattering-lift TODO above
flagged as a hard prerequisite).  The within-cell tensor-Legendre DG moment
axis — how :math:`\psi` varies in space WITHIN a cell — is a
first-class typed factor, the **spatial** sibling of the **angular**
head (:class:`~orpheus.numerics.spaces.spherical_harmonic_space.SphericalHarmonicSpace`
on a full-sphere rule).
The two "moment" notions are ORTHOGONAL axes (angular harmonics over
direction :math:`\Omega` vs spatial Legendre over within-cell position
:math:`x`); naming each as its own typed factor keeps the distinction
type-visible and dispels the collision.  That argument was written for
the S3-A0 mint, when both factors were ``FunctionSpace`` subclasses;
since 2026-09-08 (items 6.2c-ii and 6.2c-iii) both are ``MODAL`` axes of
one axis-built space, which makes the sentence more literally true, not
less — they are two axes, side by side, of the same product.

.. math::
   :label: spatial-moment-space-size

   \dim(\text{SpatialMomentAxis}) \;=\; (\text{per\_axis})^{d},
   \qquad
   \text{per\_axis} =
   \begin{cases}
     1 & \text{DD / Step (cell-average } \{1\}\text{)} \\
     2 & \text{LD (linear } \{1, P_1\}\text{)}
   \end{cases}

.. (vv-status rationale) Named-field-typing dimension identity
   dim = per_axis^d. Foundation-gated by test_spatial_moment_field_space
   and test_spatial_moment_tail_is_the_schemes_axis (the tail axis's shape
   against cell_moment_count).
.. vv-status: spatial-moment-space-size documented

The factor composes into the bulk-field spaces EXACTLY as the angular
factor does.

.. note::

   **How the width is recovered — then and now.** At S3-A0 the factor
   composed with the tensor product ``*`` and was recovered by TYPE,
   ``space.find_factor(SpatialMomentSpace).per_axis``, which is what
   closed #207 (see the note below). Since item 6.2c-iii it is an
   :class:`~orpheus.numerics.axis.Axis` of an axis-built space and is
   recovered by LABEL —
   :data:`~orpheus.numerics.moment_layout.SPATIAL_MOMENT_AXIS_LABEL` —
   which is what
   :meth:`BulkField.spatial_moments_per_axis_of
   <orpheus.transport.fields._bases.BulkField.spatial_moments_per_axis_of>`
   reads off the space.
   :meth:`~orpheus.numerics.space.TensorProductSpace.find_factor` itself
   stays, still minted by this step, still the typed bridge from a
   composed space back to a factor's metadata; its surviving queries are
   the harmonic head's :math:`L` and the Legendre head's spent axis.
   Both spellings answer the same question — *what is the moment width?*
   — off the SPACE, never off a threaded integer, which is the ruling
   this section exists to record.

The
field-space factories of the day (``AngularField.from_mesh``,
``ScalarField.from_mesh``, and
:meth:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux.from_mesh_and_L`
— the two mesh-keyed leaf factories retired at CS4b S5, the keyed moment
factory did not)
gained an OPTIONAL ``spatial_moments`` parameter (default ``1``) that
appends the factor **iff the within-cell count exceeds 1** — the
"append iff > 1" gate single-sourced from
:func:`~orpheus.numerics.moment_layout.spatial_moment_tail`
(the cell analogue that delegates to
:func:`orpheus.numerics.moment_layout.face_moment_tail`, so the cell-moment tail
and the per-face cochain tail can never disagree).  At the default the
field space is BYTE-IDENTICAL to its pre-S3 shape for EVERY scheme (DD,
Step, AND LD): this step builds the CAPABILITY only (construct-general /
select-narrow), and no production field selects the axis yet.

Why a first-class typed factor, not a bare int axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The slope-carrying flux could, in principle, be stored as a plain ndarray
with a trailing :math:`(\text{per\_axis})^d` axis and an integer remembered
somewhere for its width.  That was rejected (the user's design choice
"option b") in favour of a first-class typed factor — the
``SpatialMomentSpace`` class at S3-A0, the scheme's own ``MODAL``
:class:`~orpheus.numerics.axis.Axis` since item 6.2c-iii — for the
`coding-elegance` Pattern-4 reason
(*make illegal states unrepresentable*).

The typed-field layer validates ``values.shape == space.shape`` at
construction (:meth:`Field.__post_init__`).  Before this step the SN field
spaces were rigidly ``(ng, *spatial)`` / ``(N, ng, *spatial)`` /
``(L+1, 2L+1, ng, *spatial)`` — there was **no slot** for a trailing
spatial-moment axis, so a :math:`\hat\phi`-carrying field FAILED the gate.
A trailing slope axis was, literally, an illegal state (the gate was firing
*correctly* — see the scattering-lift status admonition above, which flagged
this as the make-or-break prerequisite).  Two ways to make the slope-carrying
field legal:

.. list-table:: Bare-int axis vs first-class typed factor
   :header-rows: 1
   :widths: 22 39 39

   * - Aspect
     - Bare ``int`` trailing axis
     - First-class typed factor
   * - Field validity
     - widen the shape gate to accept *any* trailing axis (loses the
       illegal-state guard)
     - the space DECLARES the axis; the gate stays exact — a slope field
       is now a *legal, declared* shape
   * - Querying the width
     - thread an ``int`` parameter through every call site, or re-derive
       it from a raw ``.shape[-1]``
     - ask the SPACE: ``find_factor(SpatialMomentSpace).per_axis`` at
       S3-A0 (query by TYPE, position-independent — #207), the axis's
       own labelled shape since item 6.2c-iii
   * - Self-description
     - the axis is anonymous; reading code cannot tell a spatial-moment
       axis from any other trailing axis
     - the factor's type IS its documentation; the
       angular/spatial collision is dispelled at the type level
   * - Precedent
     - none — a one-off convention
     - the EXACT mold of the angular
       :class:`SphericalHarmonicSpace` factor (one architecture, two axes)

.. note::

   **The table's right-hand column is the RULING, and it reads the same
   at both realizations.** Row by row, at item 6.2c-iii (2026-09-08):
   *field validity* — the space still DECLARES the axis, now as an
   :class:`~orpheus.numerics.axis.Axis` of an axis-built space, and the
   shape gate is still exact; *querying the width* — still asked of the
   space, by label instead of by type; *self-description* — the axis's
   ``label`` and ``MODAL``
   :class:`~orpheus.numerics.axis.BasisKind` carry what the class's type
   used to; *precedent* — the mold is unchanged and now literal, because
   items 6.2c-ii and 6.2c-iii made the angular head an axis too. The one
   thing the axis adds, which the class never had, is the MEASURE: the
   axis carries the scheme's own cell mass
   (:meth:`moment_mass_diagonal
   <orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_mass_diagonal>`),
   so the moment field's norm is its energy on the tail as well as on
   the head. That is why the class could not simply be renamed.

The typed factor is the same pattern the angular moment already uses: the
harmonic factor is a :class:`SphericalHarmonicSpace` whose ``L`` is recovered
by ``space.find_factor(SphericalHarmonicSpace).L``, NOT a bare integer
threaded through the API.  Minting the spatial sibling keeps the two axes
*symmetric* — the orthogonal-factor framing of :ref:`two-moment-axes` is then
literally how the carrier space is built (:eq:`two-moment-carrier-space`).

.. note::

   Closing #207 as a side effect.  The
   ``space.find_factor(SphericalHarmonicSpace).L`` query was DOCUMENTED in the
   :class:`HarmonicMomentFlux` docstrings (issue #207) but had never been
   IMPLEMENTED — three docstrings referenced a method that did not exist.  The
   spatial-moment work needed exactly this composition-tree query, so
   :meth:`~orpheus.numerics.space.TensorProductSpace.find_factor` was minted
   now: it returns the first tensor factor that ``isinstance(factor, T)`` and
   raises :exc:`KeyError` if absent (a structural assertion — the caller
   believes the composed space carries the factor — not a silent ``None``,
   Pattern 4).  Both moment factors (angular and spatial) were then
   queryable by type, and the latent broken claim in the docstrings was
   made true.  ⛔ Half of that is now history: item 6.2c-iii retired the
   spatial factor into an axis, so the SPATIAL width is read by label
   rather than by ``find_factor``.  The method itself is untouched and
   still live — the angular head is still a tensor factor, and
   ``find_factor(SphericalHarmonicSpace).L`` still answers (`[M]`
   2026-09-08, a widened 2-D LD moment product's factors are
   ``[SphericalHarmonicSpace, FunctionSpace]``).

The Kronecker moment ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The within-cell basis is the tensor (Kronecker) product of the per-axis 1-D
Legendre basis, ordered **x-outer / y-inner** (matching the UBLD assembler
:func:`orpheus.transport.spatial._ubld.assemble_ubld`).  The convention is fixed so
that the all-:math:`P_0` cell average is ALWAYS slot 0:

.. math::
   :label: spatial-moment-kronecker-order

   d{=}1:&\quad [\,\bar\psi,\ \hat\psi_x\,]
   \\[2pt]
   d{=}2:&\quad [\,\bar\psi,\ \hat\psi_y,\ \hat\psi_x,\ \hat\psi_{xy}\,]
   \\[2pt]
   d{=}3:&\quad [\,\bar\psi,\ \hat\psi_z,\ \hat\psi_y,\ \hat\psi_{yz},\
                  \hat\psi_x,\ \hat\psi_{xz},\ \hat\psi_{xy},\ \hat\psi_{xyz}\,]

.. (vv-status rationale) Notation/convention (the x-outer / y-inner
   Kronecker layout, slot-0 = cell average). Foundation-gated by the
   moment-ordering primitive tests.
.. vv-status: spatial-moment-kronecker-order documented

The slot-0 (cell-average) convention is single-sourced from
:data:`orpheus.numerics.moment_layout.AVERAGE_MOMENT` (the constant every moment
consumer reduces on) rather than re-spelling the literal ``0``, so a
layout change happens in ONE place and not at the scattered ``[..., 0]``
call sites.

.. note::

   At S3-A0 the ``SpatialMomentSpace`` class re-surfaced that constant as
   an ``average_moment_index`` accessor.  Item 6.2c-iii retired the class
   and the accessor with it; every consumer imports the constant
   directly.  `[M]` 2026-09-08, by AST over ``orpheus/``: **5** modules
   import :data:`~orpheus.numerics.moment_layout.AVERAGE_MOMENT` and
   read it at **10** sites (``sn/loss_representation`` 5, ``sn/solver``
   2, and one each in ``transport/spatial/linear_discontinuous``,
   ``transport/source_sinks/angular_boundary_source_sink`` and
   ``derivations/continuous/mms/sn``), and ``average_moment_index``
   appears **nowhere** in the tree.  The convention did not move — only
   the spelling of who hands it out.

Slot 0 is the link
between the two scales: the cell-average moment :math:`\bar\psi` is what the
DD/Step closure carries in full, and it is the moment the
:math:`\Sigma_s \otimes I_{\rm spatial}` lift scatters at every closure; the
remaining slots are the LD-only slope rows the lift activates.

The "append iff > 1" byte-identity policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The backward-compatibility invariant (#240 D5b) is that DD/Step field shapes
must be UNCHANGED.  This is enforced by a single policy, single-sourced so the
cell-moment tail and the per-face cochain tail can never drift apart:

.. math::
   :label: spatial-moment-append-policy

   \texttt{tail}(n) =
   \begin{cases}
     ()    & n = 1 \quad\text{(DD/Step — NO length-1 axis appended)} \\
     (n,)  & n > 1 \quad\text{(LD — a genuine trailing moment axis)}
   \end{cases}

.. (vv-status rationale) Named-field-typing byte-identity policy
   (append-iff->1). Foundation-gated by the byte-identity-at-default
   negative control (test_spatial_moment_field_space).
.. vv-status: spatial-moment-append-policy documented

The critical detail is that ``n == 1`` returns the EMPTY tuple, NOT
``(1,)`` — a length-1 axis is NOT appended.  Appending ``(1,)`` would
broadcast-equal the old shape numerically but would change ``ndarray.shape``
and ``ndim``, breaking every byte-identity gate and every consumer that reads
``.ndim``.  The empty-tuple branch keeps the DD/Step field space *literally
identical* to its pre-S3 self.  :func:`orpheus.numerics.moment_layout.face_moment_tail`
owns the policy; the cell analogue
:func:`~orpheus.numerics.moment_layout.spatial_moment_tail`
delegates to it (Pattern 7 — normalise the convention at one site), and
:meth:`BulkField.compose_spatial_moments
<orpheus.transport.fields._bases.BulkField.compose_spatial_moments>`
returns the space UNCHANGED when the
tail is ``()``.  The cell analogue lived beside the ``SpatialMomentSpace``
class until item 6.2c-iii and is now homed in
:mod:`orpheus.numerics.moment_layout` beside the face policy it delegates
to — the two halves of one "append iff > 1" rule in one module, which is
where the policy always belonged.

Construct-general, select-narrow — what this step did and did NOT do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step built the **capability** to carry the spatial-moment axis and
nothing more (the two bullets below are dated to it; the note at the end of
the subsection says where each one stands today).  The discipline is
deliberate (`coding-elegance` — construct general, select narrow,
specialize only on measured need):

* **The axis exists.**  The factor is minted, composes into every
  bulk-field space, and the space answers for its width (by
  ``find_factor`` at S3-A0; by the axis's label since item 6.2c-iii).

* **No production field selects it** *(as of S3)*.  The ``spatial_moments``
  factory parameter defaulted to ``1`` at EVERY call site and was NOT
  auto-read from
  ``mesh.scheme.spatial_basis_per_axis``.  So DD, Step, AND LD field shapes
  were unchanged at this step — not even LD carried the slope axis yet.

* **Why default-OFF even for LD.**  Auto-reading the scheme would silently
  widen LD field shapes BEFORE the consumers that FILL the axis exist — the
  iterate carrier, the cell-emit accumulation, the source seams (all S3-A
  proper, owed; see the scattering-lift subsection).  A widened axis that no
  producer fills is precisely the illegal state Pattern 4 forbids; turning the
  capability on before its producers exist would re-introduce it.  The gate
  had teeth on exactly this mistake: making
  :meth:`BulkField.compose_spatial_moments
  <orpheus.transport.fields._bases.BulkField.compose_spatial_moments>`
  auto-read the scheme turned the
  LD byte-identity foundation tests RED (mutation-verified).

The S3-A iterate / cell-emit / source seams that thread the scheme's
``spatial_basis_per_axis`` here (selecting the axis for LD) were the NEXT
sub-step, and they landed: the validator already accepted the widened
space and the scattering lift already scattered its slopes, so the change
at each call site was exactly the promised
``spatial_moments=scheme.spatial_basis_per_axis``.

.. note:: **Where the selection lives now (CS4b S5, 2026-08-24).**  Both
   bullets above are dated to the S3 step; on the angular and scalar leaf
   families neither describes the tree any more, and the mechanism the
   third one guards against has been replaced rather than switched on.
   ``[M]``
   ``hasattr(AngularField, "from_mesh") is False`` — the mesh-keyed
   factory tier retired, and with it the ``spatial_moments=`` integer.
   The widening knob is now a **property choice** on the carrier:
   :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_trial_space` is
   :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_bulk_space` with
   the scheme's own MODAL
   :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis`
   appended, and a call site widens by *reading that property instead of
   the other one*.  The construct-general / select-narrow discipline this
   subsection states is unchanged — only its spelling is: an integer that
   every call site had to re-read off the scheme and thread correctly
   became a choice between two cached properties, which cannot be
   threaded wrong.  For a slopeless closure the two are the SAME cached
   instance, so DD / Step remain byte-identical by construction rather
   than by a default.  ``[M]`` on a 4-cell slab, ``N = 4``, ``ng = 2``:
   DD reads ``(4, 2, 4)`` from both properties (``is``-identical); LD
   reads ``(4, 2, 4)`` from the bulk mint and ``(4, 2, 4, 2)`` from the
   trial mint.  The keyed moment-family factories
   (:meth:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux.zeros_for_mesh_and_L`
   and its ``from_mesh_and_L`` sibling) still take ``spatial_moments=``
   and are re-homed at S6.  See
   :ref:`theory-sn-typed-fields` for the allocator surface in full.

Verification (foundation-level)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The space and the factory widening are software invariants, so they are
verified at the **foundation** level (data-structure / factory-output
invariants, not an L0/L1/L2 solver claim — they carry no eigenvalue or flux
assertion), in two test modules:

* ``tests/numerics/test_spatial_moment_field_space.py`` — the factory
  widening: the **byte-identity-at-default negative control** for DD AND LD on
  all three carriers (:class:`AngularField`, :class:`ScalarField`,
  :class:`HarmonicMomentFlux`), the widened :math:`d{=}1` / :math:`d{=}2`
  shapes, the both-moment-factors-coexist case, and the wrong-shape rejection.
  The mutation check — auto-reading the scheme turns the LD byte-identity
  cases red — is what proves the construct-general gate has teeth.

* ``tests/numerics/test_spatial_moment_tail_is_the_schemes_axis.py`` — the
  space layer, since item 6.2c-iii: that the widened moment product is
  axis-built with the scheme's own ``moment_axis`` as its tail (the axis
  compared to :meth:`DiscretizationSchemeBase.moment_axis
  <orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis>`
  and its weights to
  :meth:`moment_mass_diagonal
  <orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_mass_diagonal>`),
  that the hub and the frame agree at width 2 exactly as at width 1
  (ruling O-5), that the tail's measure MOVES the pairing — Parseval
  across the tail, with a Euclidean-tail negative control that must
  differ — that width 1 appends nothing and a DD carrier REFUSES a
  widened request, that a widened field self-derives its scalar flux and
  truncates onto the hub's space, and that the retired class is
  unspellable.

.. note::

   **What the retirement moved.**  The S3-A0 account above named a third
   module, ``tests/numerics/test_spatial_moment_space.py``, which pinned
   the ``SpatialMomentSpace`` class: the
   :math:`(\text{per\_axis})^d` size law, the
   :meth:`~orpheus.numerics.space.TensorProductSpace.find_factor`
   round-trip, the composition shape, the ``per_axis == 1`` no-widening
   case, and ``average_moment_index`` :math:`==`
   :data:`~orpheus.numerics.moment_layout.AVERAGE_MOMENT`.  Item
   6.2c-iii retired the class, so that module retired with it and its
   surviving claims were re-keyed onto the successors: the size law and
   the no-widening case are asserted against
   :func:`~orpheus.numerics.moment_layout.cell_moment_count` and
   :func:`~orpheus.numerics.moment_layout.spatial_moment_tail` in the
   new module, the composition shape lives in both field-space modules,
   and the slot-0 identity is now the constant itself, imported
   directly by every consumer.  The head's ``find_factor`` round-trip
   survives in ``tests/numerics/test_moment_head_axis_built_premise.py``.

The design record (the angular-vs-spatial distinction, the FP resolution) is
``.claude/plans/issue_240_d5b_s3_crosswalk.md``; the closeout is
``.claude/agent-memory/method-implementer/issue_240_d5b_s3_a0_spatial_moment_space_closeout.md``.

.. _ld-ubld-unified-moment-matvec:

The unified moment matvec: a forward apply is intrinsically moment-valued (S3)
----------------------------------------------------------------------------------

Sub-step **D5b-S3** completes the apply direction: applying the per-cell
:math:`2^d \times 2^d` UBLD operator (:eq:`ld-ubld-cell-system`) to a moment
vector is intrinsically moment-valued, so the matvec carries the full
:math:`(\bar\psi, \hat\psi)` moment vector in every dimension.  The
architectural payoff is a branch removal; the *physics* payoff is the recovery
of the thick-cell diffusion limit, which hinges on a single
frame-consistency identity (ERR-061) that the rest of this subsection derives.
The source files are :mod:`orpheus.transport.spatial.linear_discontinuous`
(``cell_kernel_batch`` / ``residual_kernel_batch`` — now ONE :math:`d`-generic
moment path), :mod:`orpheus.sn.loss_representation.sweep_graph` (``_CellSolve`` / ``_CellResidual``
— the ``len(s_axes) > 1`` moment gate retired), and
:mod:`orpheus.sn.loss_representation` (the ``_spatial_moment_tail`` buffer
widening); the closeout is
``.claude/agent-memory/method-implementer/issue_240_d5b_s3_unified_matvec_closeout.md``.

The earlier increments (A/B) made the :math:`d{=}1` LD **matvec** Schur-reduce
to a scalar residual: the slope :math:`\hat\psi` was eliminated, leaving a
scalar cell-average unknown.  That was a *flat-source artifact* — with
:math:`\hat Q = 0` the slope had no global coupling, so the Krylov unknown could
stay scalar.  Increment C makes the scattering slope source
:math:`\Sigma_s\hat\phi` couple the slope GLOBALLY (the diffusion-limit-
consistent operator :eq:`ld-ubld-scattering-moment-lift`), so the slope becomes a
genuine global degree of freedom in **every** dimension.

.. math::
   :label: ld-ubld-unified-moment-residual

   (L+C)\,\vec\psi
   \;=\;
   M^{-1}\bigl(A_{\rm cell}\,\vec\psi - F_{\rm in}\bigr),
   \qquad
   (L+C-S)\,\vec\psi = \vec q_{\rm ext}\ \Longleftrightarrow\
   (L+C)\,\vec\psi - S\,\vec\psi = \vec q_{\rm ext}

.. (vv-status rationale) Governing operator-algebra residual identity (the
   M^{-1} matvec/sweep moment-source consistency). Verified by the
   matvec == sweep round-trip (foundation), not an isolated solver claim.
.. vv-status: ld-ubld-unified-moment-residual documented

Here :math:`S` is this section's moment-lifted in-scatter gain
:math:`S_{\rm full}` and the display is bulk-only — it carries neither
:math:`B` nor the :math:`(n,2n)` gain :math:`N_{2n}`.  The identity is
linear in the gain, so it holds verbatim with every member of the
shipped list :eq:`sn-within-group-with-n2n` substituted for :math:`S`.

A matvec is a forward APPLY: applying the per-cell
:math:`2^d \times 2^d` UBLD operator to the moment vector is intrinsically
moment-valued, so ``cell_kernel_batch`` and ``residual_kernel_batch`` collapse to
ONE d-generic dense path for every :math:`d` (the former :math:`d{=}1`
Schur-reduced scalar arm — and the :math:`d \ge 2` raise — are both retired).
The :math:`M^{-1}` factor in :eq:`ld-ubld-unified-moment-residual` is the
matvec/sweep moment-source consistency: the UBLD RHS folds the cell source
mass-weighted (:math:`R = M\vec S`, the test-function projection), but the
operator algebra :math:`(L+C) - S` subtracts :math:`S\vec\psi` RAW at the
``OperatorSum`` level, so the residual is divided by the diagonal Legendre mass
to put :math:`(L+C)\vec\psi` in raw per-moment units (the slope rows would
otherwise disagree by :math:`M_{ii} = \theta^{|i|}`).

The architectural headline is **branch removal** (Cardinal Rule 2): the
``len(s_axes) > 1`` moment gate at the cell-solve / cell-residual emit is GONE
(replaced by the pure scheme trait ``spatial_basis_per_axis > 1``), the
:math:`d{=}1` scalar kernel twin is retired, and there are ZERO
``isinstance(scheme, ...)`` branches — dispatch stays via the scheme PROTOCOL +
geometry-keyed ``supports()``.

.. _ld-ubld-sweep-global-frame:

The sweep-frame / global-frame involution (ERR-061 — the diffusion-limit root cause)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the load-bearing result of the whole multi-dimensional LD campaign, and
it is the kind of "what failed and why" Cardinal Rule 3 demands be archived in
full.  Threading the moment matvec made the operator *internally consistent*
(matvec :math:`\equiv` sweep round-trip to :math:`10^{-16}`, source-iteration
:math:`\equiv` Krylov on the SAME operator) — and yet the converged flux was
**wrong**: on a thick scattering slab (:math:`\Sigma_t = 40`, :math:`c = 0.99`,
:math:`\Sigma_t h = 10`/cell, vacuum) at :math:`n_x = 4` the LD scalar flux was
1.47 against the diffusion solution 2.31 (relative error 36 %), and the error
did not grow under refinement — it *shrank* as the cells thinned (the classic
flat-source-LD signature, persisting THROUGH the slope-source thread).

The cause is a frame-consistency error between two individually-correct
components.  The per-cell LD kernel produces and consumes the :math:`2^d` moment
vector in the per-ordinate **sweep frame**: each axis :math:`a` is oriented so
the *downstream* face is at the local coordinate :math:`+1`.  For an ordinate
sweeping in the NEGATIVE global direction on axis :math:`a`
(:math:`\mathrm{octant\_sign}_a = -1`) the sweep coordinate is the *reverse* of
the global coordinate, so the slope (:math:`P_1`) moment on that axis is
sign-FLIPPED relative to the global-:math:`x` slope.  But the iterate
:math:`\hat\phi` and its scattering source :math:`\Sigma_s\hat\phi` live in the
**global frame** — the angular reduction sums slopes across ordinates of BOTH
sweep directions,

.. math::
   :label: ld-ubld-slope-angular-reduction

   \hat\phi(\vec r, g) \;=\; \sum_{n=1}^{N} w_n\,\hat\psi_n(\vec r, g).

The producer (``_CellSolve.cell`` emit) stored the raw sweep-frame slope; the
consumer (``integrate_angular`` / the scattering apply) summed it as if it were
global-frame.  So the backward ordinates' opposite-signed slopes partially
CANCELLED the forward ones: at a cell with a positive global-:math:`x` gradient
the forward ordinate had :math:`\hat\psi_n = +0.048` but the backward had
:math:`-0.028` — opposite signs, the smoking gun.  The summed
:math:`\hat\phi` was :math:`\sim 6\times` too small to satisfy the LM-1989
discrete diffusion continuity (Larsen & Morel 1989, JCP 83(1):212–236, Eq. 4.9b,
:math:`\bar\phi_j + \hat\phi_j = \bar\phi_{j+1} - \hat\phi_{j+1}`), the slope
source was under-driven, and the discrete diffusion limit was the wrong
diffusion operator.

The fix is a single-sourced :math:`2^d` moment-frame **involution**,
:func:`~orpheus.transport.spatial._ubld.octant_moment_frame_signs`,

.. math::
   :label: ld-ubld-octant-moment-frame-signs

   \mathrm{sign}[o_0, \ldots, o_{d-1}]
   \;=\;
   \prod_{a=0}^{d-1} (\mathrm{octant\_sign}_a)^{\,o_a},
   \qquad o_a \in \{0, 1\},

indexed in the tensor-Legendre Kronecker layout (:math:`o_a` = the :math:`P_0` /
:math:`P_1` selector on axis :math:`a`).  The **average** moment (all
:math:`o_a = 0`) is sign-invariant (the empty product is :math:`1`); a per-axis
**slope** flips once if that axis sweeps backward; the 2-D **cross** moment
:math:`\hat\psi_{xy}` flips when an ODD number of its active axes reverse.  The
map is its own inverse, so the SAME sign vector converts global :math:`\to`
sweep on the source/probe INPUT and sweep :math:`\to` global on the
moment/residual OUTPUT.  It is applied through the shared ``_reframe`` helper at
the cell ops; the OUTGOING FACE (``psi_out``) stays sweep-frame — it propagates
along the wavefront and never crosses into the global-frame iterate (so it is
left untouched).  DD/Step (``spatial_basis_per_axis == 1``) get ``None`` (the
sign-invariant average-only moment), so they pass through ``_reframe``
untouched and stay byte-identical (the negative control); a flat scalar source
(matvec zero / flat external — only the average moment) is frame-invariant and
skipped by the ``arr.shape[-1] != frame_signs.shape[0]`` guard, so it is never
broadcast into a spurious moment axis.

After the fix the diffusion limit is recovered on BOTH the matvec (Krylov) and
the sweep (source-iteration) paths:

.. list-table:: Thick-slab LD vs DD relative error, before/after the frame fix
   :header-rows: 1
   :widths: 14 22 22 22

   * - Mesh
     - Cell optical depth
     - Before (sweep-frame slope)
     - After (global-frame slope)
   * - 1-D, :math:`n_x = 4`
     - :math:`\Sigma_t h = 10`
     - 38.9 %
     - 4.1 %
   * - 1-D, :math:`n_x = 16`
     - :math:`\Sigma_t h = 2.5`
     - 7.9 %
     - 0.2 %
   * - 1-D, :math:`n_x = 64`
     - :math:`\Sigma_t h = 0.6`
     - 0.9 %
     - 0.0 %
   * - 2-D, :math:`n = 4/8/16`
     - thick :math:`\to` thin
     - 8.4 %
     - 1.7 % :math:`\to` 0.4 %

.. warning::

   **The matvec-self-consistency gate is necessary but NEVER sufficient for a
   moment-iterate fold.**  Every component here was individually correct (the
   2×2 matched LM-1989 Eq. 4.3, the dense UBLD matched the analytic 2×2, the
   scattering produced :math:`\Sigma_s\hat\phi` at full strength, the matvec
   round-trip vanished to :math:`10^{-16}`, and source-iteration :math:`\equiv`
   Krylov on the SAME operator).  The bug was the frame consistency *between*
   two correct components — a wrong fixed point that the round-trip and the
   SI :math:`\equiv` Krylov gates are structurally BLIND to: they prove the
   operator is internally consistent, NOT that its fixed point is the
   physically-correct one (`vv-principles` §5 — "O(h²) to the wrong limit is
   still O(h²)").  The decisive evidence was a structurally-independent
   from-scratch LD-SN solver (a direct LM-1989 2×2 + source iteration, no
   ORPHEUS kernel) that reproduced ORPHEUS's WRONG value bit-for-bit when it
   summed sweep-frame slopes and RECOVERED the diffusion limit when it stored
   global-frame slopes — pinning the root cause independent of ORPHEUS's code.
   The lesson: gate the converged VALUE against a structurally-independent
   reference (the continuous diffusion solution + the independent from-scratch
   kernel), never the round-trip.  This is failure Mode 1 (sign flip) +
   Mode 6 (convention drift) — see ``error_catalog.rst`` ERR-061.

The thick-cell diffusion tripwire is
``tests/sn/verification/mms/test_mms_ld_slab.py::test_ld_thick_diffusive_limit``
(1G) and ``::test_ld_thick_diffusive_limit_2g`` (2G heterogeneous, a
group-coupled slope source — Mode 6), both ``@pytest.mark.l1
@pytest.mark.catches("ERR-061")`` and both Mode-8-safe
(``np.testing.assert_array_less`` fires under ``-O``).  The slope-frame
fingerprint is pinned by
``derivations/diagnostics/diag_240_d5b_s3_probe_11_root_cause.py`` (forward and
backward ordinate slopes must share sign in the global frame), and the
structurally-independent confirmation by
``diag_240_d5b_s3_probe_08_independent_ld.py``.

.. _ld-ubld-pure-z-collision-twin:

The pure-z collision-only twin — sweep :math:`\equiv` matvec single source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
   :label: ld-ubld-pure-z-collision

   \text{(solve)}\quad \bar\psi = \frac{Q}{\sigma_t}
   \qquad\Longleftrightarrow\qquad
   \text{(matvec)}\quad (L+C)\,\bar\psi = \sigma_t\,\bar\psi

.. (V&V scope note) Structural L21 twin identity (collision-only sweep
   == matvec for pure-z ordinates). Wired to the ERR-062 gate
   test_ld_2d_krylov_equals_si_pure_z_quadrature (foundation).

The pure-z degenerate ordinates (:math:`\mu_x = \mu_y = 0`, the :math:`\pm z`
poles of a Lebedev or product cubature in a 2-D Cartesian sweep) have no
in-plane streaming, so the cell is **collision-only**: the loss couples to
:math:`\sigma_t` alone, and the sweep balance :math:`\bar\psi = Q/\sigma_t` and
its matvec twin :math:`(L+C)\bar\psi = \sigma_t\bar\psi` are two applications of
the SAME operator (:eq:`ld-ubld-pure-z-collision`).  This is the L21 twin-path
relationship — sweep and matvec are the same physics evaluated in opposite
directions — and it is exactly the kind of paired closure that drifts when a new
axis lands on one side and is forgotten on the other.

At a multi-moment closure (LD) the source :math:`Q` and the probe
:math:`\bar\psi` carry the trailing :math:`2^d` spatial-moment axis that
:math:`\sigma_t` of shape :math:`(ng, *\text{spatial})` lacks; each moment
scales by the SAME scalar (:math:`1/\sigma_t` on the solve, :math:`\sigma_t` on
the apply), so :math:`\sigma_t` must gain a length-1 trailing axis to broadcast.
This reshape is single-sourced through
:func:`~orpheus.sn.loss_representation._moment_broadcast_sigma`
(:math:`\sigma \mapsto \sigma[\ldots, \text{None}]` iff the moment-valued
operand out-ranks :math:`\sigma`), called by BOTH the sweep ``pure_z`` arm
(:math:`Q\,/\,\texttt{\_moment\_broadcast\_sigma}(\sigma_t, Q)`) and the matvec
``pure_z`` arm
(:math:`\texttt{\_moment\_broadcast\_sigma}(\sigma, \bar\psi)\cdot\bar\psi`), so
the twin cannot diverge on the moment-axis convention.  DD/Step (no moment axis)
:math:`\to` :math:`\sigma_t` unchanged, byte-identical.

.. admonition:: ERR-062 — the matvec twin forgot the guard the sweep already had
   :class: warning

   Before this fix the sweep arm HAD the moment-broadcast guard but the matvec
   arm wrote the bare ``sigma * probe[oct_idx]``, so a moment-valued probe
   broadcast-FAILED.  The consequence:
   ``solve_sn_fixed_source(scheme=LinearDiscontinuous(), inner_solver="krylov")``
   on ANY 2-D Cartesian LD mesh whose quadrature carries pure-z ordinates raised
   ``ValueError`` at the first Krylov matvec.  The bug hid through the whole
   D5b-S3 development because every committed 2-D LD test used
   ``level_symmetric`` — which has NO pure-z ordinates — while the production
   MMS uses a Lebedev quadrature that does.  This is the canonical L21
   twin-path asymmetry recurring a THIRD time ("the matvec needs a committed
   gate, not a round-trip"): the round-trip and FFW :math:`\equiv` MFW gates
   ran on ``level_symmetric`` and never exercised the pure-z arm at all.  The
   gate is
   ``tests/sn/verification/mms/test_mms_ld_2d.py::test_ld_2d_krylov_equals_si_pure_z_quadrature``
   (``@pytest.mark.foundation @pytest.mark.catches("ERR-062")``), on a Mode-9
   degeneracy-break config: a pure-z-bearing Lebedev order-5 quadrature
   (:math:`N = 14`, genuine :math:`\mu_y` + the 2 :math:`\pm z` poles),
   heterogeneous 2-material map, 2-group asymmetric XS with non-zero self-scatter,
   NON-SQUARE :math:`5\times4`, vacuum edges.  Mutation-verified: re-introducing
   the bare ``sigma * probe[oct_idx]`` makes the gate FAIL with the exact
   ``ValueError``; with the fix Krylov :math:`\equiv` SI to :math:`\sim10^{-11}`
   (the same :math:`(L+C-S_{\rm full})` fixed point).  See ``error_catalog.rst``
   ERR-062.

The source is :mod:`orpheus.sn.loss_representation` (``_moment_broadcast_sigma``,
the ``loss_action`` matvec ``pure_z`` arm and its source-iteration sweep twin);
the closeout is
``.claude/agent-memory/method-implementer/issue_240_d5b_s3_purez_gate_closeout.md``.

.. _ld-ubld-moment-scan:

The :math:`d{=}1` moment SCAN (the production sweep) — D5b-S3 OWED-2
------------------------------------------------------------------------

The unified moment *matvec* above is the APPLY direction; the production
:math:`d{=}1` LD SWEEP (source iteration) rides the fast Blelloch parallel-prefix
scan (:class:`~orpheus.sn.loss_representation.CumprodScan`), NOT the dense
per-cell solve (L16).  Sub-step **D5b-S3 OWED-2** threads the spatial-moment
iterate :math:`\hat\phi` through that scan so the SI path recovers the SAME
diffusion-limit-consistent operator the matvec does.

.. math::
   :label: ld-ubld-moment-scan-source

   b \;=\; \underbrace{\bar S\,\frac{\mathrm{inv}}{w}}_{\text{flat (average) emission}}
       \;+\; \underbrace{\frac{\theta\,\hat S}{D_2'}
              \;-\; \frac{\theta\,|\mu| A_{\rm down}\,\hat S}{D_2'}\,
                    \frac{\mathrm{inv}}{w}}_{\text{slope source}\ \Sigma_s\hat\phi}

.. (vv-status rationale) Derivation step: the slope-augmented d=1
   moment-scan affine source. Its terminal (scan == DAG, and the diffusion
   limit on the SI path) is tested downstream.
.. vv-status: ld-ubld-moment-scan-source documented

The scan propagates the scalar downstream FACE
:math:`\psi_{\rm out} = a\,\psi_{\rm in} + b` along the cell chain with the
**slope-augmented** affine source :eq:`ld-ubld-moment-scan-source`: the flat
(cell-average) emission :math:`\bar S\,\mathrm{inv}/w` plus the slope-source
contribution :math:`\theta\hat S/D_2' - (\theta|\mu|A_{\rm down}\hat S/D_2')\,\mathrm{inv}/w`
that carries :math:`\Sigma_s\hat\phi` into the recurrence.  Then it reconstructs
the per-cell :math:`(\bar\psi, \hat\psi)` moments from the chained upstream face.
The slope-row :math:`\hat S` algebra is single-sourced through
:meth:`~orpheus.transport.spatial._ubld.D1ClosedForm._slope_fold`, shared by the
per-cell matvec Schur (:meth:`~orpheus.transport.spatial._ubld.D1ClosedForm.schur_xV`)
AND the scan
(:meth:`~orpheus.transport.spatial._ubld.D1ClosedForm.scan_slope_face_source` for the
face-chain term, :meth:`~orpheus.transport.spatial._ubld.D1ClosedForm.scan_reconstruct`
for the per-cell moments).

Why the face/cell split is necessary (the load-bearing math)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A moment-carrying parallel-prefix scan is **NOT** a drop-in widening of the
scalar scan.  For *flat-source* LD the cell average is the convex blend of the
two faces, :math:`\bar\psi = (1-w)\psi_{\rm in} + w\,\psi_{\rm out}`, so the
scalar scan can reconstruct :math:`\bar\psi` directly from the chained faces.
With a *slope* source, that closure **decouples**: :math:`\bar\psi` and
:math:`\psi_{\rm out}` no longer satisfy the convex blend, because the slope
source enters the cell balance without entering the face propagation the same
way.  The scan therefore splits the work in two:

#. the FACE chain :math:`\psi_{\rm out} = a\,\psi_{\rm in} + b` propagates with
   the slope-augmented :math:`b` (:eq:`ld-ubld-moment-scan-source`), so the next
   cell's :math:`\psi_{\rm in}` is the correct dense :math:`\bar\psi + \hat\psi`;

#. the CELL moments :math:`(\bar\psi, \hat\psi)` are reconstructed per cell from
   the chained :math:`\psi_{\rm in}` via the per-cell Schur
   (``scan_reconstruct``), **not** via ``cell_average``.

Conflating the two (using ``cell_average`` for :math:`\bar\psi` on the moment
scan) gives the WRONG cell average while the face chain still looks right — the
silent trap this split avoids.  The reconstruction was verified against a
from-scratch dense :math:`d=1` chain (face / :math:`\bar\psi` / :math:`\hat\psi`
all match to :math:`10^{-12}`) and against the live DAG (scan :math:`\equiv` DAG
to :math:`10^{-16}` on a 2G-heterogeneous non-flat config).

The same global-frame involution as the matvec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like the matvec, the scan applies the SAME
:func:`~orpheus.transport.spatial._ubld.octant_moment_frame_signs` involution
(:eq:`ld-ubld-octant-moment-frame-signs`) through the shared ``_reframe``
helper: the source moments are mapped global :math:`\to` sweep on INPUT and the
reconstructed :math:`(\bar\psi, \hat\psi)` sweep :math:`\to` global on OUTPUT, so
the angular reduction :math:`\hat\phi = \sum_n w_n \hat\psi_n`
(:eq:`ld-ubld-slope-angular-reduction`) is frame-consistent and the diffusion
limit is recovered on the source-iteration path too (the sweep-side analog of
the ERR-061 matvec fix).  The backward sweep flips the slope so forward and
backward ordinates reinforce rather than cancel.  The scalar OUTGOING FACE stays
sweep-frame — it propagates along the chain and never crosses into the global
iterate (the :math:`d=1` face cochain is :math:`2^{d-1} = 1`, scalar, so it is
not reframed).  The scan is a **consumer** of the matvec's machinery, never a
twin: the same ``_slope_fold`` powers the per-cell Schur and the scan, and the
same involution powers the DAG and the scan.  DD/Step (``per_axis == 1``) get no
moment axis, so the scan runs the existing flat slab body verbatim and stays
byte-identical (the negative control).

The SymPy module / live scheme is :mod:`orpheus.transport.spatial._ubld`
(``D1ClosedForm``),
:meth:`orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous.moment_scan_closure`,
and :meth:`orpheus.sn.loss_representation._OneDimScanWalk._run` (the slab
joint-batch moment branch); the gates are
``tests/sn/verification/mms/test_mms_ld_slab.py::test_ld_two_paths_scan_equals_dag_oracle``
(scan :math:`\equiv` DAG) and ``::test_ld_thick_diffusive_limit`` (the diffusion
limit on the SI path, the same ERR-061 catcher the matvec uses); the closeout is
``.claude/agent-memory/method-implementer/issue_240_d5b_s3_owed2_scan_closeout.md``.


.. _sn-angular-windowing:

Angular windowing — the SI iterate lives in moment space
========================================================

Wave O step #205 **Phase 5a** (commits ``93807aa`` factoring / ``b97d4f9``
eigenvalue inner / ``13ca001`` fixed-source inner, 2026-06-07) is a
**moment-reduction** of the SN within-group source-iteration *iterate*.
It is **orthogonal** to the :ref:`interior face-flux cochain <wavefront-flux-cochain>`:
where that types the *per-ordinate* interior face flux a single
sweep propagates (and explicitly frames the interior cochain as
per-ordinate, :math:`\psi^{(1)}_\Omega \in C^1`), Phase 5a observes that
the **persistent** iterate the source iteration carries *between* sweeps
does not need all :math:`N` ordinates — it needs only the
spherical-harmonic moments the scattering operator consumes. The
held iterate's angular dimension drops :math:`N \to (L+1)(2L+1)`, and
the iterate becomes :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
instead of :class:`~orpheus.transport.fields.angular_flux.AngularFlux`.

.. admonition:: Key Facts (angular windowing)
   :class: tip

   - **The SI fixed point lives in moment space.** Within-group source
     iteration is :math:`\psi_{k+1} = (L{+}C)^{-1}(S\,\psi_k + B\,\psi_k
     + q)`. The scattering source :math:`S\,\psi` is a pure function of
     the flux moments :math:`\phi_\ell^m = (M\psi)_\ell^m` — the
     per-ordinate iterate :math:`\psi` carries strictly more than the
     iteration consumes.
   - **Hold the moments, not the ordinates.** The persistent iterate is
     the moment tensor :math:`\phi \in (L{+}1, 2L{+}1, n_g, n_x, n_y)`,
     not :math:`\psi \in (N, n_g, n_x, n_y)`. Measured **18.3×**
     persistent-iterate shrink at :math:`N = 110`, :math:`L = 1`
     (:math:`N / (L{+}1)(2L{+}1) = 110/6`).
   - **The per-step source is bit-identical.** :math:`S` consuming the
     moments equals :math:`S` consuming the full angular flux **bit for
     bit** (0 ULP) under the ORPHEUS unnormalized-harmonic convention.
     The only non-bit-identical change is the SI *convergence test*,
     which moves to the moment :math:`L^2` — *more* principled, not a
     regression.
   - **2-D Cartesian only (load-bearing).** Windowing is valid where the
     sweep is a **direct** solve with no per-ordinate-iterate seed.
     Curvilinear (1-D sphere / cylinder) **must** stay full-angular: the
     Morel–Montry Carlson coupled-pole closure seeds from the previous
     iterate's per-ordinate :math:`\psi` at :math:`\mu = -1`, which the
     moment tensor cannot reconstruct. The Krylov path stays
     full-angular too. Gated on the genuine ``sn_mesh.is_cartesian and
     sn_mesh.ndim == 2`` (C5.4 / #225 — the earlier ``reduced is None``
     proxy was also true at 3-D Cartesian).
   - **Interior-bulk only.** The reflective :math:`B` coupling reads the
     full per-ordinate boundary *trace*; windowing reduces only the
     interior bulk. The biproduct :eq:`wavefront-cochain-biproduct`
     keeps the trace :math:`C^1_\partial` a distinct, **un-reduced**
     summand.
   - **A representation + typed-state win, NOT yet a peak-memory win.**
     5a shrinks the *persistent* iterate (18.3×) and makes its type
     honest. The *peak* memory drops only modestly (~1.2× measured)
     because the per-sweep **transient** full-angular machinery still
     dominates — that transient is the target of Phase 5b (interior-face
     cochain) and Phase 5c
     (:ref:`full-angular output <sn-angular-windowing-in-sweep-accumulation>`,
     the 3.06× linear peak win).


.. _sn-angular-windowing-fixed-point:

The within-group fixed point lives in moment space
--------------------------------------------------

The within-group source iteration solves, for each outer step, the
fixed-point problem

.. (vv-status rationale) The within-group source-iteration fixed point.
   This is the governing iteration the windowing reorganizes; the
   verifiable content is the SI ≡ Krylov cross-check (Krylov stays
   full-angular) and the closed-form k_inf eigenvalue, not the rendered
   recurrence. Documented, not orphan-gated.
.. vv-status: si-within-group-fixed-point documented

.. math::
   :label: si-within-group-fixed-point

   \psi_{k+1}
   \;=\; (L + C)^{-1}\!\left( S\,\psi_k + B\,\psi_k + q \right),

where :math:`L + C` is the within-group **invertible resolvent** (the
streaming + collision the sweep inverts directly), :math:`S` is the
within-group scattering gain (:ref:`pn-scattering` in
:doc:`/theory/methods/sn/slab_multigroup`), :math:`B` is the reflective boundary
coupling (:ref:`bc-extraction`), and :math:`q` the fixed external /
fission source. Both :math:`S` and :math:`B` are **lagged gains** — the
sweep never re-scatters mid-sweep (cf. the variadic driver,
:ref:`bc-extraction-variadic-driver`).

The load-bearing observation is the **arity of the scattering gain**.
:math:`S\,\psi` depends on :math:`\psi` *only* through its
spherical-harmonic flux moments. Writing the moment-projection operator
:math:`M` (the :eq:`flux-moments` quadrature contraction, the SH frame's
analysis face ``frame.analysis`` from
:meth:`Quadrature.angular_frame(L)
<orpheus.numerics.quadrature.Quadrature.angular_frame>`)

.. math::
   :label: angular-windowing-moment-projection

   \phi_\ell^m(\vec r)
   \;=\; (M\psi)_\ell^m(\vec r)
   \;=\; \sum_{n=1}^{N} w_n \, Y_\ell^m(\hat\Omega_n)\,
         \psi_n(\vec r),
   \qquad 0 \le \ell \le L,\; |m| \le \ell,

.. (vv-status rationale) Notation/definition (the moment projection M =
   flux-moments quadrature contraction, the SH frame analysis face). The
   bit-identity of the windowed and full-angular arms is foundation-verified
   (0 ULP).
.. vv-status: angular-windowing-moment-projection documented

the within-group emission factors **through the moment boundary**:

* the **isotropic** :math:`\ell = 0` (P0) in-scatter and the
  **(n,2n)** emission (:ref:`pn-scattering`, :ref:`n2n-reactions`)
  need only the scalar flux :math:`\phi_0 \equiv \phi_0^0` — the
  latter from its own operator since CS4c step 3, which changes
  nothing about this factoring: an isotropic emission reads the
  :math:`\ell = 0` moment whichever operator owns it;
* the **anisotropic** :math:`P_{\ell\ge 1}` term needs the higher
  moments :math:`\phi_\ell^m` up to the scattering order :math:`L`.

So the per-ordinate iterate :math:`\psi \in \mathbb{R}^N` is mapped, at
the very first thing the sweep's source assembly does, onto the
:math:`(L{+}1)(2L{+}1)`-dimensional moment space — and **nothing
downstream of that projection ever reads the discarded
:math:`N - (L{+}1)(2L{+}1)` angular degrees of freedom**. The iterate
carries strictly more than the iteration consumes. Angular windowing
holds the iterate at the consumed dimension: the persistent state is
the moment tensor

.. math::
   :label: angular-windowing-moment-iterate

   \phi \;\in\; \mathbb{R}^{(L+1)\times(2L+1)\times n_g \times n_x \times n_y}
   \quad(\texttt{HarmonicMomentFlux}),
   \qquad\text{not}\qquad
   \psi \;\in\; \mathbb{R}^{N \times n_g \times n_x \times n_y}
   \quad(\texttt{AngularFlux}).

.. vv-status: angular-windowing-moment-iterate documented

For :math:`N = 110` (Lebedev order 17) and :math:`L = 1`
(:math:`(L{+}1)(2L{+}1) = 6`) the angular dimension drops **18.3×**.


.. _sn-angular-windowing-factoring:

The scattering factoring — :math:`S_{\rm aniso} = \tfrac{1}{W}\,R\,\Lambda\,M`
------------------------------------------------------------------------------

Phase 5a's commit 1 (``93807aa``) makes the factoring *structural* so
that the windowed and full-angular paths share one source of truth. The
anisotropic in-scatter is the §9 operator composition (the §15.2
sum-of-tensor-products form; the production body is
``TransferOperator._redistribute_ordinates``, selected at construction
behind :meth:`TransferOperator.apply
<orpheus.transport.operators.transfer.TransferOperator.apply>` — it was the
public ``ScatteringOperator.build_aniso_source`` verb until #448)

.. (vv-status rationale) The R·Λ·M anisotropic-scattering factoring.
   A structural identity (associativity of the three-operator
   composition); the verifiable content is the bit-identity of the two
   evaluation arms, pinned by the de-risk probe Q2/Q3 (0 ULP) and the
   independent Bell & Glasstone hand reconstruction Q2b (1.5 ULP).
   Documented.
.. vv-status: angular-windowing-aniso-factoring documented

.. math::
   :label: angular-windowing-aniso-factoring

   Q^{\rm aniso}_n(\vec r)
   \;=\; \frac{1}{W}\,\bigl(R\,\Lambda\,M\,\psi\bigr)_n(\vec r),
   \qquad W = \sum_n w_n,

where (reading right to left):

* :math:`M` is the moment **projection** :eq:`angular-windowing-moment-projection`
  (the SH frame's analysis face ``frame.analysis``);
* :math:`\Lambda` is the per-:math:`\ell` block-diagonal scattering on
  moment space :math:`\Lambda = \sum_\ell P_\ell \otimes \Sigma_{s,\ell}`
  (:class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`);
* :math:`R` is the addition-theorem **reconstruction** with the
  :math:`(2\ell+1)` factor
  (the SH frame's reconstruction face ``frame.reconstruction``);
* :math:`1/W` is the producer-side normalization applied at the
  :meth:`~orpheus.transport.operators.scattering.ScatteringOperator.apply` boundary.

The associativity :math:`(R\,\Lambda)\,M` is the whole trick. The
**full-angular** path evaluates the composition as written —
:math:`R\cdot\Lambda\cdot M(\psi)`: project, scatter, reconstruct,
consuming the §5.6 :attr:`kernel <orpheus.transport.operators.transfer.TransferOperator.kernel>`
``= frame.conjugate(Λ)`` as a single composed ``np.ndarray`` operator. The
**windowed** path's iterate bulk **is** the moments :math:`\phi = M\psi`,
so :math:`M` is *already done* and only the **moment → source** map
:math:`R\,\Lambda` remains: :math:`R\cdot\Lambda(\phi)`.

Which of the two runs is a property of the **binding**, not of the
iterate. The gain retains its flux-analysis face :math:`M \otimes I`, and
that face has two ends; the binding's domain interior is required to be
one of them, and *which one it is* selects the interior body once, in
``__post_init__`` (:ref:`cs4c-ends-select-the-body`). The windowed driver
therefore binds its gains where its iterate lives —
``S.on_moment_domain()``, ``N2N.on_moment_domain()`` — and every operand
rides its own operator's domain.

⛔ **Until 2026-09-04 (CS4c step 5) this read "the dispatch is on the
iterate type".** It was accurate, and it described a shipped
non-endomorphism: the windowed driver handed a *moment* composite to a
gain bound ``(angular, angular)``, which then chose an arm from the
carrier's Python class per call — `[M]` **143 such feeds per windowed
solve** on a bit-exact frozen snapshot. Nothing could report the
mismatch, because the arm that absorbed it was registered on the operator
that did not own that domain. The same feed today is a
:class:`TypeError` naming both spaces and the re-binding verb; the
windowed snapshot ``2d_2g_p1_aniso_dd_8x4_het_si`` is bit-exact across
the change.

.. note::

   **The moment-end route takes the explicit typed edges.** Its body is
   ``self.source_reconstruction.apply(Λ.apply(φ)) / self.total_weight``:
   :math:`\Lambda` materialises a typed
   :class:`~orpheus.transport.source_sinks.harmonic_moment_source_sink.HarmonicMomentSourceSink`
   (the role-changing edge of the carrier grid,
   :ref:`scattering-carrier-grid`), which the retained :math:`R` face
   then reconstructs to an :class:`AngularSourceSink` on its own bound
   codomain. It threads the *same* :math:`\Lambda` kernel and the *same*
   frame :math:`R` face as the fused angular route, so the two agree
   numerically — `[M]` 200/200 ``array_equal`` on a 1-D GL8 :math:`P_1`
   slab and on the gate's own 2-D heterogeneous fixture
   (``tests/sn/operators/test_scattering_kernel_crosscheck.py``).

   The spelling reached this shape in three moves, each recorded: it read
   ``frame.reconstruct(…)`` until the F-1 carve moved the binding from
   frame VERBS onto the minted FACES; it read
   ``float(self.weights.sum())`` until the CS4c rebind retired the stored
   weight vector in favour of the faces' own frame measure
   (:ref:`scattering-binding-cs4c`); and it was a registered
   ``HarmonicMomentFlux`` **arm** of the angular operator until CS4c step
   5 made it the moment binding's own body.

   ⛔ The ndarray ``reconstruct_after(Λ)`` primitive
   ``_aniso_source_from_moment_values`` — described here until 2026-09-04
   as "retained as the 0-ULP crosscheck oracle" — is **retired**. The
   crosscheck's second side is no longer a private chain but the
   moment-bound operator's own public action, so the comparison moved up
   a tier and now pins two production routes against each other. See
   :ref:`scattering-carrier-grid` for the explicit-vs-fused design
   choice.

This factoring **retired** the per-:math:`\ell`
``_PerLegendreOrderScattering`` kernel, which recomputed :math:`M\psi`
independently for every Legendre order — an :math:`L`-fold redundant
projection (aggressive-retirement discipline).

.. admonition:: The :math:`Y_0^0 = 1` convention — the scalar flux is read off
   :class: note

   ORPHEUS uses **unnormalized real harmonics** (the
   "no-:math:`4\pi/(2\ell+1)`-prefactor" convention,
   :ref:`spherical-harmonics`), under which :math:`Y_0^0 = 1` *exactly*.
   Therefore the :math:`\ell = 0` moment **is** the scalar flux,

   .. math::

      \phi_0 \;=\; \sum_n w_n \, Y_0^0(\hat\Omega_n)\,\psi_n
                 \;=\; \sum_n w_n \, \psi_n
                 \;=\; (\texttt{integrate\_angular}\;\psi),

   read directly off the moment tensor's :math:`\ell{=}0` block with no
   rescale (``phi_moments.l_block(0)[0]``). This is what lets the
   windowed P0 + (n,2n) fast path consume the moments with **zero**
   conversion arithmetic, and what makes the eigenvalue outer's scalar
   flux bit-identical to the full-angular ``integrate_angular`` (the
   de-risk Q1 below proves :math:`\Phi[0,0] = \texttt{integrate\_angular}`
   bit-for-bit).


.. _sn-angular-windowing-geometry-restriction:

Why 2-D Cartesian only — the curvilinear seed obstruction
---------------------------------------------------------

Windowing is valid **only** where the within-group sweep is a *direct*
solve that does not seed from the previous iterate's per-ordinate
:math:`\psi`. There are three regimes, and only one admits it:

.. list-table:: Where angular windowing applies
   :header-rows: 1
   :widths: 26 16 58

   * - Path
     - Windowed?
     - Why
   * - **2-D Cartesian DD** (SI inner)
     - **yes**
     - The diamond-difference wavefront sweep is a direct forward
       substitution down the upwind DAG; it inverts :math:`L+C` from
       :math:`q + S\phi + B\psi_\partial` with **no** interior-iterate
       seed. The bulk seed (``initial_guess``) threads through harmlessly
       (the 2-D sweep ignores it). The moment iterate is sufficient.
   * - **Curvilinear 1-D** (sphere / cylinder)
     - **no**
     - The Morel–Montry **Carlson coupled-pole** closure seeds the
       :math:`\mu = -1` starting-direction angular flux from the
       *previous iterate's per-ordinate* :math:`\psi` (the curvilinear
       angular-redistribution recursion is initialized from the inward
       radial sweep's last iterate). A moment tensor cannot reconstruct
       that per-ordinate seed — windowing would lose the closure's
       starting data. Curvilinear stays full-angular.
   * - **Krylov** (any geometry)
     - **no**
     - GMRES iterates the full bulk vector :math:`\psi`; the Krylov
       subspace is built from full-angular matvecs. There is no moment
       sub-iterate to hold.

The gate is the genuine predicate ``sn_mesh.is_cartesian and
sn_mesh.ndim == 2`` (the C5.4 / #225 sharpening of the earlier
``reduced is None`` proxy, which was *also* true at 3-D Cartesian and
would have silently moment-windowed a 3-D solve — vv Mode 9): the
curvilinear meshes carry a non-``None`` ``reduced`` moment-reduction
descriptor and are excluded, and so is 3-D Cartesian (the in-sweep
moment emit is a 2-D kernel). The windowed product ``P @ A.inverse()``
(retyped in :ref:`windowing-retyped` below) — which the SI driver now
holds and applies **directly**, the ``_maybe_window`` factory returning
the plain sweep off that gate (#226 taxonomy step 3,
:ref:`inverse-application-driver`) — is therefore **never even
constructed** off the 2-D-Cartesian gate, so there is no illegal state
to mistype.

The restriction is **interior-bulk-only** in a second sense: the
interior-cochain biproduct :eq:`wavefront-cochain-biproduct`
:math:`C^1 = C^1_{\rm int} \oplus C^1_\partial` keeps the **boundary
trace** :math:`C^1_\partial` (the :class:`AngularBoundaryFlux` summand) a
distinct, *un-reduced* per-ordinate object. The reflective :math:`B`
coupling reads the full per-ordinate face trace via the typed
:math:`\iota_*` / :math:`\iota^*` exchange — windowing reduces only the
interior bulk and never touches the trace. The
``test_2d_windowed_si_reflective_trace_is_nonzero`` guard pins this (a
windowing that zeroed the trace would be a dropped reflective coupling,
invisible to the interior-only scalar-flux snapshot).


.. _sn-angular-windowing-bit-identity:

Bit-identity of the source, principled-equivalence of the convergence test
---------------------------------------------------------------------------

The carve has a clean correctness story split in two
(``vv-principles`` § "Bit-identity vs principled-equivalence").

**The per-step source is bit-identical.** A de-risk probe
(``derivations/diagnostics/diag_p5a_moment_consuming_scatter.py``, a
diagnostic script not retained in the repo) proved
the **moment arm** :math:`S(M\psi)` equals the **full-angular arm**
:math:`S(\psi)` **bit for bit** (``np.array_equal``, 0 ULP) before any
production code was written. Because the moment-projection operator
:math:`M` inside the windowed resolvent is built from the **same**
quadrature harmonics :math:`Y` and weights :math:`w` the scattering
operator uses internally, the stored moments equal :math:`S`'s own
internal projection of the same :math:`\psi` term-for-term — so the
per-sweep re-projection is not just *elided*, its result is *reproduced
exactly*. The probe was cross-checked against a **structurally
independent** Bell & Glasstone §1.6 hand reconstruction of the P1 source
(every factor — :math:`w_n`, :math:`Y`, the :math:`(2\ell+1)=3`, the
:math:`1/W` — written out by hand from the material data, *not* via the
project's projection primitives), which agreed at ~1.5 ULP (the expected
floating-point distance for an independent reduction order). This is the
L11 structural-independence guard the bit-exact comparison alone lacks.

.. list-table:: De-risk probe — moment arm vs full-angular arm
   :header-rows: 1
   :widths: 30 14 28 28

   * - Probe (config: 2-D P1 het, LS-S4)
     - Groups
     - Result
     - Verdict
   * - **Q1** :math:`\Phi[0,0] = \texttt{integrate\_angular}`
       (the :math:`Y_0^0 = 1` scalar-flux read)
     - 2G
     - max :math:`|\Delta| = 0`, ``np.array_equal`` True
     - bit-exact (0 ULP)
   * - **Q2** :math:`R\Lambda(\phi)` vs
       :math:`R\Lambda M(\psi)` (the aniso :math:`\ell\ge 1` arm)
     - 2G
     - max ULP :math:`= 0`
     - bit-exact (0 ULP)
   * - **Q3** full :math:`S(M\psi)` vs :math:`S(\psi)`
       (end-to-end source)
     - 2G **and** 4G
     - max ULP :math:`= 0` (both)
     - bit-exact (0 ULP)
   * - **Q2b** vs INDEPENDENT Bell & Glasstone hand
       reconstruction (L11 structural-independence ground)
     - 2G
     - max rel :math:`= 3.4\times10^{-16}`
     - principled (~1.5 ULP)
   * - **Q4** non-degeneracy: aniso :math:`> 0`,
       P1 :math:`\ne` P0, asymmetric :math:`\Sigma_s` (ERR-002 ground)
     - 2G
     - aniso max :math:`= 0.49`, :math:`|\Sigma_{s,0}-\Sigma_{s,0}^\top| = 0.1`–:math:`0.18`
     - non-degenerate (gate can see Mode 6)

**The convergence test moves to moment space (principled-equivalence).**
The *only* non-bit-identical change is the SI stopping criterion. The
full-angular path tested :math:`\lVert\psi_{k+1} - \psi_k\rVert_2`; the
windowed path necessarily tests :math:`\lVert\phi_{k+1} -
\phi_k\rVert_2`. This is the **physically-meaningful** SI criterion —
scattering iterates the *moments*, so the moment :math:`L^2` is the
natural convergence norm — i.e. the change is *more* principled, not a
regression. Because the stopping point shifts by a fraction of a
tolerance, the converged value agrees with the full-angular path within
``SAFETY × conv_tol``: the measured drift is **2-D eigenvalue
:math:`k_{\rm eff}` :math:`2.4\times10^{-14}` relative** and **scalar
flux :math:`6.3\times10^{-12}` relative**, both well inside the
regression framework's ``kind="iterative"`` gate (drift bounded by
:math:`(\text{iteration count})\times(\text{condition number})\times
\text{ULP}`, criterion 3 of the principled-equivalence test). The
eigenvalue **outer** converges on the scalar flux, which is bit-identical
via :math:`\phi_0` (the :math:`Y_0^0 = 1` read) — so *only* the inner SI
stopping point shifts; the outer power iteration is untouched.

The one *correctness* anchor (not merely *equivalence*) is the
closed-form homogeneous eigenvalue: the windowed 2-D eigenvalue solve
reproduces :math:`k_\infty = \nu\Sigma_f / \Sigma_a` (the closed-form
pillar — MMS does **not** prove eigenvalues), and the
``test_2d_p1_aniso_moment_path_carries_signal_and_si_krylov_agree`` gate
cross-checks the windowed SI flux against the **full-angular Krylov**
flux (a genuinely independent reference, since Krylov is never
windowed), confirming the windowing does not silently drop the
:math:`\ell\ge 1` moments — the central trap.


.. _sn-angular-windowing-honest-scope:

Honest scope — a persistent-iterate + typed-state win, NOT yet a peak win
-------------------------------------------------------------------------

.. warning::

   Phase 5a reduces the **persistent** iterate storage and makes the SI
   state honest. It does **NOT** by itself deliver the full peak-memory
   reduction. State this carefully — the 18.3× number describes the
   *held* iterate, not the *peak*.

   * **What 5a wins.** The held + warm-started iterate carried
     across the entire eigenvalue solve (``_psi_typed`` then; since #448
     the ``iterate`` member of the
     :class:`~orpheus.sn.solver.InnerSolve` record) and the
     convergence-test copy
     shrinks by :math:`N / (L{+}1)(2L{+}1)` — measured **18.3×** at
     :math:`N = 110`, :math:`L = 1`. The iterate **type** also becomes
     honest: the SI state *is* the moments
     (:class:`HarmonicMomentFlux`), so the representation no longer
     over-claims an angular resolution the iteration never uses.
   * **Why the peak win is modest.** The **per-sweep transient**
     full-angular machinery still dominates the peak: the resolvent's
     swept output, the ``psi_x`` / ``psi_y`` interior cochain
     :math:`C^1_{\rm int}` (:ref:`wavefront-flux-cochain`), and the per-octant
     ``.copy()`` buffers — several full-angular-sized arrays that
     storage-A still materializes within a single sweep. Measured
     **peak** reduction is only **~1.2×** for a :math:`12\times8` config
     (the held :math:`2\times` iterate restored to full-angular size
     against the windowed ``tracemalloc`` peak;
     ``derivations/diagnostics/diag_p5a_peak_memory.py``, a diagnostic
     script not retained in the repo).

   The per-sweep transient has two components, eliminated by two later
   phases. **Phase 5b** (interior-face storage-B — the rolling
   moving-frontier window over the wavefront anti-diagonals, which never
   materializes the whole interior cochain at once) cuts the *interior
   face* cochain transient and is the **3-D enabler** (a 3-D wavefront's
   full interior cochain is prohibitively large to materialize; the moving
   frontier is the only tractable representation). **Phase 5c**
   (:ref:`sn-angular-windowing-in-sweep-accumulation`) cuts the
   *full-angular output* transient by accumulating the harmonic moments
   per anti-diagonal inside the sweep — the **linear peak win** (measured
   3.06×), and the realization of the full peak reduction this honest-scope
   warning deferred.

So Phase 5a is precisely: the **persistent-iterate** shrink + the
**typed-state** correctness win + the **foundation** for 5b/5c. Phase 5b
eliminates the **interior-face transient** and is the **3-D** enabler;
Phase 5c eliminates the **full-angular output transient** (the linear peak
win). The three together deliver the asymptotic peak reduction the moment
iterate makes possible; 5a alone delivers the type and the
persistent-storage win.


.. _sn-angular-windowing-implementation:

Implementation map
-------------------

* The solver's :func:`_maybe_window <orpheus.sn.solver._maybe_window>`
  builds the typed windowed composition ``P @ A.inverse()`` — the
  :class:`~orpheus.sn.operators.windowing.WindowedSweep` — over the
  swept composite :math:`A`'s inverse (the Jacobi
  :math:`(L+C)^{-1}`, or the reified splitting matrix :math:`M^{-1}`,
  :class:`~orpheus.sn.operators.scheduled_invertible.ScheduledInvertibleOperator`
  — see :ref:`si-gauss-seidel-reification`)
  and hands it to the :class:`~orpheus.numerics.iteration.SourceIteration`
  driver, which **applies** it directly (#226 taxonomy step 3 —
  :ref:`inverse-application-driver`; the transitional
  ``_MomentWindowedResolvent`` ``.solve`` adapter that once wrapped it is
  gone). Its analysis factor :math:`P` is sourced from the scattering
  operator's **own** frame, which is what makes the stored moments equal
  :math:`S`'s internal projection term-for-term. The composite reports
  :attr:`~orpheus.numerics.operator.LinearOperator.is_invertible`
  ``= False`` — no round-trip promise (its :math:`P`
  factor is a coisometry) — and accepts the driver's ``initial_guess``
  kwarg accepted-and-ignored (:meth:`WindowedSweep.apply
  <orpheus.sn.operators.windowing.WindowedSweep.apply>`; the multi-D walk
  has no bulk-seed consumer). Gated on ``sn_mesh.is_cartesian and
  sn_mesh.ndim == 2``.
* The **eigenvalue** inner
  (:meth:`SNSolver._solve_source_iteration <orpheus.sn.solver.SNSolver._solve_source_iteration>`)
  is reconstruction-free: it returns the scalar flux read off the
  :math:`\ell{=}0` moment block (``psi_typed.bulk.l_block(0)[0]``, the
  :math:`Y_0^0 = 1` identity), since the outer power iteration rebuilds
  from the scalar flux.
* The **fixed-source** inner
  (:func:`~orpheus.sn.solver.solve_sn_fixed_source`'s SI path) adds a
  **one-shot** full-angular reconstruction for ``Solution.angular_flux``
  — it re-evaluates the converged source ``q + Σ gains·ψ`` once through
  the *un-wrapped* base resolvent (``base_resolvent.solve(...)``).
  Fixed-source must return the full per-ordinate field, and because
  :math:`S`/:math:`B` consume the moments *equal* the full angular's
  moments (de-risk proven), the reconstructed source is the same and one
  sweep reproduces the converged iterate by the fixed point — so the
  reconstruction is bit-identical to the un-windowed converged
  :math:`\psi`.

Cross-references: the moment math :eq:`flux-moments` / :eq:`pn-scatter`
and the :math:`Y_\ell^m` convention live in :ref:`pn-scattering`
(:doc:`/theory/methods/sn/slab_multigroup`); the projection :math:`M`
(equation :math:`\phi_\ell^m = \sum_n w_n Y_\ell^m \psi_n`) and the
addition-theorem reconstruction :math:`R` are the spherical-harmonic
:class:`~orpheus.numerics.frame.GalerkinFrame`'s ``analysis`` /
``reconstruction`` faces (:ref:`galerkin-projection`); the interior-face
cochain Phase 5a is orthogonal to is :ref:`wavefront-flux-cochain`; the
SI fixed point it reorganizes is the within-group inner of
:ref:`eigenvalue-posing`.


.. _sn-angular-windowing-in-sweep-accumulation:

Per-anti-diagonal moment accumulation — dropping the full-angular transient (Phase 5c)
--------------------------------------------------------------------------------------

Wave O step #205 **Phase 5c** (commit ``c7be111``, 2026-06-07) closes the
gap Phase 5a's :ref:`honest-scope warning
<sn-angular-windowing-honest-scope>` left open: the **per-sweep
full-angular transient**. Phase 5a held the *persistent* iterate as
moments (the 18.3× shrink) but still **materialized the full
per-ordinate angular field** :math:`\psi \in (N, n_g, n_x, n_y)` inside
*every* sweep, then projected it to moments **post-hoc** — a flat reduce
:eq:`angular-windowing-moment-projection` applied once at the end of the
sweep by the SH frame's analysis face ``frame.analysis.apply`` (the then
Phase-5a moment-windowed resolvent's ``solve`` called ``base.solve`` then
``self._projection.apply(full.bulk.values)``).
That transient full-angular array is the dominant peak-memory cost the
5a warning named.

Phase 5c moves the projection **into** the windowed anti-diagonal walk:
each topological level accumulates the harmonic moment tensor *directly*,
octant-by-octant into one shared global buffer, so the full angular
field is **never materialized** in the windowed iterate. The
``base.solve`` → flat-``apply`` post-projection **leaves production**; the
fused moment emit is what survives — reached at Phase 5c through the
resolvent's ``solve``, at #226 step 2 through the base resolvent's
``solve_moments``, and since #226 step 3 through the typed product's
:meth:`WindowedSweep.apply <orpheus.sn.operators.windowing.WindowedSweep.apply>`
that the :class:`~orpheus.numerics.iteration.SourceIteration` driver
applies directly (:ref:`windowing-retyped` below, and
:ref:`inverse-application-driver`).

.. admonition:: Key Facts (in-sweep moment accumulation)
   :class: tip

   - **The projection moves into the sweep.** Where 5a swept the full
     :math:`\psi` then flat-reduced :math:`\phi_\ell^m = \sum_n w_n
     Y_\ell^m \psi_n` once post-hoc, 5c accumulates the moments per
     anti-diagonal *during* the walk:
     :math:`\phi_\ell^m[\text{cells}_k] \mathrel{+}= \sum_{n\in
     \text{octant}} w_n Y_\ell^m(\hat\Omega_n)\,\psi_n[\text{cells}_k]`.
     The full :math:`(N, n_g, n_x, n_y)` field is never allocated for
     the windowed iterate.
   - **Measured peak-memory win: 3.06×.** On S8 / 4g / :math:`24\times24`
     the windowed ``solve`` peak drops 2.26 MB → 0.74 MB — the 1.47 MB
     full-angular transient is eliminated (moment tensor 0.111 MB). The
     win **grows with angular order**: the eliminated transient is
     :math:`N\,n_g\,n_x\,n_y`; the moment tensor is fixed at
     :math:`(L{+}1)(2L{+}1)\,n_g\,n_x\,n_y`.
   - **Principled-equivalence, NOT bit-identity.** The cross-octant
     ``+=`` reorders the ordinate sum vs the flat single-reduce; IEEE-754
     addition is non-associative, so the moments drift at ULP level.
     The per-cell :math:`w\,Y\,\psi` fold is **term-for-term identical**
     to the SH frame's analysis face ``frame.analysis.apply`` — only the
     accumulation *order* differs. Measured max-relative drift
     :math:`2.74\times10^{-16}` (:math:`\le 4N\varepsilon`, 4 ULP).
   - **The scalar is subsumed.** The scalar flux **is**
     :math:`\phi_0^0 = \texttt{moments}[0,0]` (:math:`Y_0^0 = 1`), read
     off the moment tensor — there is no separate scalar reduction
     (``coding-elegance`` Pattern 2: one source of truth).
   - **The fuller view is retained as a verification oracle.** The pre-5c
     "full-angular solve + flat ``frame.analysis.apply``" path is kept
     reachable and pins the optimized path
     (``feedback_aggressive_retirement`` — the "verification oracle"
     exception to retirement).
   - **2-D Cartesian only; the rest is untouched.** Gated on the genuine
     ``sn_mesh.is_cartesian and sn_mesh.ndim == 2`` (the C5.4 / #225
     sharpening of 5a's ``reduced is None`` proxy — the proxy was also
     true at 3-D Cartesian); the 1-D scan representation
     (:class:`~orpheus.sn.loss_representation.CumprodScan`) *raises*
     on a moment frame (illegal-states-
     unrepresentable). Curvilinear / Krylov stay full-angular; both
     one-shot ``Solution.angular_flux`` reconstructions stay separate
     full sweeps.


.. _sn-angular-windowing-in-sweep-transformation:

The transformation — post-hoc reduce → in-sweep accumulate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 5a's resolvent produced its moment iterate in **two arithmetic
stages**: (1) the wavefront walk materialized the full per-ordinate field
:math:`\psi \in (N, n_g, n_x, n_y)`, writing each anti-diagonal's
cell-average into the global angular buffer
(``angular_flux_octant[:, :, ii, jj] = psi_avg``); then (2) a flat
post-sweep reduce collapsed *all* :math:`N` ordinates at once,

.. math::

   \phi_\ell^m(\vec r)
   \;=\;\sum_{n=1}^{N} w_n\, Y_\ell^m(\hat\Omega_n)\,\psi_n(\vec r)
   \qquad\bigl(\texttt{frame.analysis.apply}, \;
   \texttt{einsum}\;\texttt{"n,nlm,n...->lm..."}\bigr),

which is :eq:`angular-windowing-moment-projection` again, evaluated once.
Stage (1)'s full-angular array is the transient that dominates the peak.

Phase 5c fuses the two stages. The :class:`_MovingFrontier
<orpheus.sn.loss_representation.sweep_graph._MovingFrontier>` walk
(:meth:`SweepDependencyGraph.walk_windowed
<orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_windowed>` — at 5c the
solve-direction ``apply_windowed``, collapsed into the level-op walk at
S6.4(e)) already
visited every cell exactly once, anti-diagonal by anti-diagonal, with the
per-level cell-average :math:`\psi_n[\text{cells}_k]` in hand. Phase 5a
threw that local quantity into a global angular buffer and re-read it
later; 5c **projects it on the spot** and discards it. For each level
:math:`k` (anti-diagonal ``cells_k = (ii, jj)``) of each in-plane octant,

.. (vv-status rationale) The in-sweep per-anti-diagonal moment
   accumulation. The verifiable claim is the fuller-view-oracle
   equivalence: the in-sweep accumulation equals the flat post-sweep
   frame.analysis.apply of the same swept psi within the
   reduction-order drift bound (criterion 3 of bit-identity-vs-
   principled-equivalence). Pinned by
   test_2d_windowed_product_equals_post_projection (renamed from
   test_2d_windowed_moments_in_sweep_equal_post_projection when the
   solve_moments cross-reach retired into the typed product, #226 step 2),
   anchored to the structurally-independent SI≡Krylov-full scalar
   cross-check.

.. math::
   :label: harmonic-moment-projection

   \phi_\ell^m[\text{cells}_k]
   \;\mathrel{+}=\;
   \sum_{n\in\text{octant}} w_n\, Y_\ell^m(\hat\Omega_n)\,
       \psi_n[\text{cells}_k],
   \qquad
   \texttt{moment\_buf[:, :, :, ii, jj]}
   \mathrel{+}=
   \texttt{einsum("nlm,ngd,n->lmgd",}\,
       Y_{\rm oct},\,\psi_{\rm avg},\,w_{\rm oct}\texttt{)},

accumulated into **one shared global** ``moment_buf`` of shape
:math:`(L{+}1, 2L{+}1, n_g, n_x, n_y)`. The four in-plane octants are
swept sequentially (the octant loop — since S6.4(b) the shared
``_OctantWalk.sweep_group`` frame, then ``sweep_octant_group`` — each octant
completing its full frontier walk), so the global buffer receives a **cross-octant
``+=``**: octant 1's complete contribution, then octant 2's, and so on.
The output branch (since S6.4(e) inside the ``_CellSolve`` level
operation; at 5c inside ``apply_windowed``) is a single
two-way switch at the per-level output site — *angular mode* writes
``angular_flux_octant[:, :, ii, jj] = psi_avg`` and accumulates the scalar
(``scalar_flux_buf``); *moment mode* does the :eq:`harmonic-moment-projection`
``+=`` instead. The frontier walk and the cell kernel are **untouched**,
which is exactly why the ``window ≡ full-field`` angular-mode oracle
(:ref:`Phase 5b <wavefront-flux-cochain>`) stays bit-identical: 5c adds a
branch only at the *consumer* of ``psi_avg``, never in its *production*.

The reduction-order contrast is the whole story:

.. list-table:: Post-hoc reduce (5a) vs in-sweep accumulate (5c)
   :header-rows: 1
   :widths: 22 39 39

   * -
     - **Phase 5a — post-hoc flat reduce**
     - **Phase 5c — in-sweep accumulation**
   * - Full :math:`\psi` materialized?
     - **Yes** — written to the global angular buffer, then re-read
     - **No** — projected per level and discarded
   * - Reduction shape
     - one flat ``einsum`` over all :math:`N` at once
       (``"n,nlm,n...->lm..."``)
     - per-level ``+=`` per octant
       (``"nlm,ngd,n->lmgd"``), cross-octant accumulate
   * - Per-cell arithmetic
     - :math:`\sum_n w_n Y_\ell^m \psi_n`
     - :math:`\sum_n w_n Y_\ell^m \psi_n` (**term-for-term identical**)
   * - FP reduction tree
     - one fixed order
     - reordered by the octant grouping ⟹ ULP drift
   * - Peak working set
     - moment iterate **+** full-angular transient
     - moment iterate **only**


.. _sn-angular-windowing-in-sweep-equivalence:

Why it is principled-equivalence, not bit-identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 5a's per-step source was **bit-identical** to the full-angular arm
(0 ULP — :ref:`sn-angular-windowing-bit-identity`): the moments *stored*
equalled :math:`S`'s *own* internal projection of the same :math:`\psi`,
because the flat reduce and :math:`S`'s internal reduce shared the same
single reduction order. Phase 5c **necessarily** breaks that bit-identity
— and does so for a clean, documented reason rooted in the
``vv-principles`` § "Bit-identity vs principled-equivalence" three-criteria
test. The change is **accepted** because all three criteria hold:

#. **Principled at every step — the intermediate is named.** Each
   per-anti-diagonal partial moment :math:`\sum_{n\in\text{octant}} w_n
   Y_\ell^m(\hat\Omega_n)\,\psi_n[\text{cells}_k]` is **the octant's
   contribution to the harmonic moment** :math:`\phi_\ell^m` — a
   reactor-physics quantity (the per-direction-class flux moment), not a
   "whatever the reduction order produced" array. The cross-octant ``+=``
   composes named partial moments. This is exactly the
   unnamed-intermediate → named-intermediate move the criterion blesses
   (cf. the issue-#169 ``compute_keff`` worked example: per-group
   production rate is principled; the flat cell-product array is not).
#. **Verified against a structurally-independent reference — not
   old-vs-new ULP alone.** Old-vs-new distance is *necessary but not
   sufficient* (both arms could share a systematic offset). The
   structurally-independent ground is the **full-angular Krylov** scalar:
   ``test_2d_p1_aniso_moment_path_carries_signal_and_si_krylov_agree``
   cross-checks the windowed-SI moment :math:`\ell{=}0`
   (``scalar_flux()`` = :math:`\phi_0^0`) against the full-angular Krylov
   flux (Krylov is **never** windowed — it iterates the full bulk vector),
   and the closed-form homogeneous eigenvalue :math:`k_\infty =
   \nu\Sigma_f/\Sigma_a` (the closed-form pillar; MMS does **not** prove
   eigenvalues) anchors the eigenvalue reuse.
#. **The drift is FP-non-associativity, dimensionally explainable.** For
   a single-step computation the bound is :math:`(\text{reduction
   depth})\times\varepsilon`. The reduction depth is the ordinate count
   :math:`N`; the cross-octant ``+=`` reorders the **same** :math:`N`
   summands. The de-risk probe (``derivations/diagnostics/diag_p5c_moment_accum.py``,
   git-excluded, promoted into the permanent gate then deleted per the
   diagnostic-promotion policy) measured max-relative drift
   :math:`2.74\times10^{-16}` — about :math:`78\times` under the
   :math:`4N\varepsilon \approx 2.1\times10^{-14}` bound (:math:`N = 24`
   ordinates, 4× headroom for the partial-sum nesting within octants). A
   drift *above* :math:`4N\varepsilon` would signal an algorithmic change
   masquerading as FP noise — a wrong octant-:math:`Y` slice (Mode 2), a
   missing weight (Mode 3), or an :math:`\ell/m` index drift (Mode 5) —
   and the gate would catch it.

The scalar flux is **subsumed**, not separately reduced. Because
:math:`Y_0^0 = 1` (ORPHEUS unnormalized-harmonic convention,
:ref:`spherical-harmonics`), the :math:`\ell{=}0` slot **is** the scalar
flux, :math:`\phi_0^0 = \sum_n w_n \psi_n` (the :math:`Y_0^0 = 1` read of
:ref:`sn-angular-windowing-factoring`). The moment-mode sweep therefore
returns ``(moment_buf, None)`` — no separate ``scalar_flux_buf``
accumulation — and the eigenvalue inner reads
``moment_buf[0, 0]`` for the outer power iteration's scalar flux. The
windowed scalar is identical to the angular-mode scalar up to the same
reduction-order drift; the existing ``scalar_flux_buf`` einsum
``"ngd,n->gd"`` is literally the :math:`\ell{=}0` case of the moment
einsum ``"nlm,ngd,n->lmgd"`` with :math:`Y_0^0 = 1`.


.. _sn-angular-windowing-in-sweep-oracle:

The fuller-view verification oracle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 5c **relinquishes a fuller view** — the materialized full-angular
field that the post-hoc reduce consumed. By the
``feedback_aggressive_retirement`` "verification oracle" exception, an
optimization that gives up a fuller view of a concept **keeps that fuller
view reachable** as a verification oracle that pins the optimized path.
The pre-5c "full-angular ``base.solve`` then flat
``frame.analysis.apply``" path is exactly
that fuller view: it is *not* deleted, and — since #226 step 2 — it is
the **inherited**
:meth:`OperatorProduct.apply <orpheus.numerics.operator.OperatorProduct.apply>`
body of the windowed composition itself (``P.apply(A⁻¹.apply(rhs))``, the
un-fused factor-by-factor evaluation the
:class:`~orpheus.sn.operators.windowing.WindowedSweep` overrides). The
permanent gate ``test_2d_windowed_product_equals_post_projection``
asserts the fused in-sweep
:meth:`WindowedSweep.apply <orpheus.sn.operators.windowing.WindowedSweep.apply>`
result equals that deforested oracle of the same swept
:math:`\psi`, within the de-risk bound, over the **full moment tensor —
including the** :math:`\ell\ge 1` **block**.

The :math:`\ell\ge 1` coverage is the load-bearing reason this oracle
exists. The :math:`\ell{=}0` scalar cross-check (the SI ≡ Krylov-full
test, criterion 2 above) is **blind to** :math:`\ell\ge 1`: a windowing
that silently swapped or dropped a higher moment would converge the right
scalar flux while corrupting the anisotropic source. The oracle pins the
:math:`\ell\ge 1` block that the scalar cross-check cannot see — it is the
**Mode-5** (:math:`\ell/m` index drift) and **Mode-2** (wrong octant-:math:`Y`
slice) catcher.

.. warning::

   The oracle and the system-under-test share the **same** cell kernel
   (:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch`)
   and the **same** :math:`Y` / ``weights``, differing only in reduction
   *order*. They are therefore **procedurally** independent, **not
   structurally** independent (``vv-principles`` § "Structural
   independence"). The oracle gate alone is a same-math-rearranged
   comparison and is **not sufficient** on its own — it MUST be paired
   with the structurally-independent SI ≡ Krylov-full scalar anchor
   (criterion 2) and the closed-form :math:`k_\infty` eigenvalue. The
   oracle's *value-add* is precisely the :math:`\ell\ge 1` block the
   structurally-independent scalar anchor is blind to; its
   anti-degeneracy leg asserts :math:`\max|\phi_{\ell\ge1}| > 10^{-3}\,
   \max|\phi_0|` so the pinned higher-moment block is genuinely non-zero
   on the canonical config.

The metric the gate uses is the **scale-relative** drift
:math:`\max|\phi^{\rm SUT} - \phi^{\rm oracle}| / \max|\phi^{\rm oracle}|
\le 4N\varepsilon`, **not** an element-wise
``assert_array_almost_equal_nulp``. The moment tensor spans ~3 orders of
magnitude (the :math:`\ell{=}0` scalar dominates; the :math:`\ell\ge 1`
blocks are :math:`\sim 10^{-3}\times`), so an element-wise ULP comparison
would inflate a machine-:math:`\varepsilon` *absolute* difference on a
small :math:`\ell\ge 1` element into hundreds of ULP even when the reorder
is pure FP noise. The scale-relative drift is the principled-equivalence
quantity (criterion 3).


.. _sn-angular-windowing-in-sweep-numerical-evidence:

Numerical evidence
~~~~~~~~~~~~~~~~~~~

.. list-table:: Phase 5c gates (commit ``c7be111``)
   :header-rows: 1
   :widths: 34 18 48

   * - Gate
     - Result
     - What it pins
   * - **De-risk drift** (canonical config
       ``2d_2g_p1_aniso_dd_8x4_het_si``: 8×4 het, vacuum-x /
       reflective-y, mixture B :math:`\bar\mu = 0.6` P1, S4 :math:`N = 24`)
     - **2.74e-16** rel
     - in-sweep accumulation ≡ flat post-projection, :math:`\le 4N\varepsilon`
       (4 ULP) — criterion 3 (FP-reorder, not a bug)
   * - **Peak memory** (S8 / 4g / :math:`24\times24`,
       ``diag_p5c_peak_memory`` tracemalloc)
     - **3.06×** (2.26 MB → 0.74 MB)
     - the 1.47 MB full-angular transient eliminated; moment tensor 0.111 MB
   * - **Windowing L1**
       (``test_2d_anisotropic_windowing.py``, incl. the new equivalence gate)
     - **4 ✓**
     - P1 ≠ P0 + SI ≡ Krylov-full; full-recon self-consistency
       (:math:`\Sigma w\psi = \phi`); reflective trace ≠ 0; in-sweep ≡ oracle
   * - **Eigenvalue 2-D** (``test_keff_2d.py``)
     - **19 ✓**
     - closed-form :math:`k_\infty`, P1-changes-:math:`k_{\rm eff}`,
       SI ≡ Krylov, Jacobi ≡ G-S (Mode 9), refinement convergence
   * - **Regression snapshot**
       (``test_dd_regression.py``, ``2d_2g_p1_aniso_dd_8x4_het_si``)
     - within ``SAFETY × conv_tol``
     - scalar flux drift 6920 ULP / :math:`9.81\times10^{-13}` rel
       (~10× headroom under the :math:`1.0\times10^{-11}` gate)
   * - **5b oracles + matvec consumer**
       (``test_2d_full_field_oracle.py``, window≡full graph,
       ``TestAnisoMomentSourcePath``, A2D-1 hash pin)
     - bit-identical / intact
     - the angular-mode oracle and the moment *consumer* (``S.apply``)
       are untouched — 5c changes only moment *production*

The **peak-memory win grows with angular order** because the eliminated
transient scales as :math:`N\,n_g\,n_x\,n_y` while the moment tensor is
fixed at :math:`(L{+}1)(2L{+}1)\,n_g\,n_x\,n_y`: at S4 :math:`N = 24`,
:math:`L = 1` the moment tensor is :math:`6/24 = 1/4` the angular size;
at S8 :math:`N = 80` it is :math:`6/80 \approx 1/13`; a Lebedev-order-17
:math:`N = 110` it is :math:`6/110 \approx 1/18`. The transient is the
**linear** peak win Phase 5a's honest-scope warning deferred to "Phase
5b/5c": 5a delivered the persistent-iterate shrink, 5b (the
:ref:`moving-frontier interior cochain <wavefront-flux-cochain>`) cut the
interior face transient, and 5c cuts the full-angular *output* transient.


.. _sn-angular-windowing-in-sweep-honest-scope:

Honest scope — what 5c does and does NOT do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   Phase 5c eliminates the **per-sweep full-angular transient** of the
   *windowed SI* path. State the boundaries carefully:

   * **The persistent iterate was already moments (5a).** 5c does not
     shrink the held iterate — that was 5a's 18.3×. 5c shrinks the
     *transient* the held-moment iterate's sweep still materialized.
   * **Both** ``Solution.angular_flux`` **reconstructions stay full
     sweeps.** The eigenvalue final reconstruction and the fixed-source
     windowed one-shot reconstruction each re-run a *separate*
     full-angular sweep — the within-group resolvent ``solve``
     (:func:`~orpheus.sn.coupled_system.build_within_group_system`,
     applying its ``.implicit_operator``) — to return the user-facing
     :math:`(N, n_g, n_x, n_y)` field. They are **untouched** by 5c — the
     user-facing angular flux is bit-identical across *this* carve (the
     :math:`\Sigma w\psi = \phi` self-consistency gate and the step-3
     ``np.array_equal`` reconstruction pin both hold).

     ⛔ **"Separate" was the defect, and it was corrected on 2026-09-06
     (#448,** :doc:`ERR-083 </theory/verification/error_catalog>`\ **).**
     The two reconstructions were twins, and only the fixed-source one
     re-evaluated the converged source ``q + Σ gains·ψ``; the eigenvalue
     one built a :math:`P_0`-only source by hand, so at every
     ``scattering_order ≥ 1`` it was **not** bit-identical to the fixed
     point — ``[M]`` 8.776e-02 from the converged iterate on a 421-group
     fixture at :math:`L = 2`, and the self-consistency gate this bullet
     cites did not exist on the eigenvalue path.  Both entries now
     evaluate the ONE body
     :func:`~orpheus.numerics.iteration.fixed_point_step`
     (:ref:`sn-finalize-one-step`), so the sentence is true of both for
     the first time.
   * **Krylov, 1-D, and curvilinear stay full-angular.** Krylov iterates
     the full bulk vector (no moment sub-iterate); the curvilinear
     Morel–Montry Carlson coupled-pole seed reads the per-ordinate
     iterate at :math:`\mu = -1` (lesson L21), which the moment tensor
     cannot carry. Both are gated out by the genuine
     ``sn_mesh.is_cartesian and sn_mesh.ndim == 2`` (C5.4 / #225 — 3-D
     Cartesian is excluded too); the 1-D scan
     (:class:`~orpheus.sn.loss_representation.CumprodScan`) *raises* if a
     moment frame reaches it, so the unwindowable regime is
     unrepresentable, not merely unreached.

So the three-phase arc is complete: **5a** holds the persistent iterate
as moments (typed-state + persistent-storage win); **5b** carries the
interior face cochain on a rolling moving frontier (interior-transient
elimination + 3-D enabler); **5c** projects the swept flux to moments
in-sweep (full-angular *output*-transient elimination — the linear peak
win). Together they realize the asymptotic peak reduction the moment
iterate makes possible.


.. _sn-angular-windowing-in-sweep-implementation:

Implementation map — the moment-output threading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The moment OUTPUT mode threads as an **optional projection**
(``moment_frame=None``), mirroring the ``reflect=None``
dependency-injection idiom of :func:`_sweep_scheduled <orpheus.sn.loss_representation._sweep_scheduled>` — **not** a boolean flag.

.. note::

   **Retyping (#226 step 2).** At Phase 5c the two output modes were
   **named methods** on the resolvent surface — ``solve`` (full angular)
   vs ``solve_moments`` (moments). That public ``solve_moments`` — a
   method whose output-mode argument silently changed the operator's
   *codomain* — was a composition wearing a config, and it was retired.
   The moment emit is now the typed composition ``P @ A.inverse()``
   (:ref:`windowing-retyped`); the ``moment_frame`` kwarg survives only
   on the **private** ``_solve_timed_full_field`` body, the single
   application-context entry that the fused
   :meth:`WindowedSweep.apply <orpheus.sn.operators.windowing.WindowedSweep.apply>`
   drives. The map below is updated to that state.

* the solve-direction windowed walk (at 5c ``apply_windowed``; since
  S6.4(e) :meth:`SweepDependencyGraph.walk_windowed
  <orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_windowed>` × the
  ``_CellSolve`` level operation) — the single two-way branch at the
  per-level output site (angular write vs the
  :eq:`harmonic-moment-projection` ``moment_buf`` ``+=``). The
  frontier walk and cell kernel are untouched.  The apply-direction walk
  (at 5c ``residual_windowed``; now ``walk_windowed`` × ``_CellResidual``
  — the Krylov matvec) is untouched — Krylov stays full-angular.
* the per-group octant frame (``sweep_octant_group`` then, since S6.4(b),
  ``_OctantWalk.sweep_group``) /
  :func:`_sweep_scheduled <orpheus.sn.loss_representation._sweep_scheduled>` /
  :func:`_sweep_jacobi <orpheus.sn.loss_representation._sweep_jacobi>` — thread the
  optional ``moment_frame`` (2-D Cartesian only; the 1-D scan
  ``CumprodScan.sweep`` raises on a moment frame). Moment mode skips the
  per-octant angular allocation and the scheduled 2-D body returns
  ``(moment_buf, None)``.
* the private
  :meth:`StreamingCollisionOperator._solve_timed_full_field
  <orpheus.sn.operators.streaming.StreamingCollisionOperator._solve_timed_full_field>`
  body (duck-shared by
  :meth:`ScheduledInvertibleOperator._solve_timed_full_field <orpheus.sn.operators.scheduled_invertible.ScheduledInvertibleOperator._solve_timed_full_field>`)
  is the **single** application-context entry: given a ``moment_frame`` it
  emits harmonic moments, else the full angular field. ONE
  representation-sweep call per body for both output modes (since S6.5 the
  operator's own ``loss_representation.sweep`` — the same instance the
  matvec consumes); only the bulk **wrap** differs
  (:class:`~orpheus.transport.fields.angular_flux.AngularFlux` vs
  :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`).
  The former public ``solve_moments`` cross-reach — on
  ``StreamingCollisionOperator`` **and** the dissolved ``_GaussSeidelResolvent``,
  plus the solver-side ``_solve_scheduled`` plumbing twin — retired into
  this one private body (#226 step 2, :ref:`windowing-retyped`).
* the fused entry is
  :meth:`WindowedSweep.apply <orpheus.sn.operators.windowing.WindowedSweep.apply>`
  (``P @ A.inverse()``), which calls that private body with its
  ``moment_frame``; the analysis factor
  :class:`~orpheus.sn.operators.windowing.BulkAnalysisOperator` carries
  the MINTED flux-analysis face (F-1), whose frame — and its basis
  :class:`~orpheus.numerics.basis.SphericalHarmonicBasis` ``L`` — carries
  the truncation order. Since #226 step 3 the
  :class:`~orpheus.numerics.iteration.SourceIteration` driver **applies**
  that fused ``apply`` directly (:ref:`inverse-application-driver`); the
  transitional ``_MomentWindowedResolvent`` ``.solve`` adapter that once
  forwarded to it is gone.

Cross-references: the post-hoc reduce 5c replaces is
:eq:`angular-windowing-moment-projection`; the :math:`Y_0^0 = 1`
scalar-flux read is :ref:`sn-angular-windowing-factoring`; the
moving-frontier interior cochain 5c rides is :ref:`wavefront-flux-cochain`
(Phase 5b); the persistent-iterate shrink 5c completes is
:ref:`sn-angular-windowing-honest-scope` (Phase 5a). The
principled-equivalence framework is ``vv-principles`` § "Bit-identity vs
principled-equivalence".


.. _windowing-retyped:

Windowing retyped — the moment emit as a composition
----------------------------------------------------

Phases 5a–5c built the moment-windowed path as a pair of **named
methods** on the resolvent surface: ``solve`` returned the full angular
field, ``solve_moments`` returned the harmonic moments. The taxonomy
step-2 carve (#226) retired ``solve_moments`` — because a *public method
whose output-mode argument silently changes the operator's codomain is a
composition wearing a config*, not a mode of one morphism.

The two output modes do not share a codomain: ``solve`` lands in the
full-angular composite carrier :math:`V = V_{\rm bulk} \oplus V_\partial`,
while ``solve_moments`` landed in the *moment-bulk* composite
:math:`V^{\rm mom} = \Phi_{\rm bulk} \oplus V_\partial`. A change of
codomain is, by the two-layer law of :ref:`operator-algebra` ("two
operators, one substrate — never two views, one operator"), a
**different arrow**; and an arrow whose target differs from what
:math:`A^{-1}` produces is a *composition* of :math:`A^{-1}` with a
second arrow, never a configuration flag on :math:`A^{-1}`. The honest
object is therefore

.. math::
   :label: angular-windowing-operator

   \text{windowed} \;=\; P \circ A^{-1},
   \qquad
   P \;=\; \underbrace{M_{\rm frame}}_{\text{analysis on the bulk}}
           \;\oplus\;
           \underbrace{\mathrm{Id}}_{\text{on the trace}},

.. (vv-status rationale) Definitional operator composition (windowed =
   P o A^{-1}, the coisometry factoring). Pinned representationally by
   test_2d_windowed_product_equals_post_projection, anchored to the
   SI == Krylov-full scalar and closed-form k_inf.
.. vv-status: angular-windowing-operator documented

where :math:`A^{-1}` is the swept loss inverse — this section's
:math:`A` abbreviates the swept composite
(:class:`~orpheus.sn.operators.sweep_operator.SweepOperator` on
:math:`L+C`, or on the reified splitting matrix :math:`M` —
:ref:`si-gauss-seidel-reification`), the **inner kernel** of the
honest within-group :math:`(L+C-S-N_{2n}-B)^{-1}`, never the full solve —
and :math:`M_{\rm frame}` is the
scattering frame's :attr:`~orpheus.numerics.frame.FrameBase.analysis`
face, the angular→moment reduction :math:`\phi_\ell^m = \sum_n w_n
Y_\ell^m \psi_n` (:eq:`harmonic-moment-projection`). The boundary trace
passes through **un-reduced**: windowing is interior-bulk-only (the
reflective :math:`B` coupling reads the full per-ordinate face trace —
:ref:`sn-angular-windowing-geometry-restriction`), so :math:`P` is the
identity on the :math:`V_\partial` summand.

The coisometry factoring of the windowed contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`P` — the :class:`~orpheus.sn.operators.windowing.BulkAnalysisOperator`
— is a **block coisometry**, not an isomorphism. Its bulk face satisfies

.. math::

   \text{analysis} \circ \text{reconstruction} \;=\; 4\pi\,\mathrm{I}

under the no-prefactor SH convention (:ref:`spherical-harmonics`) — the
addition-theorem tight-frame identity, pinned by
``test_pi_R_is_4pi_identity_through_the_frame``. It is emphatically **not**
:math:`\mathrm{I}`: asserting :math:`\Pi R = \mathrm{I}` was the ERR-051
mistake — the coisometry carries the :math:`4\pi` frame constant, and a
test that hard-coded :math:`= \mathrm{I}` verified the *wrong* invariant.
In the other order :math:`\text{reconstruction} \circ \text{analysis}
\neq \mathrm{I}` (moments discard the ordinate-resolved angular content
above the truncation order :math:`L`), so :math:`P` is **not
invertible**. By the product's invertibility-closure law the composite
:math:`P \circ A^{-1}` therefore honestly reports ``is_invertible =
False`` (its ``P`` factor is structurally non-invertible), and makes *no
round-trip promise* — which is the whole point: the old ``solve_moments``
name suggested a *solve* (an inverse) where the object is a **projection
composed with an inverse**.

Fusion is an evaluation strategy, not a new operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The composition itself is
:class:`~orpheus.sn.operators.windowing.WindowedSweep` (an
:class:`~orpheus.numerics.operator.OperatorProduct` subclass), spelled by
the operator algebra ``P @ A.inverse()`` — the
:meth:`BulkAnalysisOperator.__matmul__ <orpheus.sn.operators.windowing.BulkAnalysisOperator.__matmul__>`
dispatch recognises a :class:`SweepOperator` right factor and returns the
fused product (mirroring the ``L + C`` fusion precedent, #261: the
dispatch is one-directional on the specific leaf).
:meth:`WindowedSweep.apply <orpheus.sn.operators.windowing.WindowedSweep.apply>`
overrides the inherited factor-by-factor body with the substrate's
MOMENT-emit mode (the per-anti-diagonal accumulation of :ref:`Phase 5c
<sn-angular-windowing-in-sweep-accumulation>`), so the
:math:`(N, n_g, n_x, n_y)` full-angular intermediate is **never
materialized** (the ~3× linear peak-memory win). The **inherited**
:meth:`OperatorProduct.apply <orpheus.numerics.operator.OperatorProduct.apply>`
body — ``P.apply(A⁻¹.apply(rhs))``, which *does* materialize the
intermediate — is retained verbatim as the **fuller-view verification
oracle** (the aggressive-retirement oracle exception): the two
evaluations differ only in the ordinate-sum reduction *order*, so they
agree to principled-equivalence within the scale-relative
:math:`4N\varepsilon` bound (measured :math:`1.8\times10^{-16}` on the
product SUT, re-measured from the pre-migration
:math:`2.7\times10^{-16}`). The permanent gate
``test_2d_windowed_product_equals_post_projection`` (renamed/migrated
from ``test_2d_windowed_moments_in_sweep_equal_post_projection`` when the
SUT became the product) is that fused-≡-deforested pin; it is a
*representation*-equivalence pin (procedurally, not structurally,
independent — shared kernel), whose structurally-independent anchors are
the SI ≡ Krylov-full scalar cross-check and the closed-form
:math:`k_\infty` eigenvalue.

.. admonition:: What was tried and rejected — the moment-proxy residual gate
   :class: warning

   An early proposal verified the windowed path with a **moment-proxy
   residual**: compute a residual *in moment space* and read its
   smallness as evidence that ``solve_moments`` inverted its operator. It
   was rejected as **category-confused**. :math:`P` is a coisometry, so
   :math:`P \circ A^{-1}` has no inverse and no round-trip to take a
   residual *of* — a residual gate presumes a solve, but the windowed
   object is a *projection composed with a solve*. Its only honest
   correctness statements are (i) representation-equivalence to the
   deforested oracle and (ii) the structurally-independent scalar anchor.
   Reifying the composition made the confusion **structural**:
   :class:`~orpheus.sn.operators.windowing.WindowedSweep` reports
   ``is_invertible = False`` (its coisometry ``P`` factor is
   non-invertible), so there is no round-trip an inverse-residual gate
   could measure.

At #226 step 2 one transitional wrapper remained: the driver still spoke
``.solve``, so a thin ``_MomentWindowedResolvent`` adapter held the
product and mapped its ``solve`` to the product's ``apply`` (the
``initial_guess`` kwarg accepted for the
:class:`~orpheus.numerics.iteration.SourceIteration` contract and
dropped). **Taxonomy step 3 removed even that** — the SI driver now
consumes inverse-application operators *directly*, so the production SI
holds ``P @ A.inverse()`` with no wrapper. That driver-consumption model
— how the solver builds the inverse and the driver applies it, why an
apply-only step operator is legitimate, and why the adapter could finally
dissolve — is :ref:`inverse-application-driver`.

Cross-references: the reduction :math:`M_{\rm frame}` is
:eq:`harmonic-moment-projection`; the frame's ``analysis`` /
``reconstruction`` faces are the
:class:`~orpheus.numerics.frame.GalerkinFrame`'s
(:ref:`galerkin-projection`); the two-layer "operators, not views" law is
:ref:`operator-algebra`; the reified :math:`M = (L+C-B_{\rm lower})` the
windowed forward may wrap is
:ref:`si-gauss-seidel-reification`; the invertibility-closure and
principled-equivalence framework is ``vv-principles`` § "Bit-identity vs
principled-equivalence".


The boundary Gauss-Seidel schedule (multi-D)
============================================

The source-iteration splitting and its spectral rate
:math:`\rho_J = c` are derived in :doc:`slab_one_group`
(:ref:`si-within-group-splitting`). The subsections below document
the multi-D *boundary Gauss-Seidel* schedule — what it accelerates,
what it does **not**, the reified splitting matrix, the
diagonal-cubature shared-face correctness rule (ERR-056), and the
measured evidence.

Jacobi vs Gauss-Seidel — recovering the reflective-coupling rate
----------------------------------------------------------------

The Wave O BC extraction (steps O.4a.2 + O.4b, Issue #208) made the
2-D sweep **bare**: it reads ``psi.boundary.inflow`` as *given* for
the whole sweep, and the reflective coupling is applied externally
via the sibling :math:`-B` term (see :ref:`bare-sweep-extraction`).
A side effect of that architectural win was a *rate regression*: the
retired ``bc.apply``-inside-the-sweep read the **live** boundary
buffer mid-sweep (intra-sweep Gauss-Seidel), whereas the bare sweep
with a fully-lagged external :math:`B` is **inter-sweep Jacobi** —
same converged fixed point, slower SI rate.  Phase 3 recovers the
intra-sweep reflective coupling through a polymorphic, mesh-time
:class:`~orpheus.sn.loss_representation.sweep_schedule.SweepSchedule` without
re-entangling the bare sweep with the BC.  Jacobi and Gauss-Seidel
are the **same** uniform sweep-and-reflect loop — there is *no*
``if jacobi/gs`` branch in the iteration; the splitting is selected
*once* by choosing the schedule:

.. list-table:: The two within-group SI schedules
   :header-rows: 1
   :widths: 16 42 42

   * - Schedule
     - Octant grouping
     - Inter-group reflect
   * - **Jacobi**
       (``"jacobi"``)
     - ONE group containing every octant; :math:`B\,\psi_n` seeded
       once and **frozen** for the whole sweep.  Identical to the
       pre-recovery bare all-octants sweep.
     - None.  All octants read the same lagged inflow seed.
   * - **Gauss-Seidel**
       (``"gauss_seidel"``, default)
     - One group per in-plane octant
       (:class:`~orpheus.sn.loss_representation.sweep_graph.OctantLabel`), in quadrature
       sweep order.
     - After each group, its reflective **outgoing** faces are
       re-reflected (the row-restricted :math:`-B`,
       :meth:`SNMaskedBoundaryOperator.reflect_rows_inplace
       <orpheus.sn.operators.boundary.SNMaskedBoundaryOperator.reflect_rows_inplace>`),
       so a *later* group reads the **fresh** current-iterate inflow —
       the :math:`(L+C-B_{\rm lower})^{-1}` forward substitution.
       ⛔ This cell named ``SNBoundaryOperator.reflect_into_inflow``
       until 2026-09-07, and that was **already wrong when it was
       written**: the reified :math:`M` has bound
       ``reflect=self.lower.reflect_rows_inplace`` since the #226
       taxonomy step-2 carve (``cc293ef3``, 2026-07-01 —
       :ref:`si-gauss-seidel-reification`), which is also when the
       ADDITIVE semantics replaced the dissolved resolvent's whole-face
       assignment.  The verb the cell named retired at CS4c step 6
       item 6.5 (2026-09-07) with zero production callers.

In the Gauss-Seidel schedule, octants swept **before** their
specular partner keep the lagged seed (the cyclic :math:`B_{\rm
upper}` back-edges — a both-faces-reflective axis is a 2-cycle, so
one pass is only *partial* G-S); octants swept **after** read the
fresh value (the order-respecting :math:`B_{\rm lower}` edges).  The
schedule is a **mesh-time derived object** — it depends only on the
quadrature's octant partition and the mesh's reflective-face set,
not on fluxes, sources, or iteration state — so it is built once and
reused across every SI iterate (the same lifetime contract as
:class:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph`).

The selection lives in :func:`~orpheus.sn.solver._select_si_splitting`,
which decides the **boundary** half of the splitting and nothing else:
``"gauss_seidel"`` on a multi-D Cartesian mesh returns
``((L+C) - parts.lower, parts.upper)`` — the implicit operator and the
boundary GAIN — realizing the splitting
:math:`(L+C-B) = M - B_{\rm upper}` (a splitting, **not** a *regular*
splitting — :ref:`sn-boundary-gs-not-regular`): the strictly-lower half
:math:`B_{\rm lower}` folds *into* the reified forward :math:`M`
(:class:`~orpheus.sn.operators.scheduled_invertible.ScheduledInvertibleOperator`,
whose ``solve`` is the octant-group forward substitution) while the
complement :math:`B_{\rm upper}` **lags as an ordinary external gain**.
``"jacobi"`` (and any 1-D mesh) returns ``(L+C, B)`` — the whole boundary
lagged (the degenerate :math:`B_{\rm lower}=0`).  The collision gains are
not the selector's business: the SI driver names the gain triple
``(S, N₂ₙ, boundary_gain)`` itself — §14.1's order, :math:`B` LAST — the
same way in both arms, so the
:class:`~orpheus.numerics.iteration.SourceIteration` driver needs **no**
case split (the dissolved ``_GaussSeidelResolvent`` needed a bespoke
``(…, (S,))`` gain shape; the reification made the two arms uniform).
In **both** cases :math:`S` and :math:`N_{2n}` are lagged gains: the sweep
never re-scatters mid-sweep.  Only the boundary coupling gets the
Gauss-Seidel treatment.  (Until the CS4c step-5 review round, 2026-09-05,
the selector passed :math:`S` and :math:`N_{2n}` through its return tuple
and the windowed driver rebuilt two of the three slots by index — the
smell the round removed; this paragraph had meanwhile drifted to a
two-gain ``(S, parts.upper)`` that the tree never shipped after §14.1.)

.. _si-gauss-seidel-reification:

The reified splitting matrix
----------------------------

The #226 taxonomy step-2 carve replaced the duck-typed
``_GaussSeidelResolvent`` with an honest **reified splitting matrix**.
The dissolution matters because the old resolvent paired an ``apply``
with a ``solve`` of **different operators**: its ``apply`` computed
:math:`(L+C)\psi` while its ``solve`` inverted :math:`(L+C-B_{\rm
lower})`.  An operator whose forward and inverse faces disagree is not an
operator — and the disagreement is measurable: its round-trip defect was
**O(1)** (:math:`\lVert M^{-1}(M\psi)-\psi\rVert = 2.667`, the §17
falsifier-3 finding).

The boundary Gauss-Seidel is exactly a **matrix splitting** of the
within-group loss (a *splitting*, not a **regular** splitting in Varga's
sense — that stronger reading was published here until 2026-08-09 and is
refuted in :ref:`sn-boundary-gs-not-regular`):

.. math::
   :label: si-gauss-seidel-splitting

   (L+C-B) \;=\; \underbrace{(L+C-B_{\rm lower})}_{M}
             \;-\; \underbrace{B_{\rm upper}}_{N},
   \qquad
   \psi_{k+1} \;=\; M^{-1}\bigl(q + S\,\psi_k + B_{\rm upper}\,\psi_k\bigr).

.. (vv-status rationale) Governing splitting identity ((L+C-B) = M - N).
   The identity itself is the governing one and is unchanged; what was
   corrected 2026-08-09 (#341) is only its NAME — it was called a "regular
   splitting", which is Varga's term for M^-1 >= 0 AND N >= 0 and does not
   hold here (see sn-boundary-gs-not-regular). Foundation-gated by the
   reified-splitting invariants (tests/sn/solve/test_gauss_seidel_reification.py
   — the W2 round-trip, the M-SPLIT mutations, and the FP-invariance gate),
   no isolated claim.
.. vv-status: si-gauss-seidel-splitting documented

The reified :math:`M`
(:class:`~orpheus.sn.operators.scheduled_invertible.ScheduledInvertibleOperator`)
is an honest :class:`~orpheus.numerics.operator.OperatorSum` over the
leaves :math:`\{(L+C),\,-B_{\rm lower}\}`, so its ``apply`` is the leaf
sum :math:`(L+C)\psi - B_{\rm lower}\psi` and its ``inverse()`` returns a
genuine :class:`~orpheus.sn.operators.sweep_operator.SweepOperator` on
:math:`M` — the two faces of **one** operator, the way :math:`A` and
:math:`A.H` are.  Because the scheduled walk (the same uniform
sweep-and-reflect loop as Jacobi, differing only in the
:class:`~orpheus.sn.loss_representation.sweep_schedule.SweepSchedule`) is
**exact forward substitution** for :math:`M`, ``M.inverse().apply`` now
round-trips ``M.apply`` at machine precision: the W2-round-trip gate
measures :math:`5.2\times10^{-16}` (bulk) / :math:`4.4\times10^{-16}`
(trace) — the O(1) defect gone.

The row-split law
~~~~~~~~~~~~~~~~~

:meth:`SNBoundaryOperator.split <orpheus.sn.operators.boundary.SNBoundaryOperator.split>`
returns a named :class:`~orpheus.sn.operators.boundary.BoundarySplit`
pair ``(lower, upper)`` of
:class:`~orpheus.sn.operators.boundary.SNMaskedBoundaryOperator` — each
the whole-trace :math:`B` masked to a per-face set of inflow ordinate
**rows**.  Which rows belong to which half is pure **schedule-order**
semantics, computed by
:meth:`SweepSchedule.lower_inflow_rows <orpheus.sn.loss_representation.sweep_schedule.SweepSchedule.lower_inflow_rows>`:

   An inflow row :math:`(f, m')` is in :math:`B_{\rm lower}` **iff**
   ordinate :math:`m'`'s octant group is swept *strictly after* face
   :math:`f`'s reflect group.

A reflective face :math:`f` is reflected exactly **once**, after its
**last** outflowing octant group (the ERR-056 fan-in rule above), at
which point every outflow feeding :math:`f` is complete.  A row swept
strictly after that reflect therefore reads the **fresh** current-iterate
reflection — realized in-sweep by the forward substitution
(:math:`B_{\rm lower}`); a row swept at-or-before the reflect keeps the
**lagged** seed (:math:`B_{\rm upper}`, the cyclic back-edges plus every
row of a never-reflected face — vacuum, white, albedo, periodic).  The
partition is **exact**: the specular map flips one direction-cosine sign,
so a row and its source always sit in *different* octants — :math:`B` has
no octant-diagonal, and :math:`B = B_{\rm lower} + B_{\rm upper}` is a
bit-exact per-face split (the W2-split gate).  The Jacobi schedule yields
an empty lower support (:math:`B_{\rm lower}=0`, :math:`B_{\rm upper}=B`)
— the degenerate that recovers the plain lagged-:math:`B` iteration.

The additive row-masked in-sweep reflect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solving :math:`Mz = y` on a strictly-lower inflow row is the inhomogeneous
forward-substitution row :math:`z_{\rm in} = y_{\rm row} + (Bz)_{\rm
row}`.  The buffer already holds the seed :math:`y_{\rm row}` (nothing
writes a lower row before its face's reflect), so the in-sweep reflect
**accumulates** the fresh reflection onto it — the ADDITIVE row-masked
verb
:meth:`SNMaskedBoundaryOperator.reflect_rows_inplace <orpheus.sn.operators.boundary.SNMaskedBoundaryOperator.reflect_rows_inplace>`
(``bf[rows] += (B·bf)[rows]``).  This is what makes ``M.inverse()`` exact
on an **arbitrary** rhs (not merely on production's zero-lower-row
subspace), and it leaves the upper (lagged) rows carrying the seed the
splitting :math:`\psi_{k+1}=M^{-1}(q+B_{\rm upper}\psi_k)` says they carry
— the returned trace **is** the splitting's honest iterate.

.. _gs-whole-face-overwrite-rejected:

.. admonition:: What was tried and rejected — the whole-face-overwrite reflect
   :class: warning

   The dissolved resolvent used a whole-face **ASSIGNMENT**
   (:math:`\psi.{\rm inflow} \leftarrow B\cdot\psi.{\rm outflow}`, carried
   until 2026-09-07 by the verb ``SNBoundaryOperator.reflect_inflow_inplace``).
   As the in-sweep row update it is **wrong**: it *dropped*
   :math:`y_{\rm row}` and stamped fresh values onto rows the splitting
   defines as **lagged**.  It was benign in production only because a
   reflective inflow row's seed is zero there — but O(1)-wrong as a
   general inverse, which is precisely the round-trip defect the old
   pairing masked.

   ⛔ **Until 2026-09-07 this paragraph closed with a retention ruling,
   and the block below it with that ruling's rationale.**  Both are quoted
   verbatim here, because the reasoning they record is sound and is what a
   future reader must not re-derive from scratch:

      "The whole-face assignment verb is retained (single source of truth
      via ``_reflect_trace``) for callers whose inflow is wholly recomputed
      each sweep and is not a solved unknown of a linear row."

      "The verb stays because the argument above is about what a whole-face
      assignment *means*, not about who happens to call it — and because a
      sweep-tier gate that reflects between sweeps needs exactly this
      semantics."

   What follows is why CS4c step 6 item 6.5 overrode them.

   ⛔ **Both of the production callers it was retained for are gone**, and
   the last one went at #448 (2026-09-06): the direct fixed-source SI loop
   routes through the variadic driver (Wave O O.2a), and the eigenvalue
   finalize is now one step of that same driven map, in which :math:`B` is
   a gain (:ref:`sn-finalize-one-step`).  ``[M]`` by AST,
   ``SNBoundaryOperator.reflect_inflow_inplace`` had **zero** call sites in
   ``orpheus/`` (its ψ½ sibling ``reflect_corner_inplace`` retired outright
   at #448 — no consumer anywhere); its last consumer was the sweep-tier
   gates' inter-sweep helper
   (``tests/sn/_test_helpers.py::reflect_outflow_into_inflow``), which is
   where the module-level ``_reflect_outflow_into_inflow`` had moved.

   ⛔ **CS4c step 6 item 6.5 (2026-09-07) retired the verb — and its
   trace-only leaf** ``reflect_into_inflow`` **with it**, since the façade's
   body *was* the leaf.  The retained-for-semantics argument above is
   correct about the **semantics** and wrong about the **surface**: it
   treated the two as one thing.  The assignment semantics do not need a
   second verb, because *zero the inflow rows, then reflect ADDITIVELY on
   the full-inflow mask* is the same map.  The Jacobi split's ``upper``
   half IS that mask (``[M]`` 4/4 geometries — a Jacobi schedule reflects
   no face in-sweep, so ``lower`` is empty and ``upper`` carries every
   inflow row of every face), and the pair reproduces the retired
   assignment **bit-for-bit** on a NON-zero-inflow buffer (``[M]``
   ``np.array_equal`` on **40/40 seeds × 4/4 geometries**; *dropping* the
   zeroing moves the answer by :math:`O(1)`, which is the positive
   control that the assignment-vs-additive difference is real and not a
   zero-inflow artefact — its magnitude is draw-dependent, so the gate
   asserts a floor rather than a value).
   The inter-sweep reflect therefore still exists, spelled on production's
   own live verb, at
   ``tests/sn/_test_helpers.py::reflect_outflow_into_inflow`` — no
   production surface was added, and one was removed.

   ⚠ One behaviour moved with the retirement rather than dying with it.
   ``reflect_rows_inplace`` used to filter its ``faces`` argument against
   its own rows *before* calling the trace core, so a face that is not a
   boundary face of the mesh was silently dropped (4/4 geometries), while
   the retiring trace-only verb raised through ``_reflect_trace``'s guard.
   That guard was reachable from no other public surface, so item 6.5 moved
   the refusal INTO
   :meth:`SNMaskedBoundaryOperator.reflect_rows_inplace
   <orpheus.sn.operators.boundary.SNMaskedBoundaryOperator.reflect_rows_inplace>`,
   ahead of the row filter and with a public witness: an unknown face is
   now a ``ValueError`` naming the mesh's available faces on the verb
   production actually binds.

The source-subspace domain honesty note
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One subtlety is worth recording so a future reader does not mistake it for
a bug.  The sweep substrate re-derives the **outflow-definition** rows
(the walk's ``shed`` overwrites :math:`x.{\rm out}` with the streamed
value), so the scheduled walk realizes :math:`M^{-1}` **exactly on the
source subspace** :math:`\{y : y.{\rm outflow\text{-}rows}=0\}` — not on
the whole space.  This is not a limitation: every production rhs lands in
that subspace (:math:`q + S\psi + B_{\rm upper}\psi` all write bulk /
inflow rows only), and its :math:`M`-preimage is the set of
**trace-consistent** states :math:`x.{\rm out}={\rm streamed}(x.{\rm
bulk})` — i.e. actual transport states, which is exactly what a solve
output is.  It is the **same** property the already-landed
:math:`(L+C)`\ ``.solve`` has; the :math:`B` feedback of :math:`M` merely
makes it visible.  The W2-round-trip gate therefore round-trips a
trace-**consistent** state and asserts machine precision on **both** the
bulk and the trace — a *stronger* claim than the bulk-only falsifier, and
one the confused pairing still fails at O(1) (its ``apply`` lacks the
:math:`B_{\rm lower}` subtraction entirely).

Verification — the mutation redefinition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reification changed which mutations have teeth.  The spec's original
**M-SPLIT** mutation ("make the mask disagree with the in-sweep fold")
became **unrepresentable** by construction: the masked :math:`B_{\rm
lower}` single-sources the row split for **both** ``M.apply`` **and** the
in-sweep reflect, so there is no second site to make disagree.  It was
replaced by two mutations, one per gate:

* **M-SPLIT-DIR** — flip the split direction (upper-as-lower) in
  ``lower_inflow_rows``.  The flipped rows are read *before* their face's
  reflect fires, so the in-sweep fold never reaches the reader while
  ``apply`` subtracts it: the W2-round-trip defect returns to O(1)
  (``test_mutation_split_direction_reddens_round_trip``).  This is the
  Mode-1-family ``>`` vs ``<`` convention catcher.
* **M-SPLIT-PART** — doctor one half's rows after the split so the
  partition is no longer complementary: :math:`B \neq B_{\rm lower} +
  B_{\rm upper}` and the W2-split gate reddens
  (``test_mutation_partition_break_reddens_split``).

The converged fixed point is **splitting-invariant** (``vv-principles``
Mode 9): ``test_w2_fixed_point_equivalence_diagonal_cubature`` runs G-S ≡
Jacobi to solver tolerance on a config that **breaks** the degenerate
coincidences — a diagonal (``level_symmetric``) cubature with shared
faces (the ERR-056 regime; an axis-aligned ``product`` quad makes
octant-G-S accidentally exact) on a heterogeneous vacuum-x / reflective-y
box (anisotropic flux; the fully-reflective isotropic box is the Mode-9
degenerate).  That gate is **necessary but not sufficient** — the
load-bearing correctness gate is the W2-round-trip; the FP-invariance
pins only the *splitting* claim (same :math:`\psi^*`, only the rate
differs).  All gates live in ``tests/sn/solve/test_gauss_seidel_reification.py``
(``@pytest.mark.foundation`` — software invariants of the splitting, no
theory-page ``:label:`` and no ``verifies()``).

.. note::

   ⚠ **That sentence needs a qualifier that was missing until 2026-08-15
   (#344), and this gate does not need re-scoping.**  A fixed *point*
   exists only when :math:`A` is nonsingular.  Close :math:`\ge 2`
   reflective axis pairs under diamond differencing and it is **exactly
   singular**, so two correct splittings legitimately return different
   **members** of a solution manifold — ``[M]`` a boundary-G-S and a
   Jacobi trace differing by :math:`0.124184`, with **100.0000 %** of
   the difference inside :math:`\ker A`.  What stays invariant is the
   **bulk**, which is what this gate measures.  Its fixture is
   *vacuum-x / reflective-y*, i.e. **one** reflective axis pair, and
   ``[M]`` :math:`\dim\ker A = 0` there — so the gate is kernel-free
   and its claim is intact.  Full treatment, and the discriminator
   against a genuinely incoherent splitting, at
   :ref:`sn-loss-kernel-gauge`.

.. warning::

   **Honest scope — boundary G-S is NOT a scattering accelerator.**
   The recovery folds **only** the boundary reflection :math:`B`.
   It therefore accelerates the *boundary-layer transient*, NOT the
   dominant flat *scattering* :math:`c`-mode of
   :eq:`si-spectral-rate`.  Whatever it does, it is
   **not** the :math:`c^2`-halving (≈0.5×) one might naively
   expect from "Gauss-Seidel".  The :math:`c^2`-halving is the
   *scattering* Gauss-Seidel result, which does **not** apply to
   boundary-only G-S (the scattering :math:`S` is still fully
   lagged).  The dominant within-group scattering rate is recovered
   ONLY by **Krylov** (already production; rate-optimal,
   splitting-invariant — :math:`n\approx302` vs SI's :math:`n\approx
   860` on the same slab at :math:`\varepsilon=10^{-10}`) or by
   **consistent DSA** (a future feature, GitHub issue #2, with
   :math:`\rho\approx0.22` independent of :math:`c`).  A future
   reader must not mistake boundary-G-S for a scattering accelerator.

.. _sn-boundary-gs-not-regular:

Why it is a splitting but not a *regular* splitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   ⛔ **REFUTED 2026-08-09 (Issue #341).** :eq:`si-gauss-seidel-splitting`
   and the coupled-block :math:`A = M - N` were both described as *the
   regular splitting* — on this page, in
   :doc:`/theory/foundations/coupled_block_operator`, in
   :doc:`/theory/foundations/boundary_conditions`, in
   :ref:`sn-development-history`, and at six code sites.  They are
   **splittings**; neither is a *regular* splitting.  The word is dropped
   everywhere and this section is the one place the reason lives.
   Everything else about the splitting stands — see *What survives* below.

**Regular splitting** is Varga's technical term, and it is load-bearing.
:math:`A = M - N` is *regular* when :math:`M` is nonsingular with
:math:`M^{-1} \ge 0` **and** :math:`N \ge 0`, elementwise.  Its payoff is
the **comparison theorem**: for two regular splittings of the same
:math:`A` with :math:`N_{\rm GS} \le N_{\rm J}` elementwise,

.. math::

   \rho\bigl(M_{\rm GS}^{-1} N_{\rm GS}\bigr)
   \;\le\;
   \rho\bigl(M_{\rm J}^{-1} N_{\rm J}\bigr).

Boundary Gauss-Seidel and Jacobi differ by exactly the folded rows,
:math:`N_{\rm J} - N_{\rm GS} = B_{\rm lower}`, whose entries are the
non-negative specular-reflection weights.  So the comparison theorem is
precisely the statement that boundary G-S is **never slower** than Jacobi
— at every dimension, every octant order, every optical thickness.  That
is the guarantee the word *regular* was silently asserting, and it is the
only reason the word mattered.

It is measurably false.  The table in :ref:`sn-boundary-gs-rate-regime`
reads **1.95× slower** at d=3 all-reflective against **2.5× faster** at
d=2 all-reflective — same solver, same quadrature, same tolerance.  Two
splittings of one :math:`A` cannot straddle a theorem, so one of its
hypotheses must fail, and it is :math:`N \ge 0`.

The obstruction, in the limit where it is exact
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Take zero leakage — every face reflective — where the boundary coupling
*is* the whole iteration and the failure is cleanest.  Specular reflection
at face :math:`(a,\pm)` flips the :math:`a`-th direction cosine, so octant
:math:`s \in \{\pm 1\}^d` feeds only :math:`s \oplus a`: :math:`B`'s
**octant** action is the hypercube :math:`Q_d`, which is a permutation,
and permutations are non-negative.  The negativity is one level down,
*inside* an octant, in the closure itself.

For one ordinate on one source-free homogeneous DD cell write
:math:`w_a = 2|\mu_a| A_a = 2|\mu_a| V/\Delta_a` for the per-axis
streaming weight (this is the :math:`s_a` of :eq:`dd-cartesian-2d`
multiplied through by the cell volume) and
:math:`D = \Sigt{} V + \sum_b w_b` for the DD denominator.  The balance
:math:`\psi_c = \sum_b w_b \psi^{\rm in}_b / D` composed with the diamond
closure :math:`\psi^{\rm out}_a = 2\psi_c - \psi^{\rm in}_a` gives the
face-to-face transmission

.. note::

   **Two symbol overloads, local to this section.**  :math:`A_a` is the
   *area* of the cell face normal to axis :math:`a` — it always carries
   its axis subscript, and it is not the loss operator
   :math:`A = L+C-S-N_{2n}-B` that :math:`A = M - N` splits.  :math:`\Sigma` (no :math:`t`/:math:`s`
   subscript, always with the face indices :math:`a \leftarrow b` or in
   bare matrix form) is the **face-to-face transmission matrix**, not a
   cross section.  Both spellings are kept because they are the ones the
   construction site uses
   (:func:`~orpheus.sn.coupled_system.build_within_group_system`), and
   internal consistency between code and corpus outranks the local
   awkwardness.

.. math::
   :label: dd-face-transmission-spectrum

   \Sigma_{a \leftarrow b}
   \;=\; \frac{\partial \psi^{\rm out}_a}{\partial \psi^{\rm in}_b}
   \;=\; \frac{2 w_b}{D} - \delta_{ab},
   \qquad\text{that is}\qquad
   \Sigma \;=\; \frac{2}{D}\,\mathbf{1}\,\mathbf{w}^{\mathsf T} - I,

.. (vv-status rationale) Structural identity: the linearisation of the
   already-verified multi-D DD closure (dd-cartesian-2d) about a source-free
   cell — an algebraic rearrangement of an equation the sweep and matvec
   gates already pin, not a new solver claim. Its content is the SPECTRUM,
   which is exact for a rank-one-minus-identity matrix and needs no fixture;
   the closure it linearises is gated by the 2-D DD sweep/matvec suites.
.. vv-status: dd-face-transmission-spectrum documented

a **rank-one matrix minus the identity**, so its spectrum is immediate:

.. list-table:: spectrum of the multi-D DD face-to-face transmission
   :header-rows: 1
   :widths: 22 26 14 38

   * - eigenvalue
     - eigenvector
     - multiplicity
     - meaning
   * - :math:`1 - 2\,\Sigt{}V/D`
     - :math:`\mathbf 1` (all faces equal)
     - 1
     - the physical, **absorption-damped** mode
   * - :math:`-1`
     - :math:`\{v : \mathbf{w}^{\mathsf T} v = 0\}`
     - :math:`d-1`
     - :math:`\psi_c = 0 \Rightarrow \psi^{\rm out}_a = -\psi^{\rm in}_a`:
       an **undamped sawtooth**, invisible to :math:`\Sigt{}V\psi_c`

Every :math:`d`-dimensional DD cell therefore carries a
:math:`(d-1)`-dimensional subspace on which transmission is exactly
:math:`-1` and which the collision term cannot see.  A channel with gain
:math:`-1` is not elementwise non-negative, :math:`N \ge 0` fails, and the
comparison theorem has nothing left to stand on.  Note the multiplicity
:math:`d-1`: a 1-D DD cell has **no** such mode, so the obstruction is a
strictly multi-D phenomenon — it could not have been observed before the
schedule reached multi-D.  (That is *not* the reason 1-D falls back to
Jacobi; that fallback is structural — a 1-D scan is not a wavefront, so
there are no octant groups to order.)

The undamped subspace is a property of the DIAMOND closure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not of transport, and not of the boundary condition.  Run the same
construction under **step** differencing (:math:`\psi^{\rm out}_a =
\psi_c`, :math:`w'_a = |\mu_a| A_a`, :math:`D' = \Sigt{}V + \sum_b w'_b`)
and the transmission is :math:`\Sigma_{\rm step} = (1/D')\,\mathbf 1
\mathbf{w}'^{\mathsf T}` — rank one with **no** :math:`-I`, spectrum
:math:`\{(D' - \Sigt{}V)/D'\} \cup \{0\}^{d-1}`.  The same :math:`d-1`
modes are **maximally damped instead of undamped**.  DD's second-order
accuracy and its undamped face sawtooth are one property seen twice: the
closure pins the cell *average* and leaves the face *difference* free.

That is not an abstraction — it is the slow mode measured in the d=3
reflective-absorber budget study behind `Issue #340
<https://github.com/deOliveira-R/ORPHEUS/issues/340>`_.  On the converged
(:math:`n = 1631`) all-reflective pure-absorber state — extents
:math:`(1,2,3)`, cells :math:`(3,4,5)`, :math:`\Sigt{} = (0.8, 1.6)`,
:math:`\Sigma_s = 0`, ``level_symmetric`` :math:`S_4`, ``inner_tol`` 1e-13
— the ``xmin`` trace of one ordinate alternates between ratios
:math:`1.074414` and :math:`0.925586` of the intended uniform value, and
the two sum to :math:`2.000000` **exactly**.  That is the eigenvector's
signature, not a coincidence: :math:`\psi^{\rm in} + \psi^{\rm out} =
2\psi_c` *is* the closure, so a face sawtooth about :math:`\psi^\ast`
leaves every cell average at :math:`\psi^\ast`, the collision term never
sees it, and it is damped only by the weak inter-axis balance mismatch
around the reflective loop.  Hence :math:`\rho \to 1`, and hence 1631
sweeps.

What survives, and what this does not explain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Survives, unchanged.**  :math:`A = M - N` is a perfectly valid
splitting; the drivers consume it exactly as documented; the converged
**bulk** is **splitting-invariant** (any consistent splitting of
:math:`A\psi = q` shares the solution *set* — the FP-invariance gate
above, ``vv-principles`` Mode 9); and the Hackbusch (2016) §11 citation,
which is about **block partitionings**, is untouched.  Nothing in the
construction, the reification, the row-split law or the ERR-056 fan-in
rule depends on regularity.  What is void is one *inference* — that G-S
cannot be slower — and every place that inference was implied by a word
rather than argued.

⛔ **The word "bulk" in that sentence is load-bearing, and it was
"fixed point" until 2026-08-15 (#344).**  A consistent splitting shares
the solution **set**; it shares a *point* only when :math:`A` is
nonsingular, and the very configuration this subsection analyses — every
face reflective, so the undamped mode has nowhere to leak — is the one
where it is not.  The :math:`-1` eigenvalue above is not merely weakly
damped there: closed around a reflective loop its round-trip gain is
exactly :math:`+1`, so :math:`\rho \to 1` understates it and a subspace
of the trace is **frozen** rather than slowly converging — the iteration
converges anyway because that subspace is invisible to the residual.
``[M]`` the boundary-G-S and Jacobi traces differ by :math:`0.124184`
with :math:`\lVert\Pi d\rVert/\lVert d\rVert = 1.000000`.  The bulk
statement is unaffected (the kernel is pure-trace, bulk share
:math:`1.1\times10^{-28}`); see :ref:`sn-loss-kernel-gauge`.

**Does not explain the sign.**  Losing the comparison theorem makes an
inversion *possible*; it does not predict *which side* of the comparison
a given configuration lands on.  That is decided by which edges the fold
takes, and is the subject of :ref:`sn-boundary-gs-rate-regime` below.  Do
not read this section as "boundary G-S is bad at d=3" — read it as
"nothing bounds it either way, so the schedule is a question for
measurement, not for a theorem."

The construction site carries the same warning in prose:
:func:`~orpheus.sn.coupled_system.build_within_group_system` (the one
builder of the production :math:`A = M - N` record), with the selector
:func:`~orpheus.sn.solver._select_si_splitting` and
:class:`~orpheus.sn.loss_representation.sweep_schedule.SweepSchedule`
pointing back to this section.

.. _sn-boundary-gs-rate-regime:

What the boundary-G-S rate actually depends on
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   ⛔ **REFUTED 2026-08-08.** An earlier version of the box above called
   the gain *"a constant, regime-independent* **~0.86–0.92×**\ *"* (citing
   :math:`n_{\rm GS}\approx641` vs :math:`n_{\rm Jacobi}\approx697` on a
   B-2g reflective box).  It is neither constant nor regime-independent,
   and its **sign flips with dimension**.  The refutation is kept beside
   the claim because the *reason* it was wrong is the useful part: a
   single 2-D measurement was generalised to a law.

The predictive statement follows from what the splitting actually folds.
Boundary G-S folds **only** :math:`B`, so its leverage is exactly the
weight of the boundary coupling in the iteration.  That weight is maximal
at **zero leakage** — with every face reflective nothing escapes, so
:math:`B` *is* the whole inter-sweep coupling — and it collapses the
moment any face is vacuum, because the escaping fraction is not iterated
at all.

`[M]` 2026-08-08, SI sweeps to ``inner_tol = 1e-13``, level-symmetric
:math:`S_4`, 2 groups, extents :math:`(1,2)` / :math:`(1,2,3)`, cells
:math:`(3,4)` / :math:`(3,4,5)`.  Probe:
``scratch/probe_gs_vs_jacobi_rate.py``; its first row is a **control**
reproducing, to the sweep, the independently measured 1631 of the d=3
reflective budget study (``scratch/d3_absorber_diagnosis.md``, pinned by
``derivations/diagnostics/diag_d3_absorber_02_si_rate_scaling.py``) —
without that control the table below would be one more unverified
instrument.

.. list-table:: :math:`n_{\rm GS}` vs :math:`n_{\rm Jacobi}`
   :header-rows: 1
   :widths: 42 14 14 12 18

   * - configuration
     - G-S
     - Jacobi
     - ratio
     - verdict
   * - d=2 all-reflective, pure absorber
     - 258
     - 648
     - **0.40**
     - G-S 2.5× faster
   * - d=2 all-reflective, :math:`c = 0.5`
     - 259
     - 645
     - **0.40**
     - G-S 2.5× faster
   * - d=3 all-reflective, pure absorber
     - 1631
     - 838
     - **1.95**
     - G-S ~2× SLOWER
   * - d=3 all-reflective, :math:`c = 0.5`
     - 1598
     - 832
     - **1.92**
     - G-S ~2× SLOWER
   * - d=2, one vacuum axis, :math:`c = 0.5`
     - 34
     - 35
     - 0.97
     - a wash
   * - d=3, one vacuum axis, :math:`c = 0.5`
     - 208
     - 214
     - 0.97
     - a wash
   * - d=3, two vacuum axes, :math:`c = 0.5`
     - 33
     - 33
     - 1.00
     - a wash

⚠ **The magnitudes above are fixture-specific; only the SIGN and the
leakage-dependence are robust.**  The gate
``tests/sn/verification/analytical/test_si_convergence_rate.py::
test_boundary_gs_recovers_reflective_2d_si`` measures a *second* d=2
zero-leakage point — B-2g, :math:`8\times8`, ``product(2,4)`` — and gets
:math:`641/697 = 0.92`, against :math:`0.40` here.  Same sign, same
regime, magnitude differing by more than 2×.  That is precisely why that
gate asserts only the strict inequality :math:`n_{\rm GS} < n_{\rm
Jacobi}` and not a ratio: **the inequality is the law, the ratio is a
fixture reading.**  Any future gate on this effect should do the same.

Three readings, in order of how much they should change what you do:

#. **With any leakage the choice is immaterial** (0.97–1.00 at every
   dimension).  Since essentially every *physical* configuration leaks
   somewhere, the default rarely matters in production; it matters in
   the all-reflective verification fixtures.
#. **At zero leakage the splitting matters a lot, and the sign depends on
   dimension** — a 2.5× win at d=2, a ~2× loss at d=3.  An all-reflective
   d=3 box is the one configuration where ``inner_schedule="jacobi"`` is
   worth asking for explicitly.
#. **Scattering does not change the picture**: the :math:`c = 0.5` rows
   track the pure-absorber rows to within 2 %.  That is exactly what the
   "G-S touches only :math:`B`" scope claim predicts, so this measurement
   *confirms* the honest-scope box even while refuting its number.

.. warning::

   ⛔ **REFUTED 2026-08-09 (Issue #341) — reading 2's second clause.**
   *"The sign depends on dimension"* is **not** a law; :math:`d` is not
   the discriminating variable, and the reading above is kept verbatim
   only because the way it was minted is the point (see the box below).
   Readings 1 and 3, and every number in the table, stand.

   `[M]` 2026-08-09, same probe construction as the table (SI sweeps to
   ``inner_tol = 1e-13``, ``max_inner = 20000``, ``level_symmetric``
   :math:`S_4`, 2-group pure absorber :math:`\Sigt{} = (0.8, 1.6)`,
   :math:`\Sigma_s = 0`, all faces reflective, flat source
   :math:`Q = (1.0, 0.5)/\!\sum_n \mathcal W_n`), varying **only** the
   mesh — with the table's own d=3 row re-run first as a control:

   .. list-table:: two **d=2** zero-leakage fixtures where boundary G-S LOSES
      :header-rows: 1
      :widths: 46 14 14 12 14

      * - configuration
        - G-S
        - Jacobi
        - ratio
        - verdict
      * - d=3 extents (1,2,3), cells (3,4,5) — **control**
        - 1631
        - 838
        - 1.95
        - reproduces the table row exactly
      * - d=2 extents (1,2), cells (1,1)
        - 202
        - 38
        - **5.32**
        - G-S 5.3× SLOWER *at d=2*
      * - d=2 extents (6,6), cells (2,2)
        - 54
        - 47
        - **1.15**
        - G-S slower *at d=2*

   The first d=2 row is a **worse** loss than the d=3 row the dimension
   story was built on, at the dimension the story calls a win — and the
   **mesh alone**, at fixed cross sections, quadrature and tolerance,
   moved the d=2 ratio from :math:`1.15` to :math:`5.32`, straddling and
   then exceeding the d=3 value.  One counterexample is enough to kill a
   law; three say :math:`d` was merely correlated with whatever the real
   variable is on the fixtures first measured.  A natural place to look
   is the per-cell :math:`\Sigt{}V/D`, the *only* parameter in the damped
   eigenvalue of :eq:`dd-face-transmission-spectrum` and a quantity the
   mesh moves directly — but that is a **hypothesis, not measured here**.
   What is settled: **do not branch a production default on** ``ndim``.

⚠ The **sign** at zero leakage is measured but still not *predicted*.
What #341 did establish is the **structural obstruction**
(:ref:`sn-boundary-gs-not-regular`): the splitting is not *regular*, so
no comparison theorem bounds the two rates and an inversion is
**permitted** in either direction.  What remains open is which side a
given configuration lands on — a plausible-sounding story (more octant
groups ⟹ a longer serial chain in the forward substitution) is *not*
verified, and the naive expectation runs the other way.  Do not repeat
the mistake this box exists to record: **do not promote a mechanism to a
law on one measurement** — the refuted reading above is the second time
that mistake was made on this very effect.  Tracked as `Issue #341
<https://github.com/deOliveira-R/ORPHEUS/issues/341>`_.

The diagonal-cubature shared-face rule (ERR-056)
---------------------------------------------------

The Gauss-Seidel schedule must assign each reflective face to the
**LAST** octant group (in sweep order) that outflows through it —
NOT the first.  This is a **correctness** requirement, not merely a
rate optimisation, and the distinction is invisible on axis-aligned
quadratures:

* On an **axis-aligned** quadrature
  (:meth:`Quadrature.product <orpheus.numerics.quadrature.Quadrature.product>`
  — each octant outflows a single face), every reflective face is
  outflowed by exactly **one** octant group, so "last" trivially
  equals "the only one".
* On a **diagonal / spherical** cubature
  (:meth:`Quadrature.lebedev <orpheus.numerics.quadrature.Quadrature.lebedev>`,
  :meth:`level_symmetric <orpheus.numerics.quadrature.Quadrature.level_symmetric>`
  — each octant outflows **two** in-plane faces), a face is shared
  by :math:`\ge 2` octant groups (e.g. ``xmax`` is outflowed by
  every ``+x`` octant: :math:`(+1,0)`, :math:`(+1,+1)`,
  :math:`(+1,-1)`).

Reflecting a shared face after only the **first** outflowing group
absorbs the *not-yet-swept* octants' slots — and because the interior
cochain :math:`C^1_{\rm int}` (carried by the rolling
``_MovingFrontier``) is rebuilt and :math:`\iota_*`-seeded each
``.solve``, those slots still hold the **inflow seed**, not real
outflow.  The reflect then
propagates garbage and the iteration converges to the fixed point of
the **wrong** operator (ERR-056).  Crucially, this seed-contamination
does **not** self-correct at convergence — unlike a lagged-but-real
Jacobi coupling, which reads the previous iterate's genuine value.
Deferring the reflect to the **last** outflowing group guarantees
the face's outflow is complete before it is reduced; octants reading
the face that are swept *before* its reflect simply keep the lagged
seed (the valid cyclic back-edge → partial one-pass G-S).  This is
the general principle that *a face shared by multiple work-units
must be reduced only after the last contributing unit completes* —
the same fan-in discipline KBA wavefront scheduling
(:cite:`Pautz2002`; Adams & Larsen 2002 §VI on parallel sweeps) and
multigroup Gauss-Seidel over shared down-scatter targets require.
The full post-mortem (symptom, root cause, the
``test_gs_diagonal_quadrature_shared_face_assigned_to_last_group_only``
structural pin) is catalogued as **ERR-056**.

White and vacuum faces are **excluded** from the reflective set:
vacuum has no coupling (:math:`B=0`), and white reflection couples
*all* ordinates on a face, so the octant-order G-S degenerates to
Jacobi anyway — only **specular** reflection admits the
order-respecting forward-substitution acceleration.

Numerical evidence
--------------------

All measured 2026-06-05 on this branch
(:func:`tests.sn.verification.analytical.test_si_convergence_rate`,
GL/``product`` :math:`N=8`, ``inner_tol`` as noted).  The Jacobi and
Gauss-Seidel counts are compared **in-process** (no hardcoded
baseline — Jacobi is a permanent live control, so the gates cannot
go stale).

.. list-table:: SI iteration counts — boundary G-S recovery vs the splitting-invariant controls
   :header-rows: 1
   :widths: 34 13 13 13 27

   * - Configuration
     - :math:`n_{\rm Jacobi}`
     - :math:`n_{\rm GS}`
     - ratio
     - Notes
   * - B-2g reflective **box** (2-D, ``inner_tol`` 1e-8)
     - 697
     - 641
     - 0.92×
     - Same **bulk** fixed point: scalar-flux rel-L\ :sub:`∞`
       :math:`4.86\times10^{-8}` (rate-only, ``vv-principles``
       Mode 9).  The box is all-reflective, so the *trace* agrees
       only up to :math:`\ker A` (:ref:`sn-loss-kernel-gauge`).
   * - B-2g reflective **slab** (1-D, ``inner_tol`` 1e-8)
     - 655
     - 655
     - 1.00×
     - **No-op** by design — the 1-D scan is not a wavefront;
       ``_select_si_splitting`` falls back to Jacobi.
   * - B-2g **vacuum** slab (G-4 negative control)
     - 128
     - 128
     - 1.00×
     - :math:`B=0` ⟹ the schedule is inert; proves the recovery
       touches *only* reflective coupling.
   * - B-2g reflective slab, **Jacobi vs Krylov**
       (``inner_tol`` 1e-10)
     - ≈860
     - — (302)
     - 2.85×
     - The splitting-invariant **Krylov** floor: an SI splitting can
       never beat it (the rate-optimal anchor, not the target).

The analytic Jacobi anchor (:eq:`si-spectral-rate`) predicts
:math:`\log(10^{-8})/\log(0.975) = 728` for the B-2g slab; the
measured 655 gives ratio **0.90** — the finite-slab leakage +
multigroup correction discussed with the SI-rate derivation
(:doc:`slab_one_group`).  The
**eigenvalue** path surfaces the analogous measurand via
:attr:`IterationHistory.total_inner_iterations
<orpheus.sn.solution.IterationHistory.total_inner_iterations>`
(the Phase 3 measurement seam): A-2g reflective :math:`n=10` gives
SI ``total_inner`` 371, Krylov 310 (with :math:`n_{\rm outer}=3` for
both — the **outer** count is splitting-invariant; the inner SI count
is where the recovery shows).

For comparison, a clean 1-D textbook DSA spike (the future issue-#2
target) gives **8–21×** :math:`c`-independent speed-up on a vacuum
slab (:math:`\rho\approx0.22`), but a *naive* finite-difference DSA
**diverges** on a reflective boundary — the concrete confirmation
that DSA needs the consistent diffusion operator the σ\ :sub:`r`-fold
trap (:doc:`slab_one_group`) already implied.


.. _sn-loss-kernel-gauge:

The loss operator is SINGULAR on a closed reflective box
=========================================================

The previous section closed on a *local* fact and used it only
negatively: every multi-D diamond cell carries a
:math:`(d-1)`-dimensional face subspace on which transmission is exactly
:math:`-1` (:eq:`dd-face-transmission-spectrum`), invisible to
:math:`\Sigt{}V\psi_c`, so :math:`N \ge 0` fails and no comparison
theorem bounds the schedule.

Close the mesh on itself and the same mode becomes something stronger
than an obstruction to a rate proof.  With **at least two axes
reflective at both ends** the sawtooth has nowhere to leak: it returns
to itself, its round-trip gain is exactly :math:`+1`, and the slow mode
of :ref:`sn-boundary-gs-rate-regime` becomes an **exact null vector**.
On any :math:`d \ge 2` Cartesian diamond-difference mesh with
:math:`\ge 2` reflective axis pairs the within-group loss operator
(:eq:`sn-within-group-with-n2n`)

.. math::

   A \;=\; L + C - S - N_{2n} - B

is **exactly singular**.  The kernel is a *pure-trace* object (the bulk
share of the null projector is :math:`1.1\times10^{-28}`, measured
below), so both bulk gains — :math:`S` and :math:`N_{2n}` — annihilate
it: each is an interior-only body lifted to the composite by
**extension-by-zero on the trace**
(:func:`~orpheus.transport.operators.lift.lift_bulk_action`), so it reads
a bulk-zero mode as zero and writes nothing to the trace.  The closed
form derived here is therefore a property of the diamond closure and the
reflective faces alone.  ``Aψ = q`` then has a solution *manifold*, not
a solution, and a converged solve returns whichever member the iteration
happened to freeze — a function of the cold start and of the schedule,
not of the problem.  That is not a corner case: :func:`~orpheus.sn.solver.solve_sn`
has no ``boundary_condition`` parameter at all and a bare
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` resolves to all-reflective,
so this **is** the standard :math:`\kinf` lattice.

Three facts make it tractable rather than alarming, and this section is
their derivation, their evidence and their remedy:

#. the kernel is a **pure-trace** object with a **closed form** — no
   eigensolve and no SVD of :math:`A`;
#. every mirror-**even** functional annihilates it *by theorem*, so
   :math:`\keff`, the scalar flux, every reaction rate and every normal
   current are untouched — and so is the exact solution, which is why
   the minimum-norm member **is** the physical answer;
#. what is *not* untouched is the one class of functional that is
   mirror-**odd**, and there the un-gauged solve reports a quantity that
   cannot physically exist.

.. admonition:: Key result
   :class: tip

   ``[M]`` all-reflective 2-D box, extents :math:`(1, 2)`, cells
   :math:`(3,4)`, 2-group pure absorber :math:`\Sigt{} = (0.8, 1.6)`,
   :math:`\Sigma_s = 0`, ``level_symmetric`` :math:`S_4`, **uniform
   isotropic source** :math:`q_n = 1/W` (so the exact flux is flat and
   *every* current is zero), SI + boundary Gauss-Seidel,
   ``inner_tol = 1e-13``.

   The returned boundary trace is :math:`6.09\times10^{-2}` (relative)
   away from the analytic answer, and **100.00000000 %** of that
   deviation lies in :math:`\ker A`.  It surfaces as a
   :math:`+7.381060\times10^{-2}` net current running **sideways along a
   mirror face**.  Projecting the kernel component out recovers the
   analytic trace to :math:`1.04\times10^{-13}` relative and drops the
   tangential current to :math:`\sim10^{-14}`, while the normal
   currents, :math:`\keff`, and every summed face moment do not move.

From an undamped face mode to a null vector of A
-------------------------------------------------

``[M]`` the bulk share of the null projector is :math:`1.1\times10^{-28}`,
so set :math:`\psi_c \equiv 0` and read off what survives of the diamond
closure and the cell balance in a source-free cell:

.. math::

   \psi_{{\rm out},a} = 2\psi_c - \psi_{{\rm in},a}
   \;\;\xrightarrow{\;\psi_c = 0\;}\;\;
   \psi_{{\rm out},a} = -\psi_{{\rm in},a} ,
   \qquad
   \sum_a |\mu^n_a| \, A_a(i_\perp) \, \psi^n_{{\rm in},a}(i) = 0 .

The first relation is **direction-independent** — :math:`s_a = +1` reads
it left to right and :math:`s_a = -1` right to left, and it is the same
relation — so along every mesh line the face values simply alternate and
the whole face field on axis :math:`a` collapses to ONE function of the
transverse index, the **sawtooth**

.. math::
   :label: dd-null-sawtooth

   \psi^n_a(k, i_\perp) \;=\; (-1)^k \, \varphi^n_a(i_\perp) ,


.. implements:: dd-null-sawtooth
   :by: orpheus.sn.operators.loss_kernel_gauge._FacePlacement

   **Implemented by** 3 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: dd-null-sawtooth
   :by: orpheus.sn.operators.loss_kernel_gauge._block_support

.. implements:: dd-null-sawtooth
   :by: orpheus.sn.operators.loss_kernel_gauge._build_block_table

.. (vv-status rationale) Structural identity: the null-space specialisation of
   the already-verified multi-D DD closure (dd-cartesian-2d) at psi_c = 0 — an
   algebraic rearrangement, not a new solver claim. Its content is that every
   null vector has this SHAPE, which is asserted end-to-end by the foundation
   suite tests/sn/operators/test_loss_kernel_gauge.py: the constructed basis is
   annihilated by the PRODUCTION matvec
   (test_EVERY_basis_vector_is_annihilated_by_the_production_matvec) and its
   dimension equals a dense SVD of the assembled operator
   (test_the_dimension_matches_a_DENSE_SVD_of_the_assembled_operator).
.. vv-status: dd-null-sawtooth documented

with :math:`k \in \{0, \dots, n_a\}` the face index along axis :math:`a`
and :math:`i_\perp` the transverse cell index.  **This is why the kernel
is a trace object**, and why the two boundary closures reduce to one
condition each: specular on axis :math:`a` gives
:math:`\varphi^n_a = \varphi^{R_a n}_a` (both faces give the *same*
condition — the far face's :math:`(-1)^{n_a}` cancels between the two
sides), and vacuum gives :math:`\varphi^n_a \equiv 0`.

One substitution empties the balance of all physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert :eq:`dd-null-sawtooth` into the balance.  The inflow face index
is :math:`k = i_a` for :math:`\mu_a > 0` and :math:`k = i_a + 1` for
:math:`\mu_a < 0`, so
:math:`\psi^n_{{\rm in},a}(i) = s_a (-1)^{i_a} \varphi^n_a(i_\perp)` with
:math:`s_a = \operatorname{sign}\mu^n_a`.  Multiply through by
:math:`(-1)^{\sum_b i_b}` and absorb every coefficient into

.. math::

   Y^n_a(i_\perp) \;:=\;
   |\mu^n_a| \; A_a(i_\perp) \; (-1)^{\sum_{b \neq a} i_b}
   \; \varphi^n_a(i_\perp) .

Every factor on the right depends only on :math:`i_\perp` — the face
area :math:`A_a(i_\perp) = \prod_{b \neq a} h_b(i_b)` included, **which
is why a graded mesh needs no separate treatment**.  What is left is

.. math::
   :label: dd-null-balance-combinatorial

   \sum_{a \in S} s_a \, Y_a\bigl(s_{\neq a} ;\, i_{\neq a}\bigr)
   \;=\; 0 ,

.. (vv-status rationale) Structural identity: the substitution image of the
   cell balance under dd-null-sawtooth — a change of variables, not a new
   solver claim. Its consequences ARE gated by the foundation suite
   tests/sn/operators/test_loss_kernel_gauge.py, which asserts that the basis
   built from this equation's solution set is annihilated by the production
   matvec and spans a dense SVD's null space; the physics-independence it
   predicts is pinned by test_the_basis_never_reads_a_CROSS_SECTION.
.. vv-status: dd-null-balance-combinatorial documented

where

* :math:`S` is the set of axes that are **simultaneously reflective and
  non-tangential** for that ordinate — a vacuum axis drops out because
  its :math:`\varphi \equiv 0`, a tangential axis because its
  :math:`|\mu_a|` is zero;
* :math:`s_{\neq a}` and :math:`i_{\neq a}` are the ordinate's signs and
  the cell index on the *other* axes: :math:`Y_a` is blind to
  :math:`i_a` (it is a function on the face) and blind to :math:`s_a`
  (that is exactly the specular condition).

Read it in words: *a sum of functions, each blind to one coordinate and
one sign, vanishing identically.*  The cross sections, the mesh
spacings, the quadrature weights and the scattering ratio have all
cancelled.  That is the structural reason the dimension is
mesh-independent at :math:`d = 2`, independent of :math:`c`, and exactly
proportional to :math:`n_g`; ``[M]`` an absorber and a fissile mixture
on the same box give **bit-identical** residuals
(:math:`2.799\times10^{-16}`), which is what makes the kernel a
**Stratum-1** geometry-only object, cached once per mesh on
:attr:`SNMesh.loss_kernel_gauge <orpheus.sn.mesh.augmented_mesh.SNMesh.loss_kernel_gauge>`
and reused across every group, every outer and every eigenvalue iterate.

Both counting laws, as theorems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expanding each :math:`Y_a` in the sign characters
:math:`\chi_T(s) = \prod_{b \in T} s_b` splits
:eq:`dd-null-balance-combinatorial` into one independent equation per
subset :math:`U \subseteq S`.  :math:`|U| \le 1` contributes nothing;
:math:`|U| \ge 2` is the classical additive-separable (ANOVA) problem
whose solution space has dimension

.. math::

   \kappa(U) \;=\;
   \sum_{a \in U} \prod_{b \in U \setminus a} n_b
   \;-\; \prod_{b \in U} n_b
   \;+\; \prod_{b \in U} (n_b - 1)
   \qquad\Longrightarrow\qquad
   \begin{cases}
     \kappa(\{a,b\}) = 1, \\[2pt]
     \kappa(\{a,b,c\}) = n_a + n_b + n_c - 1 .
   \end{cases}

Summing over :math:`U`, over the free spectator axes and over the
ordinate **orbits** under the reflection group
:math:`\langle R_a : a \text{ reflective}\rangle`:

.. math::
   :label: dd-null-counting-law

   \dim \ker A
   \;=\;
   n_g \sum_{\rm orbits} \;
   \sum_{\substack{U \subseteq S({\rm orbit}) \\ |U| \ge 2}}
   \kappa(U) \prod_{c \notin U} n_c
   \;+\; \#\{\text{tangential trace DOFs}\}


.. implements:: dd-null-counting-law
   :by: orpheus.sn.operators.loss_kernel_gauge._anova_dimension

   **Implemented by** 2 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: dd-null-counting-law
   :by: orpheus.sn.operators.loss_kernel_gauge.predicted_kernel_dimension

.. (vv-status rationale) Structural identity: a combinatorial count derived
   from dd-null-balance-combinatorial, carrying no solver claim. It is
   nonetheless doubly gated by the foundation suite
   tests/sn/operators/test_loss_kernel_gauge.py — the law is evaluated without
   building a vector by predicted_kernel_dimension and compared BOTH against the
   rank the construction's SVD finds
   (test_the_dimension_matches_the_combinatorial_counting_law) and against a
   dense SVD of the assembled operator
   (test_the_dimension_matches_a_DENSE_SVD_of_the_assembled_operator), with the
   two closed-form specialisations pinned separately
   (test_the_counting_law_reproduces_the_two_closed_form_specialisations).
.. vv-status: dd-null-counting-law documented

and the specialisations fall out.  Note the **orbit** count: at
:math:`d = 2` the group is :math:`\langle R_x, R_y\rangle` of order 4,
:math:`\mu_z` is *not* flipped (z is not a mesh axis), so the count uses
the full 3-D ordinate count :math:`N` and reads :math:`N/4`.

.. list-table:: ``dim ker A``, closed form vs the construction, measured 2026-08-15
   :header-rows: 1
   :widths: 34 22 11 11 22

   * - configuration (all 2-group unless noted)
     - quadrature
     - built
     - law
     - closed form
   * - :math:`d=2` :math:`(3,4)`, all reflective
     - ``level_symmetric(4)``
     - 12
     - 12
     - :math:`n_g N/4`
   * - :math:`d=2` :math:`(5,6)`, all reflective
     - ``level_symmetric(4)``
     - 12
     - 12
     - mesh-**independent**
   * - :math:`d=2` :math:`(3,4)` **graded** (non-uniform :math:`h`)
     - ``level_symmetric(4)``
     - 12
     - 12
     - the area cancels
   * - :math:`d=2` :math:`(3,4)`, all reflective
     - ``lebedev(11)``
     - 18
     - 18
     - (+ 224 in **T**)
   * - :math:`d=2` :math:`(3,4)`, all reflective
     - ``product(4,4)``
     - **0**
     - 0
     - pure **T** (224)
   * - :math:`d=2` :math:`(3,4)`, x-**vacuum** / y-reflective
     - ``level_symmetric(4)``
     - **0**
     - 0
     - one pair is not enough
   * - :math:`d=2` :math:`(3,4)`, **linear-discontinuous** closure
     - ``level_symmetric(4)``
     - **0**
     - —
     - the closure damps it
   * - :math:`d=3` :math:`(2,2,2)`, all reflective
     - ``level_symmetric(4)``
     - 66
     - 66
     - :math:`n_g (N/8)(2\textstyle\sum n - 1)`
   * - :math:`d=3` :math:`(3,4,5)`, all reflective
     - ``level_symmetric(4)``
     - 138
     - 138
     - :math:`2\cdot3\cdot23`
   * - :math:`d=3` :math:`(3,4,5)`, z-vacuum
     - ``level_symmetric(4)``
     - 60
     - 60
     - :math:`n_g (N/4)\, n_c`
   * - :math:`d=3` :math:`(3,4,5)`, x-vacuum
     - ``level_symmetric(4)``
     - 36
     - 36
     - :math:`n_c = 3`
   * - :math:`d=1` :math:`(8,)`, reflective
     - ``level_symmetric(4)``
     - **0**
     - 0
     - :math:`|S| \le 1`, no :math:`U`

The two columns are computed by structurally different routes and that
is the point: ``law`` is
:func:`~orpheus.sn.operators.loss_kernel_gauge.predicted_kernel_dimension`,
which walks the combinatorics of :eq:`dd-null-counting-law` and **builds
no vector at all**, while ``built`` is the rank the construction's
weighted SVD finds.  A bookkeeping error in either would separate them.
The small rows are additionally pinned against a dense SVD of the
assembled :math:`A`; the large ones are not, because they cannot be —
the dense factorisation is :math:`O(n_{\rm dof}^3)` and already costs
``[M]`` **23 s** at :math:`n_{\rm dof} = 3744` against a closed-form
build the driver reports as ``0.00 s``.

The full construction — the **pair generators** that span the solution
set of :eq:`dd-null-balance-combinatorial`, the blocked direct sum over
(ordinate orbit, group) that keeps a :math:`(12,12,12)` :math:`S_8`
:math:`n_g{=}4` projector at ``[M]`` **150 MiB** where a dense basis
would be **17.6 GiB**, and the single :math:`\sqrt G`-weighted SVD that
does the rank reduction and the :math:`G`-orthonormalisation together —
is the derivation of record in the module docstring of
:mod:`~orpheus.sn.operators.loss_kernel_gauge` and is not repeated here.

.. _sn-loss-kernel-what-a-user-sees:

What a user actually sees
--------------------------

On the Key-result fixture the exact flux is flat, so **every** current
component is zero everywhere.  The solver already gets the *normal*
currents right.  What it reports un-gauged is a net current running
**parallel to a mirror surface** — a quantity that cannot exist.

.. list-table:: :math:`J_b = \sum_n w_n \mu_b \psi_n` summed over each face, un-gauged :math:`\to` gauged
   :header-rows: 1
   :widths: 22 16 31 31

   * - quadrature
     - face
     - :math:`J_{\rm tangential}`
     - :math:`J_{\rm normal}` (control)
   * - ``level_symmetric(4)``
     - ``ymin``
     - :math:`+7.381060\times10^{-2} \to +1.55\times10^{-14}`
     - :math:`-2.00\times10^{-15}`, unmoved
   * - ``level_symmetric(4)``
     - ``ymax``
     - :math:`+7.381060\times10^{-2} \to -1.57\times10^{-14}`
     - :math:`+1.08\times10^{-14}`, unmoved
   * - ``level_symmetric(4)``
     - ``xmin`` / ``xmax``
     - :math:`-4.44\times10^{-16}`, unmoved
     - :math:`0.0` / :math:`-7.27\times10^{-15}`, unmoved
   * - ``lebedev(11)``
     - ``ymin``
     - :math:`+5.650846\times10^{-2} \to +1.94\times10^{-15}`
     - :math:`-3.33\times10^{-16}`, unmoved
   * - ``lebedev(11)``
     - ``ymax``
     - :math:`+5.650846\times10^{-2} \to -1.82\times10^{-14}`
     - :math:`+5.27\times10^{-15}`, unmoved

The normal currents are not decoration: they are the **negative
control**.  Without them the gate would be satisfied by any
transformation that merely shrinks the trace.

.. _sn-loss-kernel-blindness:

Why no scalar summary reveals it — a blindness theorem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every generator of the kernel carries a **non-trivial sign character on
every axis it touches**: its :math:`Y_a \propto \chi_{U\setminus\{a\}}(s)`
with :math:`U \setminus \{a\}` non-empty, because it always contains the
partner axis.  Over one ordinate orbit the quadrature weight :math:`w_n`
and the cosine magnitudes :math:`|\mu_a|` are **constant**, so for any
angular weight :math:`F` that is *even* under the reflection group

.. math::
   :label: sn-kernel-mirror-blindness

   \sum_{n \in {\rm orbit}} F(n)\, \chi_{U \setminus \{a\}}\bigl(s(n)\bigr)
   \;=\; 0
   \qquad\text{exactly, at every mesh, order and quadrature.}

.. (vv-status rationale) Structural identity: a character-orthogonality
   statement over a finite reflection group, carrying no solver claim — it says
   which functionals CANNOT see the kernel, and it is what makes
   psi_exact G-orthogonal to ker A (hence the minimum-norm gauge canonical
   rather than conventional). Both faces are gated by the foundation suite
   tests/sn/solve/test_every_entry_gauges_its_trace.py: the mirror-ODD
   tangential current collapses while the mirror-EVEN normal currents do not
   move (test_the_spurious_TANGENTIAL_current_along_a_mirror_is_gone), and
   keff is unchanged against an independent analytic k_inf anchor
   (test_the_bulk_and_keff_are_untouched).
.. vv-status: sn-kernel-mirror-blindness documented

So **every mirror-even functional annihilates the whole R component of**
:math:`\ker A` — not below a tolerance, not at this order: exactly.  A
functional summed over a face is mirror-even.  Measured on the
Key-result fixture, un-gauged versus gauged:

.. list-table:: mirror-EVEN functionals are bit-invariant under the gauge, while the trace moves 6.08 %
   :header-rows: 1
   :widths: 30 24 23 23

   * - functional
     - parity
     - ``level_symmetric(4)``
     - ``lebedev(11)``
   * - :math:`|\Omega\cdot n|^0` face moment (4 faces)
     - EVEN
     - :math:`0.0` on 3, :math:`8.9\times10^{-16}` on 1
     - :math:`0.0` on all 4
   * - outgoing partial current :math:`J^+` (4 faces)
     - EVEN
     - :math:`0.0` on all 4
     - :math:`0.0` on 3, :math:`2.2\times10^{-16}` on 1
   * - :math:`G`-weighted total over :math:`\Gamma`
     - EVEN
     - :math:`3.6\times10^{-15}`
     - :math:`3.6\times10^{-15}`
   * - :math:`\keff` (eigenvalue entry)
     - EVEN
     - unchanged, :math:`= 1.875` to :math:`10^{-9}`
     - —
   * - :math:`J_{\rm tangential}` on a mirror face
     - **ODD**
     - :math:`7.4\times10^{-2} \to 10^{-14}`
     - :math:`5.7\times10^{-2} \to 10^{-14}`

The theorem also settles *why the gauge is canonical rather than
conventional*.  :math:`\psi_{\rm exact}` on this configuration is a
constant — the most mirror-even state there is — so it is
:math:`G`-orthogonal to :math:`\ker A` (``[M]`` :math:`1.27\times10^{-15}`;
directly, the projector annihilates a uniform trace at ``[M]``
:math:`5.0\times10^{-18}` relative).  The **minimum-**\ :math:`\lVert\cdot\rVert_G`
member of the solution manifold therefore *is* the physical answer, and
the projection **recovers** it rather than choosing among equals.

.. warning::

   ⚠ **Two limits on the safe list, and both are measured.**

   *First*, the theorem covers component **R**.  The tangential
   component **T** — trace slots with :math:`\Omega\cdot n = 0`, whose
   rows *and* columns of :math:`A` are identically zero — is a
   different object, and there the :math:`|\Omega\cdot n|^0` face moment
   is **NOT** blind (``[M]`` :math:`2.99\times10^{-2}` on
   ``lebedev(11)``).  Any functional carrying at least one power of
   :math:`|\Omega\cdot n|` kills **T**; the *unweighted* angular
   integral at a face — a half-range scalar flux, a face-averaged
   reaction rate — does not.  ⟹ the honest safe condition is
   **mirror-even in the angle AND carrying** :math:`\ge 1` **power of**
   :math:`|\Omega\cdot n|`, i.e. a current-type functional.  On the
   default ``level_symmetric`` path ``[M]`` :math:`\dim T = 0` and
   :math:`p = 0` is safe; on ``product`` and ``lebedev`` it is not
   (``[M]`` :math:`\dim T = 224` on both, at :math:`(3,4)`
   :math:`n_g{=}2`).

   *Second*, an odd weight is not automatically a **sighted** test: on
   the :math:`a`-face the mode carries character
   :math:`\chi_{U\setminus\{a\}}`, and a weight pairs non-zero only when
   its own character matches.  ``[M]`` a :math:`\operatorname{sign}(\mu_x\mu_y)`
   weight reads :math:`1.9\times10^{-18}` (blind) where
   :math:`\operatorname{sign}(\mu_x)` reads :math:`4.4\times10^{-2}`.
   The reliably sighted probes are a single-ordinate reading and the
   tangential current.

.. _sn-loss-kernel-parity:

The excitation is a PARITY effect — and it is not a property of the mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A singular operator is a statement about the *equation*.  Whether a
given solve actually **lands** off the canonical member is a separate
question, and its answer is the sharpest argument for warning at run
time rather than documenting and moving on.

``[M]`` 2026-08-15, :func:`~orpheus.sn.solver.solve_sn`, all-reflective
2-D box extents :math:`(1,2)`, 2-group **fissile** mixture,
``level_symmetric`` :math:`S_4`, SI + ``gauss_seidel``,
``inner_tol = 1e-13``; the reading is
:math:`\lVert\Pi t\rVert / \lVert t\rVert` on the **returned** trace:

.. list-table:: the same physics, eleven meshes
   :header-rows: 1
   :widths: 14 12 14 30 30

   * - cells
     - :math:`n_x`
     - :math:`n_y`
     - ``gauss_seidel``
     - ``jacobi`` (control)
   * - :math:`(3,3)`
     - odd
     - odd
     - :math:`6.822354\times10^{-2}`
     - :math:`\sim10^{-15}`
   * - :math:`(3,4)`
     - odd
     - even
     - :math:`6.080483\times10^{-2}`
     - :math:`1.17\times10^{-15}`
   * - :math:`(3,2)`
     - odd
     - even
     - :math:`7.824118\times10^{-2}`
     - :math:`\sim10^{-15}`
   * - :math:`(5,4)`
     - odd
     - even
     - :math:`4.436713\times10^{-2}`
     - :math:`\sim10^{-15}`
   * - :math:`(5,5)`
     - odd
     - odd
     - :math:`4.099523\times10^{-2}`
     - :math:`1.11\times10^{-15}`
   * - :math:`(4,3)`
     - even
     - odd
     - :math:`1.01\times10^{-14}`
     - :math:`\sim10^{-15}`
   * - :math:`(4,4)`
     - even
     - even
     - :math:`5.05\times10^{-15}`
     - :math:`\sim10^{-15}`
   * - :math:`(4,5)` / :math:`(2,3)` / :math:`(2,2)` / :math:`(6,5)`
     - even
     - —
     - :math:`2.1\times10^{-15} \dots 7.6\times10^{-15}`
     - :math:`\sim10^{-15}`

Every row returns :math:`\keff = 1.8750000000`, the analytic
:math:`\kinf = \nu\Sigma_f/\Sigma_a` — the blindness theorem,
end to end.

⟹ **on these symmetric fixtures the excitation is present iff the FIRST
axis has an ODD cell count**; :math:`n_y`'s parity is irrelevant, and
the two populations are separated by **13 orders**, so any warning
threshold in :math:`[10^{-12}, 10^{-4}]` gives the same verdict.  The
mechanism is the schedule: boundary Gauss-Seidel groups octants in the
quadrature's own order, which is lexicographic in the sign signature
with :math:`\operatorname{sign}\mu_x` slowest-varying (``[M]``
``(-1,-1,-1), (-1,-1,+1), \dots`` — x is the outer axis), and the mode's
:math:`(-1)^k` alternation closes with net :math:`+1` over an even cell
count while leaving a residual sign flip over an odd one.  A different
octant order therefore moves the excited axis, which couples this to
`#343 <https://github.com/deOliveira-R/ORPHEUS/issues/343>`_.

.. warning::

   ⛔ **"Even** :math:`n_x` **is safe" is FALSE as a statement about the
   mesh, and reading it that way builds an inert gate.**  Parity governs
   **excitation**, never the kernel.  ``[M]`` :math:`\dim \ker A = 12`
   at :math:`(2,2)`, :math:`(3,4)`, :math:`(4,4)`, :math:`(5,6)` **and**
   :math:`(6,8)` alike, and the even-\ :math:`n_x` box is excited the
   moment the source stops being symmetric:

   .. list-table::
      :header-rows: 1
      :widths: 18 34 24

      * - cells
        - source
        - :math:`\lVert\Pi t\rVert/\lVert t\rVert`
      * - :math:`(3,4)`
        - uniform isotropic
        - :math:`6.080483\times10^{-2}`
      * - :math:`(3,4)`
        - anisotropic :math:`(1+\mu_x)/W`
        - :math:`5.815861\times10^{-2}`
      * - :math:`(4,4)`
        - uniform isotropic
        - :math:`6.7\times10^{-14}`
      * - :math:`(4,4)`
        - anisotropic :math:`(1+\mu_x)/W`
        - :math:`1.756363\times10^{-2}`

   This is ``vv-principles`` #13's congruence-class trap: a refinement
   ladder of :math:`4, 8, 16, 32` is a **single parity class**, and a
   gate written on it is green, authoritative and structurally unable to
   fail.  It has already fired twice in this campaign — an exit map
   probed only :math:`4\times4` and :math:`6\times6` and concluded the
   gauge could not bite at the eigenvalue entry at all.  **Assert**
   ``dim ker == 0``, never infer kernel-freedom from a mesh property.

   ⟹ the user-facing statement, and it is stronger than
   non-determinism: **a** :math:`3\times N` **mesh reports a**
   :math:`\sim 7\,\%` **spurious tangential current, and a**
   :math:`4\times N` **mesh reports none.**
   The defect appears and vanishes under a mesh change that
   alters nothing qualitative, so it is **not discoverable by
   inspection** — which is why the solver says so out loud instead of
   leaving it to this page.

.. _sn-loss-kernel-the-gauge:

The gauge — returning the canonical member
-------------------------------------------

:class:`~orpheus.sn.operators.loss_kernel_gauge.LossKernelGauge` is the
:math:`G`-orthogonal projector :math:`\Pi` onto :math:`\ker A`, and the
operation the solver performs at every exit that returns a trace is

.. math::
   :label: sn-loss-kernel-gauge-projection

   \psi \;\longmapsto\; \psi - \Pi\psi ,
   \qquad
   A(\psi - \Pi\psi) \;=\; A\psi
   \quad\text{because}\quad \Pi\psi \in \ker A .

.. (vv-status rationale) Structural/representational: the definition of the
   shipped operation plus its residual-neutrality identity, which follows from
   the definition of a kernel and carries no solver claim of its own. Both are
   gated by the foundation suite tests/sn/operators/test_loss_kernel_gauge.py —
   idempotence and G-self-adjointness
   (test_it_is_an_idempotent_G_self_adjoint_projector) and the
   no-certificate-may-move contract on all six fixtures
   (test_gauging_cannot_move_any_convergence_certificate) — and end to end by
   tests/sn/solve/test_every_entry_gauges_its_trace.py.
.. vv-status: sn-loss-kernel-gauge-projection documented

The right-hand identity is the whole safety argument: the projection is
**residual-neutral by construction**, so firing it at a converged exit
**cannot move any convergence certificate**.  ``[M]`` on a deliberately
truncated SI solve the exit balance defect reads ``0.3111434602740818``
on the raw and on the gauged iterate alike, while the gauge correction
goes :math:`3.59\times10^{-2} \to 4.9\times10^{-17}`.  It is applied
*after* the defect is measured, so the number reported describes the
object the caller actually receives.

Scope is **component R only**.  :math:`\ker A` splits as
:math:`T \oplus R`, separated exactly by the metric's zero set (``[M]``
:math:`\max|B_R|` on :math:`G = 0` rows is :math:`0.000000\times10^{0}`
on ``level_symmetric``, ``product(4,4)`` *and* ``lebedev(11)``):

* **R** — the genuine trace underdetermination, on current-carrying
  ordinates.  Its modes carry non-zero trace metric, so the
  minimum-:math:`G`-norm member is unique and the projection is
  well-posed.
* **T** — the tangential slots.  ⛔ They lie in :math:`\ker G` (``[M]``
  :math:`\max|G|` on them is :math:`0.000000\times10^{0}` against
  :math:`\min|G| = 2.78\times10^{-1}` / :math:`7.64\times10^{-2}` on the
  rest), so *every* value has the same — zero — :math:`G`-norm and there
  is no minimum-norm representative to choose.  :math:`B^{\mathsf T} G B`
  is **singular** the moment :math:`T \neq 0`, so R and T must never be
  orthonormalised together.  T is left **untouched**, which is the
  correct action and not an omission: :math:`Gt = 0` makes
  :math:`t \perp_G \operatorname{span} R`, so :math:`(I - \Pi)t = t`.
  Typing T away so the directions are unrepresentable is filed
  separately.

What ships
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - surface
     - what it is
   * - :attr:`SNMesh.loss_kernel_gauge <orpheus.sn.mesh.augmented_mesh.SNMesh.loss_kernel_gauge>`
     - the cached projector.  On the mesh because the kernel is
       geometry-only; **zero blocks** on a non-singular configuration,
       so :meth:`gauge <orpheus.sn.operators.loss_kernel_gauge.LossKernelGauge.gauge>`
       is the identity and no consumer needs a ``None`` branch
   * - :func:`~orpheus.sn.operators.loss_kernel_gauge.gauge_freedom`
     - a **three-state** verdict — present / absent / **UNDETERMINED** —
       with a sentence naming the deciding conjunct.  Both conjuncts are
       *derived*: the closure is asked whether it leaves a face mode
       undamped, the mesh is asked how many reflective axis pairs close
   * - :class:`~orpheus.sn.operators.loss_kernel_gauge.GaugeFreedomWarning`
     - fires when the trace was **repaired**, or when the closure could
       not be classified.  Deliberately **not** a
       :class:`~orpheus.numerics.convergence.ConvergenceWarning`: the
       solve converged perfectly and the ambiguity is in the *equation*
   * - :attr:`IterationHistory.gauge_correction <orpheus.sn.solution.IterationHistory.gauge_correction>`
     - the measured :math:`\lVert\Pi\psi\rVert/\lVert\psi\rVert`.
       ``None`` means **not measured**, never *"measured and zero"* — the
       :attr:`balance_defect <orpheus.sn.solution.IterationHistory.balance_defect>`
       discipline

The three-state predicate is not ceremony.  ``[M]``
``linear_discontinuous`` **damps** the face mode at :math:`d=2`
(spectral radius :math:`0.860702 < 1`) and is **UNDETERMINED** at
:math:`d=3` — its ``assemble_inflow_axis`` defers the general
interior-axis interleave, so the closure cannot be driven and its
damping is unknown.  A caller who read UNDETERMINED as *absent* would
skip the gauge on a scheme nobody examined; instead the solver warns
loudly and states that the trace was **not** gauge-fixed.

.. list-table:: cost, single-process, measured 2026-08-15
   :header-rows: 1
   :widths: 26 16 12 12 16 18

   * - configuration
     - trace DOFs
     - blocks
     - :math:`\dim\Pi`
     - build (cached)
     - apply
   * - :math:`d=2` :math:`(3,4)` :math:`S_4` :math:`n_g{=}2`
     - 672
     - 12
     - 12
     - 3.0 ms
     - 85 µs
   * - :math:`d=3` :math:`(3,4,5)` :math:`S_4` :math:`n_g{=}2`
     - 4512
     - 6
     - 138
     - 22.0 ms
     - 184 µs

Against solves that take seconds and :math:`O(10^3\!-\!10^4)` sweeps on
the same fixtures, and with the build amortised over every group, outer
and eigenvalue iterate, the projection is well under 0.2 % of one solve.

Every public entry gauges — including the two that cannot be exercised
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gauge is the sibling of the exit balance defect
(:eq:`sn-exit-balance-defect`) with one sharpening that changes the
verification obligation: **that one REPORTS and this one MUTATES.**  A
forgotten balance-defect site loses a diagnostic; a forgotten gauge site
silently returns a non-physical answer.  Both fixed-source arms project
— SI *and* Krylov, the latter's own comment notwithstanding (it
described the returned boundary as "the matvec's :math:`B_1''` face
residual", which is the boundary block of :math:`A\psi`, a different
object from the solution vector's; ``[M]`` it is a flux trace, and the
sentence has been corrected).  ``solve_sn_adjoint`` and
``solve_sn_adjoint_fixed_source`` are wired for uniformity and are
**structurally inert**: the adjoint routes through :math:`(L+C)^{\mathsf H}`,
whose transpose solve is 1-D-scan-only, and a 1-D problem has at most
one reflective axis pair.

.. note::

   ⚠ **Do not read a green adjoint test as evidence the gauge works** —
   that is *inert*, not *verified*.  Coverage is gated by
   ``tests/sn/solve/test_every_entry_gauges_its_trace.py``, whose entry
   list is **derived from the module** rather than hand-written, so a
   new ``solve_sn*`` entry that forgets to gauge cannot pass by being
   unknown to the gate.

.. _sn-loss-kernel-remedies:

The remedy hierarchy
---------------------

The gauge is a *repair*, and the solver's warning says so: it names what
would remove the freedom **at the root**, asked of the closure registry
rather than tabulated, so a closure added tomorrow appears in that
sentence with no edit anywhere.

.. list-table::
   :header-rows: 1
   :widths: 6 30 64

   * - #
     - remedy
     - what it costs, and when it is available
   * - 1
     - **Change the spatial closure** — the root fix
     - The freedom is a property of the *diamond* closure, not of
       transport and not of the boundary condition.  ``[M]``
       ``linear_discontinuous`` gives :math:`\dim\ker A = 0` on the
       identical box at :math:`d=2`.  ⚠ At :math:`d=3` **no closure in
       this build damps the mode** — LD is UNDETERMINED there — so at
       :math:`d=3` this remedy is not currently available and the
       warning says exactly that.
   * - 2
     - **Break one reflective axis pair**
     - A mode needs a closed loop to return to itself; ``[M]`` at
       :math:`d=2` a single vacuum face collapses :math:`\dim\ker A`
       from 12 to 0.  Only available if the physics allows it — an
       infinite-lattice :math:`\kinf` calculation is all-reflective by
       definition.
   * - 3
     - **Gauge it** — what ships
     - Exact by construction, not by fixture; recovers
       :math:`\psi_{\rm exact}`'s member; bulk bit-untouched; no
       certificate moves; ``[M]`` well under 0.2 % of one solve.
   * - 4
     - ⛔ **Switch the default schedule to Jacobi** — REJECTED
     - ``jacobi`` does land on the canonical member (``[M]``
       :math:`\sim10^{-15}` on every row above), and so does Krylov
       (``[M]`` :math:`4.1\times10^{-14}`).  But this is a **rate**
       decision taken for a **correctness** reason, it re-opens the
       schedule question of :ref:`sn-boundary-gs-rate-regime`
       (``[M]`` Jacobi is 2.5× *slower* at :math:`d=2` and 2.3× *faster*
       at :math:`d=3`), and its correctness rests on five fixtures
       rather than on a theorem.  The projection is the only option that
       is exact by construction.
   * - 5
     - ⛔ **Dense SVD of** :math:`A` **for the null space** — REJECTED
     - Exact, and priced out: ``[M]`` **23.0 s** at
       :math:`n_{\rm dof} = 3744` against a sub-10 ms closed-form build
       — :math:`\ge 2000\times` at the one size where both were run —
       with the dense matrix alone :math:`O(n_{\rm dof}^2)` (5.5 TB at
       :math:`(12,12,12)` :math:`S_8` :math:`n_g{=}4`).

.. _sn-loss-kernel-corrections:

What this corrects, and what was tried and failed
---------------------------------------------------

.. attention::

   ⛔ **RETRACTION (2026-08-15, #344) — "the converged fixed point is
   splitting-invariant" is FALSE as an unqualified universal**, and this
   page asserted it in three places: the Key Facts card, the
   FP-invariance paragraph of :ref:`sn-boundary-gs-not-regular`'s
   verification subsection, and its *What survives* summary.  The
   companion claim on :ref:`sn-solver-operator-algebra-coordinator` —
   that source iteration and Krylov "converge to the same fixed point" —
   is the same statement and had the same defect.

   Every one of them presupposes that a fixed **point** exists.  When
   :math:`A` is singular there is a fixed **manifold**: :math:`\ker A`
   is splitting-invariant, but the *complementary* invariant subspace is
   not, so the oblique projector whose range the iteration freezes
   differs by schedule and two perfectly correct splittings legitimately
   return **different members**.  ``[M]`` on the Key-result fixture, the
   boundary-Gauss-Seidel and Jacobi traces differ by
   :math:`\lVert d\rVert = 0.124184` with
   :math:`\lVert\Pi d\rVert/\lVert d\rVert = 1.000000` — the difference
   is *entirely* kernel content, with :math:`1.77\times10^{-13}` outside
   it.

   **What survives, and it is most of the claim.** The bulk *is*
   invariant — ``[M]`` the scalar flux, :math:`\keff` and every reaction
   rate are unchanged, because the kernel is pure-trace — and every
   citation of splitting-invariance in this book rests on a bulk or
   eigenvalue measurand.  The three sentences are now scoped to say
   *which* object is invariant, and the gates they cite are unaffected:
   ``test_w2_fixed_point_equivalence_diagonal_cubature`` runs on a
   heterogeneous vacuum-x / reflective-y box, which closes **one**
   reflective axis pair, and ``[M]`` :math:`\dim\ker A = 0` there.

   The corpus statement of the underlying doctrine is
   :ref:`the Mode-9 entry <verification-test-design-modes>`, which now
   carries the premise clause.

⚠ The discriminator matters, because *"the boundary moved and the bulk
did not"* is **also** the signature of a genuinely incoherent schedule —
one whose :math:`M - N \neq A`, the ERR-056 family.  Three checks
separate them, and all three were run: (a) :math:`M - N \equiv A`
bit-exactly for both splittings; (b) with the kernel removed the two
schedules agree on the **boundary** as well as the bulk; (c) the
difference lies in :math:`\ker A` (``[M]`` :math:`1.000000`).  An
incoherent splitting moves the bulk too.

.. list-table:: refuted candidates, kept with the structural reason each failed
   :header-rows: 1
   :widths: 34 66

   * - candidate
     - why it fails
   * - ":math:`N/4` counts ordinate **quartets within one octant**"
     - :math:`N/4` is the **orbit** count of
       :math:`\langle R_x, R_y\rangle` acting on the FULL 3-D ordinate
       set.  Decisive: at :math:`d=2` the ordinates
       :math:`(\mu_x,\mu_y,+\mu_z)` and :math:`(\mu_x,\mu_y,-\mu_z)` lie
       in **different** orbits and give **independent** modes — the
       quartet reading predicts 3 where the measurement is 12.
   * - "the modes are spatially **uniform** on each face"
     - The balance forces the *checkerboard-weighted* function to be
       constant, so the raw face profile is :math:`(-1)^{i_\perp}`.  A
       uniform ansatz has :math:`\lVert Ab\rVert/\lVert b\rVert = O(1)`,
       not :math:`10^{-16}`.  The uniform reading came from summing the
       projector diagonal over an orthonormal basis of a symmetric
       space — a **mass**, not a mode.
   * - "a loop/cocycle **parity** condition on closed reflective loops"
     - The :math:`(-1)^{n_a}` accumulated around a loop **cancels
       identically** between the two sides of the specular
       identification, so the loop imposes no condition at all.  Had it
       imposed one, :math:`\dim\ker A` would depend on cell-count
       parity; ``[M]`` it does not — :math:`12` at :math:`(2,2)`
       through :math:`(6,8)`.
   * - "one mode **per ordinate**, not per orbit"
     - Over-complete by :math:`2^d`: the specular condition
       :math:`\varphi^n_a = \varphi^{R_a n}_a` collapses each orbit's
       ordinate-local solutions onto ONE.  It destroys *independence*,
       not the residual — which is why every verification row reports
       :math:`\sigma_{\min}` of the stacked basis, not only
       :math:`\lVert Ab\rVert`.
   * - "treat the tangential slots **with the same machinery** as R"
     - The construction divides by :math:`|\mu_a|`; for a tangential
       ordinate that is :math:`0/0`.  Worse, T sits in :math:`\ker G`,
       so one :math:`B^{\mathsf T} G B` orthonormalisation over
       :math:`R \oplus T` is **singular**.
   * - "a refinement fit :math:`\dim \propto n_x n_y n_z`"
     - The law is :math:`2\sum n - 1`, a **sum**, because the free
       functions live on **planes** of cells indexed by one coordinate,
       never on the full cell set.
   * - "#344 reading 1 — the singularity is **tangential-slot
       bookkeeping**"
     - Both readings are real and they are a **direct sum**: they are
       the :math:`|S| \ge 2` and the :math:`|S| < 2`-with-a-free-slot
       branches of the same equation.  ``[M]`` ``product(4,4)`` at
       :math:`d=2` is pure T (:math:`R = 0`); ``level_symmetric`` is
       pure R (:math:`T = 0`).
   * - ⛔ "the even-\ :math:`n_x` split is **detector blindness** — the
       mode is present but the :math:`(-1)^{i_\perp}` profile makes a
       uniform detector read zero"
     - A true statement about detectors, and **not** the explanation.
       ``[M]`` at even :math:`n_x` the returned deviation has norm
       :math:`2.2\times10^{-13}\dots4.6\times10^{-13}` and is only
       15–31 % inside :math:`\ker A` — the mode is **absent from the
       iterate**, not hidden from the probe.  Two mechanisms share one
       parity fingerprint, and only a *kernel-content* measurement (not
       a deviation measurement) tells them apart.

.. warning::

   ⛔ **The first witness gate designed for this defect could not have
   failed, and it is worth knowing why.**  The proposed acceptance test
   was *"the* :math:`|\Omega\cdot n|^0` *full-face moment moves, and
   only where the quadrature has tangential ordinates"*.  ``[M]`` the
   moment moves **nowhere** — not under ``lebedev`` either — and it
   **could not have**, by :eq:`sn-kernel-mirror-blindness`: a
   face-summed moment is mirror-even and every kernel mode is
   mirror-odd.  The gate would have shipped green, authoritative and
   structurally unfalsifiable (``plan-authoring`` §6c), reached by
   reasoning from a hunch instead of from a theorem the same document
   already contained.  The witness that works is the mirror-**odd**
   tangential current of :ref:`sn-loss-kernel-what-a-user-sees`.

   The same trap sits one level down in the construction: the modes'
   Gram was measured **exactly diagonal** and the reading was
   *vacuous* — at :math:`d=2` each orbit carries exactly one mode, and a
   :math:`1\times1` Gram is diagonal for free.  ``[M]`` at
   :math:`d \ge 3` it is **43 % off-diagonal**, and declaring
   ``DIAGONAL`` there would have normalised every coefficient by the
   wrong number, silently.  The module docstring's §5 carries the
   measurement and the fix.

What broadens next
==================

* **Curvature** (spherical/cylindrical): the angular cell balance
  activates — direction is no longer constant in flight, so the
  angular axis acquires its own cell balance, redistribution
  coefficients, and starting-direction state. The walk becomes a
  sequential per-ordinate march; the closure machinery of
  :doc:`/theory/foundations/discretization` §5 is applied on the
  angular axis. That is Part B of this book.
