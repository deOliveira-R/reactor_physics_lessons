.. _operator-adjoint:

============================
The Composite Metric Adjoint
============================

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
      concept: operator_adjoint
      role: "the operator algebra's metric-correct Hilbert adjoint op.H = G⁻¹AᵀG over FullFieldSpace (block-diagonal metric; singular-trace pseudo-inverse)"
      depends_on: [operator_algebra]
      related: [frame]
      status: "extracted from operator_algebra.rst; content verbatim, provisional header"


This page develops the **metric-correct Hilbert adjoint** ``op.H`` of a
composed S\ :sub:`N` transport operator — the **G-adjoint**
:math:`A^{\dagger} = G^{-1} A^{\mathsf T} G`, defined by the reciprocity
identity :math:`\langle A\psi, \varphi\rangle_G = \langle \psi,
A^{\dagger}\varphi\rangle_G` over the block-diagonal
:class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace` metric
(bulk block :math:`V_{\rm cell}\,w_n`; singular trace block
:math:`|\Omega\cdot\hat n_f|\,w_n`, inverted by its Moore–Penrose
pseudo-inverse). It is the adjoint face of the operator algebra developed
in :doc:`/theory/foundations/operator_algebra`.

.. important::

   This is the **operator's** composite adjoint — the Hilbert adjoint
   ``op.H`` of a whole loss composite
   :math:`A = L + C - S - N_{2n} - B`, taken
   over the *physical* phase-space inner product. It is **distinct** from
   the **frame's** ``R.H`` / Petrov–Galerkin **test-space** adjoint
   developed in :doc:`/theory/foundations/frame`: the frame adjoint
   turns a reconstruction into an analysis across a *trial/test* pair,
   whereas the G-adjoint here turns an operator over on the *same* inner
   product. Different level, different object.


.. _g-adjoint:

The composite metric-correct G-adjoint — ``op.H`` over ``FullFieldSpace``
=========================================================================

This section completes the boundary-condition extraction narrative
documented at :ref:`bc-extraction` in
:doc:`/theory/foundations/boundary_conditions`.

Wave O step **O.2b R5** (`Issue #208
<https://github.com/deOliveira-R/ORPHEUS/issues/208>`_, commits
``89b2f62`` / ``0efd233`` / ``5c06196``, 2026-06-05) discharges the
second open item of :ref:`bc-extraction-operator-output-o2`: it makes
``op.H`` — the Hilbert adjoint of an SN operator composite — the
**metric-correct G-adjoint**

.. (vv-status rationale) The G-adjoint defining identity. The
   verifiable claim is the reciprocity ⟨Aψ,φ⟩_G = ⟨ψ,A†φ⟩_G plus the
   block-diagonal G-fold op.H = G⁻¹AᵀG, both pinned against a
   structurally-independent dense-transpose-plus-explicit-diagonal-G
   oracle (derivations/diagnostics/diag_p42_adjoint_oracle.py) — NOT a
   code-to-code comparison against another ORPHEUS adjoint path. This
   is the algebra-of-record ground for the equation.
.. vv-status: g-adjoint-definition documented

.. math::
   :label: g-adjoint-definition

   A^{\dagger} \;=\; G^{-1}\,A^{\mathsf T}\,G ,

acting on a composite ``bulk ⊕ boundary`` field (the timeless operator
carrier :class:`~orpheus.transport.full_field.FullField`; a driver's
history-bearing
:class:`~orpheus.transport.timed_full_field.TimedFullField` iterate reaches
``op.H`` via MRO, as in the reciprocity gate). Before
R5, ``op.H`` silently reduced to the plain **Euclidean** transpose
:math:`A^{\mathsf T}` because the SN operators advertised no
metric-bearing ``domain`` / ``codomain``; the wrapper had no metric to
read, so :math:`G` defaulted to the identity. R5 supplies the metric by
giving the FULL streaming leaf a direct-sum function space, and the
already-existing :class:`~orpheus.numerics.operator.AdjointOperator`
wrapper turns that into the correct G-adjoint **with no change to the
wrapper**.

.. admonition:: Key Facts (composite G-adjoint)
   :class: tip

   - **The Hilbert adjoint is the G-adjoint** :math:`A^{\dagger} =
     G^{-1} A^{\mathsf T} G`, NOT the Euclidean transpose. The
     reciprocity identity it satisfies is :math:`\langle A\psi,
     \varphi\rangle_G = \langle \psi, A^{\dagger}\varphi\rangle_G`.
   - **G is block-diagonal** on :math:`V = V_{\rm bulk}\oplus
     V_{\rm trace}`: bulk block :math:`G_{\rm bulk} = V_{\rm cell}\,w_n`
     (the full phase-space measure :math:`\mathrm dV\,\mathrm d\Omega`);
     trace block :math:`G_{\rm trace} = |\Omega\cdot\hat n_f|\,w_n` (the
     partial-current surface measure). Both carry :math:`w_n`; they
     differ only in the spatial measure (cell volume vs. oriented face).
   - **The carrier is** :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`
     — a direct sum that dispatches the metric **per block** to its
     leaf spaces (a pure composition, no new metric arithmetic).
   - **The metric-blind adjoint is unrepresentable — the metric lives
     on the shared space, NOT any leaf's domain.** Since P4.5 W-D, ``C``
     / ``S`` / ``F`` carry the SAME composite ``full_field_space`` as
     ``L`` / ``B`` (so the within-group
     :class:`~orpheus.numerics.operator.OperatorSum` guard *validates*
     the loss composition); the metric applies **once at the op level**
     because the :class:`~orpheus.numerics.operator.AdjointOperator`
     wrapper reads it off the *composite* ``domain`` / ``codomain`` of
     the summed operator, never per-leaf. Because every loss leaf now
     carries the composite metric, no composite adjoint is metric-blind,
     and a non-adjointable operand still makes the recursive
     :attr:`~orpheus.numerics.operator.LinearOperator.is_adjointable`
     report ``False``, so ``.H`` **raises**
     :class:`~orpheus.numerics.operator.MissingAdjoint` **eagerly at**
     ``.H`` **construction** — it never silently goes Euclidean. (The
     **Supersession note below** records the retirement of an earlier
     ``S`` / ``F`` no-``apply_transpose`` mechanism that predated the
     #112 / #118 / #276 adjoint work.)
   - **The trace metric is singular** on tangential :term:`ordinates <ordinate>`
     (:math:`|\Omega\cdot\hat n| = 0`), so :math:`G^{-1}` is the
     **Moore–Penrose pseudo-inverse** (zero on the null space). This is
     **exact** for the adjoint: the inflow / outflow selectors exclude
     tangential ordinates, so those slots are identically zero in every
     matvec output.
   - **Discriminating gate:** the L11 wrong-metric control (drop
     :math:`|\Omega\cdot\hat n|` from the trace block) breaks
     reciprocity by :math:`6.4\times10^{-2}` (slab) /
     :math:`8.3\times10^{-1}` (sphere) / :math:`1.9\times10^{-3}` (cyl)
     — all :math:`\gg 10^{-3}`, proving the weighting is load-bearing.


The G-adjoint, derived from reciprocity
---------------------------------------

The Hilbert adjoint :math:`A^{\dagger}` of a linear operator on a space
with inner product :math:`\langle\cdot,\cdot\rangle_G` is *defined* by
the reciprocity (turn-over) identity

.. (vv-status rationale) The DEFINING adjoint reciprocity
   ⟨Aψ,φ⟩_G = ⟨ψ,A†φ⟩_G. Pinned by the foundation-tagged
   tests/sn/operators/test_g_adjoint_reciprocity.py, which by design
   carries no verifies() (an algebraic identity over the operator-algebra
   ground truth, anchored to the structurally-independent dense-transpose-
   plus-explicit-diagonal-G oracle), matching the sentineled
   g-adjoint-definition.
.. vv-status: g-adjoint-reciprocity documented

.. math::
   :label: g-adjoint-reciprocity

   \langle A\psi,\,\varphi\rangle_G
   \;=\;
   \langle \psi,\,A^{\dagger}\varphi\rangle_G
   \qquad\forall\,\psi,\varphi \in V .

The diagonal metric is represented by a (block-diagonal) Gram matrix
:math:`G`, so the inner product is :math:`\langle a,b\rangle_G = a^{\mathsf
T} G\, b`. Substituting into :eq:`g-adjoint-reciprocity` and solving for
:math:`A^{\dagger}` recovers :eq:`g-adjoint-definition` in three steps:

.. (vv-status rationale) The three-step algebra deriving
   A† = G⁻¹AᵀG from reciprocity. Derivation step; its terminal result is
   g-adjoint-definition (sentineled), pinned by the foundation-tagged
   reciprocity oracle.
.. vv-status: g-adjoint-derivation documented

.. math::
   :label: g-adjoint-derivation

   \langle A\psi, \varphi\rangle_G
   = (A\psi)^{\mathsf T} G\, \varphi
   = \psi^{\mathsf T}\,(A^{\mathsf T} G)\,\varphi
   \;\overset{!}{=}\;
   \psi^{\mathsf T} G\, (A^{\dagger}\varphi)
   = \langle \psi, A^{\dagger}\varphi\rangle_G ,

so :math:`A^{\mathsf T} G = G\,A^{\dagger}` and therefore
:math:`A^{\dagger} = G^{-1} A^{\mathsf T} G`. When :math:`G = I` (the
Euclidean default) this collapses to the bare transpose
:math:`A^{\dagger} = A^{\mathsf T}` — which is precisely the
**wrong** adjoint for the SN composite, because the bulk and boundary
blocks are integrated against *non-uniform* measures (a cell of twice
the volume contributes twice the inner-product weight; a grazing
ordinate contributes :math:`|\Omega\cdot\hat n|` less surface current).
The whole point of R5 is to supply the correct :math:`G` so the
reciprocity holds under the *physical* measure, not the counting
measure of array indices.

This is the discrete twin of the continuous transport reciprocity
:math:`\int \mathrm dV\!\int \mathrm d\Omega\; \varphi\,(A\psi) = \int
\mathrm dV\!\int \mathrm d\Omega\; \psi\,(A^{\dagger}\varphi)` (Lewis &
Miller 1993, §3.7) — the phase-space integral :math:`\mathrm
dV\,\mathrm d\Omega` is what the bulk metric :math:`V_{\rm cell}\,w_n`
discretizes, and the surface partial-current integral :math:`\int_\Gamma
|\Omega\cdot\hat n|\,\mathrm dA\,\mathrm d\Omega` is what the trace
metric :math:`|\Omega\cdot\hat n_f|\,w_n` discretizes.


The block-diagonal metric — why each block carries its measure
--------------------------------------------------------------

The composite lives on the **direct sum** :math:`V = V_{\rm bulk}\oplus
V_{\rm trace}`, so its Gram matrix is **block-diagonal** (the bulk and
trace degrees of freedom are distinct coordinates — there is no
cross-term, the two integrals are over different domains):

.. (vv-status rationale) The block-diagonal metric
   G = diag(G_bulk, G_trace), G_bulk = V_cell w_n,
   G_trace = |Ω·n_f| w_n. Definitional structure; the |Ω·n_f| weighting
   is proven load-bearing by the L11 wrong-metric control in the
   foundation-tagged reciprocity suite.
.. vv-status: g-adjoint-block-metric documented

.. math::
   :label: g-adjoint-block-metric

   G \;=\;
   \begin{pmatrix} G_{\rm bulk} & 0 \\ 0 & G_{\rm trace} \end{pmatrix},
   \qquad
   G_{\rm bulk} = V_{\rm cell}\,w_n ,
   \qquad
   G_{\rm trace} = |\Omega\cdot\hat n_f|\,w_n .

**The bulk block** :math:`G_{\rm bulk} = V_{\rm cell}\,w_n`. The bulk
inner product is the discretization of the full phase-space integral

.. (vv-status rationale) The bulk inner-product discretization
   ∫dV∫dΩ ab → Σ_i V_i Σ_n w_n a b. Definitional (notation of the
   phase-space measure); pinned by the foundation-tagged
   metric-population cross-check.
.. vv-status: g-adjoint-bulk-inner-product documented

.. math::
   :label: g-adjoint-bulk-inner-product

   \langle a, b\rangle_{G_{\rm bulk}}
   \;=\;
   \int_{\mathcal D}\!\mathrm dV
   \int_{4\pi}\!\mathrm d\Omega \; a\,b
   \;\longrightarrow\;
   \sum_{i\in\text{cells}} V_i
   \sum_{n\in\text{ordinates}} w_n \; a_{n,i}\,b_{n,i} .

The two :term:`quadratures <quadrature>` factor: the **cell volume** :math:`V_i`
discretizes :math:`\mathrm dV` and the **angular quadrature weight**
:math:`w_n` discretizes :math:`\mathrm d\Omega`. The product
:math:`V_i\,w_n` is therefore the diagonal phase-space measure
:math:`\mathrm dV\,\mathrm d\Omega`. In code
(:meth:`SNMesh.full_field_space <orpheus.sn.mesh.augmented_mesh.SNMesh.full_field_space>`)
it is built as

.. code-block:: python

   g_bulk = w_n[:, None, None, None] * V[None, None, :, :]   # (N, 1, nx, ny)

— a ``(N, 1, nx, ny)`` array. The leading axis carries the per-ordinate
:math:`w_n`; the two spatial axes carry the per-cell :math:`V`; the
**energy-group axis is a singleton** because the phase-space measure is
**group-independent** (a group does not change a cell's volume or an
ordinate's solid angle). The singleton broadcasts over the energy axis
of the ``(N, ng, nx, ny)`` bulk tensor at metric-application time, so
the same :math:`(N,1,nx,ny)` weight serves every group with no
duplication. This is exactly the leading-axis broadcast convention of
:meth:`FunctionSpace._broadcast_metric <orpheus.numerics.space.FunctionSpace>`.

**The trace block** :math:`G_{\rm trace} = |\Omega\cdot\hat n_f|\,w_n`.
The boundary inner product is the discretization of the **partial-current
surface** integral

.. (vv-status rationale) The trace partial-current inner-product
   discretization ∫dA∫dΩ |Ω·n_f| ab → Σ_f Σ_n |Ω·n_f| w_n a b.
   Definitional; the cosine weighting is the L11 discriminating control
   in the foundation-tagged reciprocity suite.
.. vv-status: g-adjoint-trace-inner-product documented

.. math::
   :label: g-adjoint-trace-inner-product

   \langle a, b\rangle_{G_{\rm trace}}
   \;=\;
   \int_{\Gamma}\!\mathrm dA
   \int_{4\pi}\!\mathrm d\Omega \; |\Omega\cdot\hat n_f| \; a\,b
   \;\longrightarrow\;
   \sum_{f\in\text{faces}}
   \sum_{n} |\Omega_n\cdot\hat n_f|\,w_n \; a_{n,f}\,b_{n,f} .

The boundary degrees of freedom are :term:`angular fluxes <angular flux>` *on a surface*, and
the physically meaningful surface functional is the **partial current**
:math:`J^{\pm} = \int_{\Omega\cdot\hat n \gtrless 0} |\Omega\cdot\hat
n|\,\psi\,\mathrm d\Omega`, so the surface measure carries the
**cosine factor** :math:`|\Omega\cdot\hat n_f|`. This is the same
:math:`|\Omega\cdot\hat n|`-weighted inner product under which the
reflective / white boundary operators are self-adjoint (see
:ref:`bc-extraction` and :mod:`orpheus.numerics.spaces.angular_trace_space`),
already populated in sub-step 4.1 (commit ``89b2f62``). R5 reuses it
verbatim as the trace block — it does **not** re-derive it.

**Both blocks carry** :math:`w_n` **— they differ only in the spatial
measure.** This is the structural symmetry that makes the direct-sum
metric a clean composition rather than two unrelated weightings: angular
integration is identical on both blocks (the same quadrature
discretizes :math:`\mathrm d\Omega` everywhere), and the spatial measure
specializes — a 3-D **volume** :math:`V_{\rm cell}` for bulk degrees of
freedom that live *in* a cell, a 2-D **oriented surface**
:math:`|\Omega\cdot\hat n_f|\,\mathrm dA` for trace degrees of freedom
that live *on* a face.


The carrier — :class:`FullFieldSpace`, a per-block metric dispatcher
--------------------------------------------------------------------

The metric :eq:`g-adjoint-block-metric` is carried by a new function
space :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`
(``orpheus/numerics/spaces/full_field_space.py``). It holds the two leaf
spaces — a bulk :class:`~orpheus.numerics.space.FunctionSpace` whose
``inner_product_weights`` is :math:`G_{\rm bulk}`, and the existing
:class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace` whose
``inner_product_weights`` is :math:`G_{\rm trace}` — and **overrides**
the three metric primitives to dispatch **per block**:

.. code-block:: python

   def apply_metric(self, x):            # G ⊙ x
       return self._rebuild(
           x,
           self.bulk_space.apply_metric(x.bulk.values),
           self.trace_space.apply_metric(x.boundary.values),
       )

Each override splits the composite field into its ``.bulk`` /
``.boundary`` blocks, routes each to its leaf space's metric method, and
rebuilds the composite. This is a **pure Pattern-2 composition**: the
direct-sum space owns only the *structure* (how to split and recombine);
it introduces **no new metric arithmetic** — the per-axis broadcast and
the pseudo-inverse already live in
:class:`~orpheus.numerics.space.FunctionSpace`. The inner product is
correspondingly the **sum of block inner products**
:math:`\langle a, b\rangle_G = \langle a_{\rm bulk}, b_{\rm
bulk}\rangle_{G_{\rm bulk}} + \langle a_{\rm trace}, b_{\rm
trace}\rangle_{G_{\rm trace}}` — the defining property of a direct-sum
Hilbert space.

The space is **duck-typed** on the composite field (``.bulk`` /
``.boundary`` leaves, each a frozen dataclass with a ``.values``
ndarray; rebuilt via :func:`dataclasses.replace`) so that the
``numerics`` layer never imports the ``transport`` layer — an
architectural firewall that keeps the operator-algebra primitives
domain-agnostic. The identity is the inherited ``(name, shape)`` tuple
(``shape = (n_bulk + n_trace,)``), with the block spaces as
``compare=False`` leaf metadata. The composite is **axes-less**, so the
identity flip (structural ``__eq__``, CS4c step 6, 2026-09-07) leaves it
exactly as it was — and ``(name, shape)`` here IS content identity,
because :meth:`FullFieldSpace.from_blocks
<orpheus.numerics.spaces.full_field_space.FullFieldSpace.from_blocks>`
derives ``name = "full_field#<digest>"`` by folding each member's
``(name, shape)`` pair. ⛔ This sentence read *"``name = "full_field"``
… so two composites over meshes of the same total dimension compare
equal"* until 2026-09-07: that bare name is what made any two composites
of equal flat dimension identical — the R2 block-blindness the landed
**CS4b S3** re-key retired. Two composites compare equal iff their
MEMBERS do, and the
:class:`~orpheus.numerics.operator.OperatorSum` composition guard
accepts the full within-group loss ``L + C - S - N_2n - B`` (every
operand — :math:`L`, :math:`C`, :math:`S`, :math:`N_{2n}` and :math:`B`
— reports the same
composite domain; P4.5 W-D gave the previously ``None``-spaced
:math:`C`/:math:`S`/:math:`F` real spaces and de-SN-ified the name from
``"sn_full_field"``).
The mesh exposes it as the cached property
:meth:`SNMesh.full_field_space <orpheus.sn.mesh.augmented_mesh.SNMesh.full_field_space>`.

**The wrapper is unchanged.** The whole apparatus plugs into the
**pre-existing** :class:`~orpheus.numerics.operator.AdjointOperator`,
which realizes ``A.H`` as

.. (vv-status rationale) The AdjointOperator realization
   (A†φ) = G_V⁺ ⊙ Aᵀ(G_W ⊙ φ). Structural identity of the wrapper code;
   pinned end-to-end by the foundation-tagged reciprocity oracle
   (op.H vs the explicit block-diagonal G-fold).
.. vv-status: g-adjoint-wrapper-action documented

.. math::
   :label: g-adjoint-wrapper-action

   (A^{\dagger}\varphi)
   \;=\;
   \underbrace{G_V^{+}}_{\substack{\text{domain}\\\text{inverse-metric}}}
   \;\odot\;
   A^{\mathsf T}\!\Bigl(
   \underbrace{G_W}_{\substack{\text{codomain}\\\text{metric}}}\!\odot\,\varphi
   \Bigr) ,

calling ``codomain.apply_metric`` *before* the transpose and
``domain.apply_inverse_metric`` *after* (operator.py, ``AdjointOperator.apply``).
The wrapper is metric-*representation*-agnostic: the SAME code path
serves the flat-ndarray spherical-harmonic :math:`(L+1, 2L+1)`
leading-axis metric AND the composite ``bulk ⊕ trace`` metric on a
structured field. R5 added a function space, not a line of adjoint
logic — the cleanest possible realization of the
:ref:`metric-lives-at-the-leaf <eigenvalue-posing>` principle.


The G-adjoint applies the metric once at the op level — the adjoint-axis predicate
----------------------------------------------------------------------------------

The subtle architectural point of R5 is **where** the metric lives in
the adjoint, and it survives the P4.5 W-D change to the bulk leaves'
domains. Since W-D, all six operators — :math:`L`
(:class:`~orpheus.sn.operators.streaming.StreamingOperator`), the collision
multiplier :math:`C = M[\sigma_t]`
(:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`),
:class:`~orpheus.transport.operators.scattering.ScatteringOperator` (``S``),
:class:`~orpheus.transport.operators.n2n.N2NOperator` (:math:`N_{2n}`,
which joined at CS4c step 3 — see the note below),
:class:`~orpheus.transport.operators.fission.FissionOperator` (``F``), and
:math:`B` (:class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`) —
carry the **same** composite ``full_field_space`` (threaded through
``from_solver_data`` / ``sn_mesh.full_field_space``), so the
within-group :class:`~orpheus.numerics.operator.OperatorSum` guard
*validates* the loss composition (``domain = None`` survives only on
the bare / test constructor).

.. note::

   **Precision on the** :math:`F` **row since CS4c step 4
   (2026-08-30).**  The sentence above is about the operators the
   *composite* posings compose, and it remains exactly true of them:
   :class:`~orpheus.transport.operators.fission.FissionOperator` is
   still minted from ``sn_mesh.full_field_space`` at the eigen-:math:`M`
   posing site, so the daggered pencil's ends validate natively — and
   :math:`N_{2n}`
   (:class:`~orpheus.transport.operators.n2n.N2NOperator`) joined the
   list at step 3 on the same space.  What changed is that the
   **k-outer** no longer holds that operator: it feeds bare
   :math:`(n_g, *\text{spatial})` scalar arrays, so ``SNSolver`` binds
   the fission **energy** binding
   :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`
   on the mesh's *scalar bulk* space instead.  That is the binding-arity
   table made honest rather than a weakening: an operator's ends now
   name the space its consumer actually feeds it, and reading "every
   fission operator in the tree carries ``full_field_space``" off this
   paragraph would be wrong for that one.  See
   :ref:`sn-fission-binding-adjoint`.

The architecturally interesting fact is
that this domain plumbing does **not** change where the metric is
applied in the adjoint, and that the metric-blind adjoint is made
**unrepresentable by the recursive** ``is_adjointable`` **predicate, not
by any leaf's domain**. Two mechanisms.

**(1) The metric applies ONCE at the op level — never per leaf.** The
:class:`~orpheus.numerics.operator.AdjointOperator` wrapper realizes
``op.H`` as :math:`G_V^{+}\odot(\cdot)^{\mathsf T}\!\bigl(G_W\odot(\cdot)\bigr)`,
reading :math:`G` off the **wrapped operator's** ``domain`` /
``codomain``. When the wrapped operator is the *sum* :math:`(L+C-B)`,
that is the sum's composite ``full_field_space`` (the
:class:`~orpheus.numerics.operator.OperatorSum` ``domain`` is the
first-non-``None`` of its operands — now redundant, since all operands
agree on the same composite). The adjoint therefore applies the metric
**once on the composite**:

.. (vv-status rationale) The sum-conjugation distribution
   (L+C−B)† = G⁻¹(L+C−B)ᵀG = Σ G⁻¹(·)ᵀG. Structural identity (the metric
   is applied once at the op level, never per leaf); pinned by the
   foundation-tagged reciprocity suite on the (L+C−B) composite.
.. vv-status: g-adjoint-sum-conjugation documented

.. math::
   :label: g-adjoint-sum-conjugation

   (L + C - B)^{\dagger}
   \;=\;
   G^{-1}\,(L + C - B)^{\mathsf T}\,G
   \;=\;
   G^{-1} L^{\mathsf T} G
   + G^{-1} C^{\mathsf T} G
   - G^{-1} B^{\mathsf T} G ,

distributing the **same** :math:`G^{-1}(\cdot)^{\mathsf T} G`
conjugation across every leaf in the sum. Although each leaf now
*advertises* the composite domain, the
:class:`~orpheus.numerics.operator.AdjointOperator` applies :math:`G`
at the **sum** level, never re-applying it per summand — the metric
weighting belongs to the *space*, applied once where the composite
enters and leaves, so a leaf carrying the composite domain is **not** a
double-application risk (the leaves' own ``apply_transpose``, where
defined, is the metric-blind Euclidean transpose; the metric is layered
on once by the sum's adjoint wrapper). The bulk leaves are pure
:math:`(N,ng,nx,ny)\to(N,ng,nx,ny)` endomorphisms whose transpose is
well-defined Euclidean-wise.

**(2) Every loss leaf carries the metric, so no composite adjoint is
metric-blind.** The concern the original design guarded against was an
adjoint taken over a sum that does **not** contain ``L`` and therefore,
under the *pre*-W-D design, carried no metric :math:`G`. That concern is
now **moot**: since P4.5 W-D every loss leaf — ``L``, ``C``, ``S``,
``F``, ``B`` — carries the SAME composite ``full_field_space``, whose
metric the :class:`~orpheus.numerics.operator.AdjointOperator` reads off
the composite domain. Every composite adjoint that constructs is
therefore metric-correct, whichever leaves it contains. The **surviving**
guard is general and lives on the adjoint axis: any operator with a
non-adjointable operand reports
:attr:`~orpheus.numerics.operator.LinearOperator.is_adjointable`
``= False``, and its :attr:`~orpheus.numerics.operator.LinearOperator.H`
raises :class:`~orpheus.numerics.operator.MissingAdjoint` **eagerly at
construction** — never silently Euclidean. This is
**illegal-states-unrepresentable** (Cardinal Rule 2) realized through the
recursive predicate, orthogonal to the W-D domain plumbing (which serves
the *forward* composition guard).

.. note:: **Supersession — the S/F adjointability update.** An earlier
   version of this section made the full prompt-loss adjoint
   :math:`(L + C - S - N_{2n} - F - B)^{\dagger}` *intentionally
   unreachable* by
   having ``S`` and ``F`` carry **no** ``apply_transpose`` — a
   non-adjointable operand blocked the composite. **That mechanism is
   retired.** The #112 fission dyad-swap :math:`F^{\mathsf T}` and the
   #118 / #276 scattering Euclidean transpose :math:`S^{\mathsf T}`
   (via
   :attr:`~orpheus.transport.operators.transfer.TransferOperator.full_transfer_kernel`,
   named ``ScatteringOperator.full_scatter_kernel`` until #426 step 2)
   gave both leaves a working ``apply_transpose``, so
   :class:`~orpheus.transport.operators.scattering.ScatteringOperator`
   and :class:`~orpheus.transport.operators.fission.FissionOperator` now
   report ``is_adjointable = True`` (``is_invertible`` still ``False`` —
   a source operator is not invertible). The metric-blindness concern is
   moot regardless, for two reasons: **(a)** every leaf carries the
   composite ``full_field_space`` metric (above); and **(b)** the
   within-group loss is never fused into a single
   :math:`(L + C - S - N_{2n} - F - B)` operator —
   :func:`~orpheus.sn.coupled_system.build_within_group_system` returns the
   :class:`~orpheus.sn.coupled_system.WithinGroupSystem` record whose
   ``implicit_operator`` ``(L+C)`` and ``explicit_gains`` ``(S, N2N, B_a)``
   keep ``S`` / :math:`N_{2n}` /
   ``B`` as **lagged gains** and
   ``F`` handled at the eigenvalue / DSA **outer** layer (where the
   adjoint posing row daggers :math:`M = F` as :math:`M^{\dagger}`; see
   :ref:`eigenvalue-posing`), not via a within-group composite adjoint.


The singular trace metric and the pseudo-inverse — exactness
------------------------------------------------------------

The trace block :math:`G_{\rm trace} = |\Omega\cdot\hat n_f|\,w_n` is
**singular**: on a **tangential** ordinate (one with
:math:`\Omega\cdot\hat n_f = 0`, grazing the face) the cosine factor
vanishes, so that diagonal entry of :math:`G` is zero. A literal
:math:`G^{-1}` does not exist. R5 uses the **Moore–Penrose
pseudo-inverse** :math:`G^{+}` — :math:`1/G` where :math:`G \neq 0`, and
:math:`0` on the null space (:meth:`FunctionSpace.apply_inverse_metric
<orpheus.numerics.space.FunctionSpace>`):

.. code-block:: python

   nonzero = wb != 0.0
   return np.where(nonzero, x / np.where(nonzero, wb, 1.0), 0.0)

This is not an approximation — it is **exact** for the adjoint, by the
following argument. The pseudo-inverse zeroes the tangential components
of the adjoint output. That is the correct value because the tangential
trace slots are **identically zero in every matvec output** in the first
place: the boundary inflow / outflow selectors
(:meth:`AngularTraceSpace.outflow_indices_for_face
<orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace>`) classify an ordinate
as inflow (:math:`\Omega\cdot\hat n < -\epsilon`), outflow
(:math:`> +\epsilon`), or **tangential** (:math:`|\cdot|\le\epsilon`) —
and the boundary operators read/write **only** the inflow and outflow
slots. The tangential slots are never sourced. Consequently:

* the tangential components of :math:`A\psi` are zero (no operator
  writes them), so they carry **zero weight** in
  :math:`\langle\cdot,\cdot\rangle_G` anyway (:math:`G_{\rm trace} = 0`
  there);
* the pseudo-inverse returns zero on exactly those components, so the
  reconstructed adjoint output :math:`G^{+} A^{\mathsf T}(G\varphi)`
  agrees with the true G-adjoint on the **range** of the operator (the
  inflow ⊕ outflow subspace), and is zero on the orthogonal complement
  (the tangential null space) — which is where the true adjoint is also
  zero.

The pseudo-inverse and the true inverse-restricted-to-the-range
**coincide on the subspace the operator actually touches**, so the
reciprocity :eq:`g-adjoint-reciprocity` holds to round-off (it would
fail only if a matvec ever sourced a tangential slot — which the
selectors forbid). The trace-block residuals in the verification table
below (:math:`\le 3.6\times10^{-15}`) confirm there is no measurable
contamination from the null space.


Numerical evidence
------------------

The defining ground is the dense-probe oracle
``validate_composite_adjoint``
(committed in ``tests/sn/operators/test_g_adjoint_reciprocity.py``). It is
**structurally independent** of the production path: it assembles the
operator's dense matrix by probing :math:`op.\text{apply}` on unit
vectors, builds the diagonal metric :math:`G` **explicitly** from
:math:`V`, :math:`w_n`, and ``trace.inner_product_weights`` (it never
calls :meth:`FullFieldSpace.apply_metric
<orpheus.numerics.spaces.full_field_space.FullFieldSpace>`), and
compares ``op.H.apply(φ)`` against the explicit fold :math:`G^{+}\bigl(
op_{\rm dense}^{\mathsf T}\,(G\cdot\varphi)\bigr)` for
:math:`op = (L + C - B)`.

.. list-table:: ``op.H`` vs. the explicit block-diagonal G-fold :math:`G^{+}(op^{\mathsf T}(G\varphi))`, :math:`op = L + C - B`
   :header-rows: 1
   :widths: 20 22 22 36

   * - Geometry
     - bulk block :math:`|\Delta|_\infty`
     - trace block :math:`|\Delta|_\infty`
     - G-reciprocity :math:`\langle op\,\psi,\varphi\rangle_G = \langle\psi, op.H\,\varphi\rangle_G` (rel)
   * - slab (2-group)
     - :math:`7.1\times10^{-15}`
     - :math:`2.5\times10^{-16}`
     - :math:`6.5\times10^{-17}`
   * - sphere (2-group)
     - :math:`1.7\times10^{-13}`
     - :math:`3.6\times10^{-15}`
     - :math:`1.6\times10^{-15}`
   * - cylinder
     - :math:`2.8\times10^{-14}`
     - :math:`1.8\times10^{-15}`
     - :math:`6.8\times10^{-17}`

Both blocks match the explicit fold to round-off, and the defining
reciprocity holds to :math:`\le 1.6\times10^{-15}` across all three
geometries. Because the oracle's :math:`G` is built from raw mesh
quantities while production's :math:`G` is built by
:class:`FullFieldSpace`, the agreement also cross-validates the **metric
population** (a wrong :math:`V` or a transposed :math:`w_n` would shift
the bulk residual off round-off).

**The L11 wrong-metric negative control** is the discriminating gate
that proves the :math:`|\Omega\cdot\hat n|` weighting is **load-bearing**
and not an inert decoration. The control re-evaluates reciprocity under
a *deliberately wrong* trace metric — the angular weight :math:`w_n`
alone, with the :math:`|\Omega\cdot\hat n|` cosine factor dropped — while
``op.H`` is still the adjoint built for the **true** metric. A
correct-but-redundant weighting would leave reciprocity intact; a
load-bearing one must break it:

.. list-table:: L11 control — drop :math:`|\Omega\cdot\hat n|` from the trace metric (must break reciprocity, :math:`\gg 10^{-3}`)
   :header-rows: 1
   :widths: 30 35 35

   * - Geometry
     - reciprocity residual (rel)
     - verdict
   * - slab
     - :math:`6.4\times10^{-2}`
     - **broken** (:math:`\gg 10^{-3}`)
   * - sphere
     - :math:`8.3\times10^{-1}`
     - **broken** (:math:`\gg 10^{-3}`)
   * - cylinder
     - :math:`1.9\times10^{-3}`
     - **broken** (:math:`> 10^{-3}`)

All three break by orders of magnitude relative to the round-off
reciprocity of the correct metric — the cosine factor is doing real
work. (The cylinder margin is the smallest because its single curved
face has the narrowest spread of :math:`|\Omega\cdot\hat n|` over the
ordinate set; the slab and sphere carry the decisive controls, which is
why the L11 test gates on slab and sphere specifically.)

These results are pinned by two foundation-tagged test files:

* ``tests/sn/operators/test_g_adjoint_reciprocity.py`` —
  the G-adjoint reciprocity on slab / sphere / cylinder / slab-2g /
  sphere-2g (5), a metric-population cross-check that
  ``op.codomain.inner_product`` matches an independent reference built
  directly from ``omega_dot_n`` / ``volumes`` (5), and the L11
  wrong-metric control on slab / sphere (2). The reciprocity inner
  products are evaluated with an **independent** Gram fold so a wrong
  *metric* cannot mask a wrong *adjoint*.
* ``tests/numerics/test_full_field_space.py`` — pins the
  :class:`FullFieldSpace` identity semantics (flat direct-sum ``shape``,
  ``(name, shape)``-only identity with ``compare=False`` block
  metadata — the composite carries no ``axes``, so the 2026-09-07
  identity flip leaves this arm untouched — dict-key usability, the
  composite's own ``inner_product_weights`` staying ``None``).

Both files are ``@pytest.mark.foundation`` (software invariants over the
operator-algebra ground truth — they carry **no** ``verifies()`` label
because there is no solver-level theory equation being checked; the
defining identity is the algebra of :eq:`g-adjoint-definition` itself,
anchored to the structurally-independent oracle, not to a discretization
claim). The **forward path is bit-identical**: the :term:`diamond-difference <diamond difference>`
regression suite (69 passed) confirms the adjoint addition does not
perturb the forward matvec — R5 added a *new* capability (``.H``), it
did not touch ``apply``.

.. note::

   This section discharges the **adjoint-metric** half of the "what
   remains for O.2" list in :ref:`bc-extraction-operator-output-o2`
   together with **Gate-1.3** (the O.2 adjoint verification gate). The
   remaining open item is the **residual column** —
   :meth:`AngularBoundaryResidual.from_balance
   <orpheus.transport.residuals.angular_boundary_residual.AngularBoundaryResidual.from_balance>`
   has no operator-output consumer until the O.2 named-composition
   driver types the affine boundary balance at the solver level. The
   G-adjoint of this section is what gives the
   :ref:`adjoint posing row <eigenvalue-posing>` its
   :math:`G`-weighted transpose for free — the daggered eigenvalue
   problem :math:`A_{\rm loss}^{\dagger}\psi^{\dagger} = \lambda
   M^{\dagger}\psi^{\dagger}` now has a concrete, verified ``.H`` to
   build on.
