.. _sn-solver-operator-algebra-coordinator:

SNSolver as an operator-algebra coordinator
============================================

This chapter is where the book's machinery converges: the operators
the preceding chapters built — the swept :math:`(L+C)`, the scattering
:math:`S`, the boundary gain :math:`B`, the fission :math:`F` —
coordinated into the production eigenvalue solve, and, from the
converged flux, the frame projections (homogenisation, condensation)
that hand a coarse problem back to the same solver.

At construction time :class:`~orpheus.sn.solver.SNSolver` caches exactly
**three** operators — the ones that are cross-section read-through, and
therefore survive a rebind untouched:

* :attr:`SNSolver.scattering_op` —
  :class:`~orpheus.transport.operators.scattering.ScatteringOperator`
  carrying the P0 in-scatter + the Pℓ Galerkin reconstruction (Wave D
  Issue 13).
* :attr:`!SNSolver.n2n_op` —
  :class:`~orpheus.transport.operators.n2n.N2NOperator`, the
  :math:`(n,2n)` emission.  It was a **passenger inside**
  ``scattering_op`` until CS4c step 3 (2026-08-30), when the channel
  became first-class because its bundling — scattering-like or
  production-like — is context-dependent and must not be decided at
  the operator level (:ref:`n2n-reactions`, :ref:`sn-n2n-adjoint`).
  Both within-group builds are threaded the pair, and every
  ``(n,2n)`` verb on the solver — since #448 that is the group-rate
  accumulations alone; the ``_add_n2n_source`` delegator retired with
  the hand-built finalize source that was its only caller — routes
  through its energy binding's field.
* :attr:`SNSolver.fission_op` —
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`
  carrying the rank-1-in-energy fission emission (Wave D Issue 13).
  It held the ANGULAR binding
  :class:`~orpheus.transport.operators.fission.FissionOperator` on the
  composite ``full_field_space`` until CS4c step 4 (2026-08-30), when
  the fission channel became two bindings of one datum and every
  consumer was re-pointed at the one it actually feeds: the k-outer
  hands this operator bare :math:`(n_g, *\text{spatial})` scalar arrays,
  so it binds the mesh's **bulk** space and
  :meth:`~orpheus.sn.solver.SNSolver.compute_fission_source` is a thin
  delegator to its ``apply``.  The angular binding is still minted —
  once, at the eigen-:math:`M` posing site, where a composite operator
  is what the pencil needs (:ref:`sn-fission-binding-adjoint`).

The loss composite :math:`L+C` is deliberately **not** cached on the
solver.  Its one spelling is
:func:`~orpheus.sn.coupled_system.build_streaming_collision`, reached
through :func:`~orpheus.sn.coupled_system.build_within_group_system`,
which builds the composite it actually inverts — so a solver-held second
copy would be a twin free to drift from the operand the sweep uses.

.. note::

   Before 2026-07-28 the solver also exposed an ``(L, S, F)`` "operator
   triple" (``SNSolver.L`` / ``.S`` / ``.F``).  It was **retired**: the
   attributes had no production reader, and ``SNSolver.L`` was a
   misnomer — it held the *composite* :math:`L+C`, whereas :math:`L`
   throughout this book is the :math:`\sigma`-free streaming leaf
   (:ref:`the affine collision split <operator-algebra>`).  Consumers
   needing the composite call
   :func:`~orpheus.sn.coupled_system.build_streaming_collision`
   directly.

Each of the cached operators is a
:class:`~orpheus.numerics.operator.LinearOperator`
in the Wave A operator-algebra sense: predicate-typed, composable
under :class:`~orpheus.numerics.operator.OperatorSum` and
:class:`~orpheus.numerics.operator.OperatorProduct`, and protocol-
conforming so the iteration primitives in
:mod:`orpheus.numerics.iteration` consume them without SN-specific
plumbing.  The within-group inner solve is built once from a single
source of truth — the :func:`~orpheus.sn.coupled_system.build_within_group_system`
builder assembles the :class:`~orpheus.sn.coupled_system.WithinGroupSystem`
record, the honest within-group decomposition :math:`(L+C,\ S,\ B)` as a
named splitting :math:`A = M - N`: the invertible resolvent
:math:`M = (L+C)` plus its two lagged coupling gains :math:`N = (S,\ B_a)`
(the bulk scattering :math:`S` and the trace boundary reflection :math:`B`;
zero within-group fission), handed to the **variadic** driver
:math:`\text{Driver}(L_{\rm resolvent},\,*\text{gains})` (Wave O step
O.2a — the transitional :math:`S + B` fold is retired; see
:ref:`bc-extraction-variadic-driver` in :doc:`/theory/foundations/boundary_conditions`).
:func:`_within_group_krylov` wraps the matching
:class:`~orpheus.numerics.iteration.KrylovAcceleration` — and the
decomposition is shared verbatim across the eigenvalue source-iteration
inner (:meth:`SNSolver._solve_source_iteration`), the eigenvalue Krylov
inner (:meth:`SNSolver._solve_krylov`), and both fixed-source paths.

.. admonition:: Key Facts
   :class: tip

   * The within-group system is built ONCE, from a single source of
     truth (:func:`~orpheus.sn.coupled_system.build_within_group_system`):
     the named splitting :math:`A = M - N` — the invertible resolvent
     :math:`M = (L+C)` plus the lagged gains :math:`N = (S,\ B)`.
     Fission is never inside the swept operator; it enters as the
     :math:`1/k`-scaled outer source.
   * Both inner paths — source iteration and Krylov — consume that SAME
     decomposition over the SAME one-walk discretization (matvec ≡
     :term:`sweep`, #206 Phase C): the same solution **set**, different
     rate and memory.  On a closed reflective diamond box that set is a
     *manifold*, not a point — :math:`A` is exactly singular there — so
     the two arms return different **members** and every exit projects
     onto the canonical one (:ref:`sn-loss-kernel-gauge`,
     :ref:`sn-exit-gauge`).
   * :meth:`~orpheus.sn.solver.SNSolver.compute_keff` reports **fission
     production over net removal** (:eq:`sn-keff-update`) — the
     eigenvalue of the map the inner solve actually poses (#291, #259).
     Leakage reads the typed boundary trace; reflective faces are a
     structural zero, so lattice anchors hold to the ULP.
   * Homogenisation and condensation are the frame page's
     Petrov-Galerkin consumers; the SN layer only orchestrates:
     :meth:`Solution.homogenize <orpheus.sn.solution.Solution.homogenize>`
     returns a mesh-coupled ``MaterialMesh``;
     :meth:`Solution.condense <orpheus.sn.solution.Solution.condense>`
     returns a portable ``dict[int, Mixture]`` — the
     condense/homogenize asymmetry law
     (:ref:`sn-condense-homogenize-asymmetry`).


The within-group inner solve consumes the primitives directly
-------------------------------------------------------------

:class:`SNSolver`'s within-group inner solve **is** the
:class:`~orpheus.numerics.iteration.SourceIteration` /
:class:`~orpheus.numerics.iteration.KrylovAcceleration` primitive — not
a verbatim replica of its loop.
:meth:`SNSolver._solve_source_iteration` constructs a
:class:`SourceIteration` from the :func:`~orpheus.sn.coupled_system.build_within_group_system` SSoT and
runs it; :meth:`SNSolver._solve_krylov` constructs a
:class:`KrylovAcceleration` from :func:`_within_group_krylov` and runs
that.  The Layer-3 resolvent of the SN row in the
:ref:`eigenvalue-posing` architecture is exactly these primitive
instances.

The primitive is **type-agnostic and angular-capable**: it operates on
the typed :class:`~orpheus.transport.timed_full_field.TimedFullField`
composite, which carries the full :term:`angular flux` on its bulk.  Pℓ
anisotropic scattering therefore rides the angular bulk with no special
plumbing — :meth:`ScatteringOperator.apply` on the timeless
:class:`~orpheus.transport.full_field.FullField` operator carrier (the
driver's :class:`~orpheus.transport.timed_full_field.TimedFullField` iterate
reaches it via MRO) reads the angular moments off the composite and builds
the anisotropic source inside :meth:`ScatteringOperator.apply
<orpheus.transport.operators.transfer.TransferOperator.apply>` — the
:math:`\ell \ge 1` half being the redistribution body the binding's ends
select (``TransferOperator._redistribute_ordinates`` on the angular end) —
all inside the primitive's normal RHS path.  There is **no scalar-flux
limitation** and **no pending "Approach A" cleanup**: the earlier
framing — that :class:`SourceIteration` carried only :term:`scalar flux` and SN
had to replicate the loop verbatim until the angular state could be
threaded through — was a property of an interim scalar-only carrier
that the typed composite retired.  The
``.claude/skills/algebra-of-record`` "Branch 2 implements the same
operator algebra" discipline is satisfied: SN is the discretized
Branch-2 consumer of the shared primitive, not a parallel loop.

The (L + C − S − N₂ₙ − B)·ψ = (1/k)·F·ψ framing at the solver level
-------------------------------------------------------------------

Beyond driving the within-group inner solve, the :math:`(L+C,\ S,\ F)`
framing organises the solver's outer API surface:

* :meth:`SNSolver.compute_fission_source` returns
  :math:`F\,\phi/k` — a thin delegator to ``F.apply`` with the
  :math:`1/k` outer-loop scaling applied at the solver level.
* :meth:`SNSolver.solve_fixed_source` solves
  :math:`(L+C-S-N_{2n}-B)\,\psi = q_{\rm ext}`
  (:eq:`sn-within-group-with-n2n`; with :math:`q_{\rm ext}` the
  fission source built by ``compute_fission_source``).  Two paths:

  * ``inner_solver="source_iteration"`` — sweep-driven fixed-point
    iteration; the resolvent :math:`(L+C)^{-1}` is realised by the
    one-walk WDD sweep.
  * ``inner_solver="krylov"`` — GMRES on the honest within-group
    matvec, with the sweep resolvent as preconditioner — the same
    one-walk discretization either way (matvec ≡ sweep, #206 Phase C).

* :meth:`SNSolver.compute_keff` returns **fission production over net
  removal**, :eq:`sn-keff-update` — the volume-weighted method-layer
  functional :math:`R_{\nu\Sigma_f}(\phi) / (R_{\Sigma_a}(\phi) + L -
  E_{2n}(\phi))`, derived in :ref:`sn-keff-estimator` below.  The
  SN-specific volume weighting lives here (in the typed
  :class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`
  fields) — one honest realization of the same discipline the
  operator-form :meth:`KEigenvalue.compute_keff` spells with the
  measure absorbed into the operators' action.  (Pre-#291 this method
  returned the leakage-blind :math:`\sum F\phi V / \sum \Sigma_a\phi V`
  ratio; see :ref:`sn-keff-estimator` for why that was a
  non-eigenvalue on any vacuum-bounded problem.)

The solver-level :math:`1/k` scaling (in
:meth:`~SNSolver.compute_fission_source`) and the volume-weighted
eigenvalue estimate (in :meth:`~SNSolver.compute_keff`) are exactly the
points where SN's specifics live; the rest of the solver is
operator-algebra coordination over the canonical
:func:`~orpheus.numerics.eigenvalue.power_iteration` boundary.  These
two K-specific hooks are also precisely *why* the Layer-4 loop is not
yet literally K/α-agnostic — relocating the eigenvalue scaling into the
algorithm is the first step of the α-wave (see the honest-scope caveat
in :ref:`eigenvalue-posing`).

The eigenvalue :math:`\keff` is determined by **power iteration**: an
outer loop updates :math:`k` from the net-removal balance
:eq:`sn-keff-update` (fission production over absorption + leakage −
:math:`(n,2n)` emission), with an inner loop that solves the
within-group scattering problem.

.. _sn-keff-estimator:

The reported eigenvalue: fission production over net removal
------------------------------------------------------------

:meth:`~orpheus.sn.solver.SNSolver.compute_keff` reports the eigenvalue
of the problem the inner solve **actually poses**.  This is the SN
symptom (#291) and the MoC/CP/homogeneous root (#259) of a single
discipline: *the reported* :math:`k` *must be the eigenvalue of the
fixed-source map every method scales only fission by* :math:`1/k`
*through* — scattering and the :math:`(n,2n)` emission are plain gains
assembled **inside** :meth:`~orpheus.sn.solver.SNSolver.solve_fixed_source`,
never scaled by :math:`1/k`.  An estimator that disagrees with its own
posed problem converges cleanly and silently to a **non-eigenvalue
ratio**.

.. math::
   :label: sn-keff-update

   k \;=\; \frac{R_{\nu\Sigma_f}(\phi)}
                {R_{\Sigma_a}(\phi) \;+\; L \;-\; E_{2n}(\phi)}

.. (V&V scope note) Governing/definitional identity: the reported k
   IS the eigenvalue of the posed fixed-source map, not a solver
   eigenvalue-correctness claim against an external analytical reference
   (that rests on the multi-group heterogeneous L1/L2 references
   elsewhere on this page). The label is wired to the cross-engine
   consistency gate tests/sn/eigenvalue/test_keff_estimator_gate.py
   (reported k == the converged fixed-point map ratio k* = P(Mφ*)/P(φ*),
   map-ratio ground-truth noise ≤ 2e-11) with in-file mutation teeth.

The three terms are typed volume-integrated reaction-rate functionals
and one boundary functional:

* **Numerator** :math:`R_{\nu\Sigma_f}(\phi) = \int_V \nu\Sigma_f\,\phi\,dV`
  — the fission production, the typed
  :class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`
  over :math:`\nu\Sigma_f` (the :math:`\phi^\dagger\!=\!1` degenerate of
  the homogenization Petrov–Galerkin bilinear).  The :math:`(n,2n)`
  emission is **not** production here — contrast
  :meth:`~orpheus.sn.solver.SNSolver.compute_production_rate`, the
  ERR-052 renormalisation scale anchor, which keeps *total* physical
  production (fission **plus** the :math:`(n,2n)` emission).  The role
  split is the load-bearing #259 correction: the same physical
  :math:`(n,2n)` neutrons are a **scale** quantity for the outer
  renormalisation but a **removal-reduction** in the eigenvalue balance.
* **Absorption** :math:`R_{\Sigma_a}(\phi) = \int_V \Sigma_a\,\phi\,dV`,
  with :math:`\Sigma_a = \Sigma_f + \Sigma_c + \Sigma_L +
  \sum_{g'}\Sigma_{2,g\to g'}` — i.e. ``absorption_xs`` counts the
  :math:`(n,2n)` **collision once** (the neutron is removed from its
  incident group by the collision).  See
  :attr:`~orpheus.data.macro_xs.mixture.Mixture.absorption_xs`.
* **Leakage** :math:`L` — the net vacuum-boundary outflow (below).  On a
  reflective (lattice) problem it is a **structural zero**.
* **Emission** :math:`E_{2n}(\phi) = \int_V \sum_{g,g'} 2\,\Sigma_{2,g'\to
  g}\,\phi_{g'}\,dV` — the :math:`(n,2n)` **emission** (two neutrons out
  per collision; the factor 2).  A gain, so it **reduces** net removal.

The net :math:`(n,2n)` effect on removal is therefore
:math:`\underbrace{\sum_{g'}\Sigma_{2,g\to g'}}_{\text{collision, in }\Sigma_a}
- \underbrace{2\Sigma_2}_{E_{2n}} = -\Sigma_2` — **one extra neutron
gained** per collision, exactly the physics of a neutron-doubling
reaction.

**Balance identity (divergence-telescoping).**  The angle- and
group-summed discrete cell balance for cell :math:`i` in the posed
eigenproblem is

.. math::
   :label: sn-keff-cell-balance

   \underbrace{\sum_{f\in\partial i}\!\bigl(\textstyle\sum_g J_g\bigr)\,\Delta A_f}
              _{\text{net face flow}}
   \;+\; \Sigma_{t,i}\,\phi_i\,V_i
   \;=\; \frac{1}{k}\,R_{\nu\Sigma_f,i}
        \;+\; \Sigma_{s,i}\,\phi_i\,V_i
        \;+\; E_{2n,i}

.. (vv-status rationale) Derivation step (the divergence-telescoping cell
   balance). Its terminal result sn-keff-update is verified by the k* map-ratio
   gate (tests/sn/eigenvalue/test_keff_estimator_gate.py); definitional.
.. vv-status: sn-keff-cell-balance documented

(streaming + total collision on the left; the isotropic fission source
scaled by :math:`1/k`, plus the *unscaled* scatter and :math:`(n,2n)`
gains, on the right).  Summing over **all** cells, every interior face
is shared by two cells with opposite outward normals and equal current
(continuity), so the interior face-flow terms **telescope to zero** —
only the domain-boundary faces survive, and their sum is the net
leakage :math:`L`.  With :math:`\Sigma_t - \Sigma_s = \Sigma_a` this
collapses to

.. math::

   \frac{R_{\nu\Sigma_f}(\phi)}{k} \;=\; R_{\Sigma_a}(\phi) \;+\; L
                                        \;-\; E_{2n}(\phi),

which is :eq:`sn-keff-update` rearranged.  This is the same discrete
divergence discipline the diffusion page states as
:math:`\mathbf 1^{\mathsf T}(C-S)=\Sigma_a` with interior-face
telescoping (see :ref:`diffusion-leakage-boundary-leaves`); SN and
diffusion report the *same* balance-law eigenvalue, differing only in
how the streaming operator is discretised.

The leakage functional
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
   :label: sn-leakage-functional

   L \;=\; \sum_{f\,\in\,\text{vacuum}} \oint_{f} dA\,
           \sum_g J_g(\mathbf{r}_f)\,,
   \qquad
   J_g \;=\; \sum_m (\Omega_m\cdot\hat n_f)\, w_m\, \psi_{m,g}

is the face-area integral of the boundary trace's **net outward
current**.  The angular-to-scalar reduction :math:`J_g` is
:meth:`AngularBoundaryFlux.net_current
<orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux.net_current>`
— the single source of the :math:`\Omega\cdot\hat n\,w` contraction, the
angular sibling of the scalar trace's :math:`J = J^+ - J^-`
(:meth:`ScalarBoundaryFlux.net_current
<orpheus.transport.fields.scalar_boundary_flux.ScalarBoundaryFlux.net_current>`).
It is spelled through the trace space's own atoms — the signed
projection table
:attr:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.omega_dot_n`
and the :math:`|\Omega\cdot\hat n|\odot w` partial-current metric (using
the identity :math:`\operatorname{sign}(\Omega\cdot\hat n)\cdot
|\Omega\cdot\hat n|\,w = \Omega\cdot\hat n\,w`) — so no consumer
re-derives the cosine weighting.  Tangential :term:`ordinates <ordinate>` carry zero weight
and drop out.

The face measure :math:`dA` is supplied by
:meth:`SNSolver._face_area_of`, matching the cell
``volume_measure`` exactly so the balance identity closes:

.. list-table:: Boundary-face measure by geometry
   :header-rows: 1
   :widths: 30 30 40

   * - Geometry
     - Face measure :math:`\Delta A`
     - Source
   * - 1-D slab
     - :math:`1` (per unit cross-section)
     - :attr:`MaterialMesh.areas <orpheus.transport.mesh.material_mesh.MaterialMesh.areas>`
   * - 1-D cylinder
     - :math:`2\pi R` (per unit height)
     - ``MaterialMesh.areas``
   * - 1-D sphere
     - :math:`4\pi R^2`
     - ``MaterialMesh.areas``
   * - 2-D Cartesian
     - transverse edge-cell widths (unit depth)
     - ``mesh.axes`` transverse extent
   * - 3-D Cartesian
     - :math:`\Delta A_{\mathbf c} = \prod_{j\ne a}\Delta_j[c_j]`
       (transverse-area outer product)
     - ``mesh.axes`` transverse extents

The :math:`d \ge 2` Cartesian arms are ONE generic body: the outer
product of the *other* axes' edge widths in **ascending axis order** —
the same codimension-1 enumeration as
:func:`~orpheus.transport.mesh.axis.face_shape`, so the measure array
broadcasts cell-for-cell against the ``(ng, *face_spatial)`` net
current, and the 2-D width vector is just the single-transverse-axis
degenerate (bit-identical to the pre-3-D spelling).

The 3-D arm originally shipped as a **typed refusal**
(``NotImplementedError``): guessing the transverse product's cell
ordering would silently mis-weight the leakage sum, and Cardinal Rule 1
forbids returning a wrong-but-plausible number.  The wire landed
2026-07-13 when the first 3-D vacuum eigenvalue consumer arrived (the
d=3 Mode-9 G-S≡Jacobi gate), with the ordering pinned twice in
``tests/sn/eigenvalue/test_keff_estimator_gate.py``: an **object-level
pin** (face measure ≡ the boundary layer's ``volumes / Δ_axis``, the
mesh's own ascending-axis enumeration — vv Mode-12 discipline: pin the
object, not only the k functional) and the **k* map-ratio gate** on a
Mode-2 asymmetric all-vacuum box, whose teeth are proven by permanent
in-process mutants — a reversed transverse enumeration moves the
reported k by a measured **13.9 %** against the estimator-independent
:math:`k^*` (clean agreement :math:`6\times10^{-10}`), and a transposed
enumeration crash-REDs on the broadcast.  A trace carrying a
``#251`` transverse face-moment tail is refused loudly at the
consumption site (the face integral must consume ONLY the
transverse-average moment — higher Legendre face moments integrate to
zero over each face cell — and that slot-0 read has no consumer yet).

Reflective faces are a structural zero
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reflective law equates a face's inflow to its reflected outflow
**exactly**, so the net current there vanishes *by construction*.
:meth:`~orpheus.sn.solver.SNSolver._boundary_leakage_rate` therefore
**skips** reflective faces — it never accumulates them, rather than
accumulating a value that ought to be zero but carries
:math:`\pm`-cancelling angular-sum floating-point noise.

This is a deliberate design choice with a bit-level payoff.  On an
all-reflective (lattice) problem :math:`L` is a structural ``0.0``, and
on a :math:`\Sigma_2`-free mixture :math:`E_{2n}` is exactly ``0.0`` (the
per-material :math:`(n,2n)` loop adds nothing), so
:eq:`sn-keff-update` reduces **bit-identically** to the historical
lattice functional ``production / absorption``.  Every pre-existing
reflective eigenvalue anchor is preserved to the last ULP — the
unification adds terms that vanish structurally, not numerically, on the
cases it must not perturb.

The scale bridge: trace of the last inner solve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The leakage term reads the **typed** boundary trace of the last inner
solve (``self._inner.iterate.boundary`` — the
:class:`~orpheus.sn.solver.InnerSolve` record each inner leaves behind),
whereas the numerator/denominator
reaction rates consume the bare-array flux :math:`\phi` the estimator is
handed.  These two representations can be at **different scales**:
:func:`~orpheus.numerics.eigenvalue.power_iteration` renormalises
:math:`\phi` to unit production rate *between* the inner solve and the
:math:`k`-update (ERR-052), so the stored trace belongs to the
**un-renormalised** last iterate while the estimator's :math:`\phi` is
its renormalised multiple.

Leakage is degree-1 homogeneous in :math:`\psi`, so the fix is a single
rescale by the fission-production ratio of the two fluxes
(``self._phi_of_trace``, stored alongside the trace at **both**
inner-path returns) — exactly ``1.0`` when the caller passes the
returned flux itself.  The **contract** is therefore: the flux handed to
:meth:`~orpheus.sn.solver.SNSolver.compute_keff` must be a scalar
multiple of the last inner solve's flux (true for ``power_iteration`` and
for every manual solve-then-estimate loop).

If a vacuum face exists but **no** inner solve has stored a trace,
:meth:`~orpheus.sn.solver.SNSolver._boundary_leakage_rate` raises
``RuntimeError`` — the leakage cannot be answered honestly, and silently
returning it as zero would *reproduce the #291 omission*.  Fail loud;
never return a non-eigenvalue.

The R7 :math:`(n,2n)` convention fork
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The historical spelling put the :math:`(n,2n)` emission in the
**numerator** as production,

.. math::
   :label: sn-keff-old-n2n

   k_{\text{old}} \;=\; \frac{R_{\nu\Sigma_f} + E_{2n}}{R_{\Sigma_a}},

.. (vv-status rationale) Definitional/historical contrast: the superseded
   (n,2n)-in-numerator spelling. No current code implements it; its bias is
   characterised in the pre-fix table (#291 commit d1daaac).
.. vv-status: sn-keff-old-n2n documented

which is a **non-eigenvalue** of the posed map whenever
:math:`\Sigma_2 \neq 0` *and* :math:`k \neq 1`.  The reason is exactly
the posing asymmetry: the inner solve scales **only** fission by
:math:`1/k`; the :math:`(n,2n)` emission is an *unscaled* gain in the
sweep source.  So the eigenvalue of that map is
:math:`k^\star = R_{\nu\Sigma_f}/(R_{\Sigma_a} - E_{2n})` (reflective,
:math:`L=0`), and putting the *unscaled* emission in the numerator does
not recover it.  Writing :math:`f = R_{\nu\Sigma_f}`,
:math:`a = R_{\Sigma_a}`, :math:`e = E_{2n} = 2s_2` and substituting
:math:`f = k^\star (a - e)`:

.. math::
   :label: sn-keff-old-bias

   k_{\text{old}}
   \;=\; \frac{k^\star (a - e) + e}{a}
   \;=\; k^\star \;+\; \frac{2 s_2\,(1 - k^\star)}{a}.

.. (vv-status rationale) Mathematical identity (the derived bias of the
   superseded estimator, k_old - k* = 2 s_2 (1 - k*) / a). Historical
   characterisation; no current implementing code.
.. vv-status: sn-keff-old-bias documented

The two agree only when :math:`s_2 = 0` (no :math:`(n,2n)`) or
:math:`k^\star = 1` (critical).  For a supercritical
:math:`k^\star > 1` the correction is negative
(:math:`k_{\text{old}} < k^\star`); for subcritical, positive.  The MoC
and CP pages carry the same fork
(:eq:`moc-keff-update`, :eq:`cp-keff-update`); CP was the one member
already spelled on net removal.

What was tried and found
~~~~~~~~~~~~~~~~~~~~~~~~~~

The #291 bias was characterised pre-fix (commit ``d1daaac``, Gauss–
Legendre :math:`n=8`, map-ratio ground truth noise :math:`\le 2\times
10^{-11}`) across the five gate configurations:

.. list-table:: Pre-fix reported :math:`k` vs the posed-problem eigenvalue :math:`k^\star`
   :header-rows: 1
   :widths: 40 18 18 24

   * - Configuration
     - Pre-fix reported
     - Posed :math:`k^\star`
     - Bias
   * - homog. 2G vacuum slab (width 8)
     - 1.83767525
     - 0.98163269
     - :math:`+87.2\%` (:math:`L/A = 0.872`)
   * - het. vacuum sphere P\ :sub:`0`
     - 0.86484694
     - 0.70601977
     - :math:`+22.5\%`
   * - het. vacuum sphere P\ :sub:`1`
     - 0.85080423
     - 0.67876772
     - :math:`+25.3\%`
   * - reflective control (:math:`\Sigma_2=0`)
     - 1.87500000
     - 1.87500000
     - :math:`\equiv` (bias :math:`1.2\times 10^{-10}`)
   * - reflective :math:`\Sigma_2\neq 0`
     - 1.92857143
     - 2.61278195
     - :math:`-26.2\%` (R7 defect)

The two failure classes are visible in one table: the vacuum rows are
the **leakage omission** (#291) — the reported :math:`k` overshoots by
the leakage-to-absorption ratio :math:`L/A`; the last row is the **R7
:math:`(n,2n)` convention** — zero leakage, yet a
:math:`-26.2\%` error because the emission was mis-posed as production.
The reflective-control row is exactly the bit-identity guarantee above.
The exact check on the R7 row is
:math:`0.78/(0.5185 - 0.2200) = 2.61278`, and
:math:`(0.78 + 0.2200)/0.5185 = 1.92857` reproduces the old value —
matching :math:`k_{\text{old}} = k^\star + 2s_2(1-k^\star)/a` term for
term.

Post-fix, reported :math:`k` and the map-ratio :math:`k^\star` agree to
:math:`\le 6\times 10^{-10}` on all five.  The P\ :sub:`0`\ –P\ :sub:`1`
sphere gap :math:`\Delta` roughly **doubled** (:math:`1.404\times
10^{-2} \to 2.725\times 10^{-2}`) but stays inside the diagnostic
:math:`(10^{-3}, 5\times 10^{-2})` band — the P\ :sub:`1` anisotropic
correction is now measured against the correct eigenvalue on both
solves.

The V&V decision was a **principled re-baseline** (per ``vv-principles``
bit-identity-vs-principled-equivalence): the old reported :math:`k` was a
*different functional* from the posed problem's eigenvalue, so the new
value is not a regression to be tolerance-matched but a correction to be
verified against a structurally-independent reference (the fixed-point
map ratio).

Verification
~~~~~~~~~~~~

The permanent gate is
``tests/sn/eigenvalue/test_keff_estimator_gate.py``: it asserts the
reported :math:`k` equals the converged fixed-point map ratio
:math:`k^\star = P(M\phi^\star)/P(\phi^\star)` across the four physics
regimes — {vacuum slab, vacuum sphere (pinning the :math:`4\pi R^2`
face-area convention), reflective bitwise-degenerate, reflective
:math:`\Sigma_2\neq 0`} — with **in-file mutation teeth**: a
leakage-drop mutation reds the vacuum legs while staying bitwise-green
on reflective; a leakage sign-flip crash-reds through the scale-bridge
guard; and the old :math:`(n,2n)`-in-numerator convention reds the
:math:`\Sigma_2\neq 0` leg.

This is a **consistency** gate: the map ratio is the structurally-
independent ground truth for "does the estimator return the eigenvalue
of its own posed map", and is blind by construction to *which*
eigenvalue that is.  The SN solver's eigenvalue **correctness** — that
the posed map's eigenvalue is the *physically right* :math:`k` — rests
on the multi-group heterogeneous L1/L2 references in
:doc:`/theory/verification/sn`, not on this gate.

Two Inner Solvers
-----------------

**Source iteration (sweep-based):**

- Operator: :math:`(L+C)^{-1}` (the one-walk WDD sweep)
- Iterate: the typed field composite (angular bulk + boundary trace)
- Fixed-point: :math:`\psi^{(k+1)} = (L+C)^{-1}(S\,\psi^{(k)} +
  N_{2n}\,\psi^{(k)} + B\,\psi^{(k)} + q_{\rm ext})` — the
  ``explicit_gains`` triple ``(S, N2N, B_a)``
- Convergence rate: spectral radius of
  :math:`(L+C)^{-1}(S+N_{2n}+B)` — the
  :term:`scattering ratio` :math:`c` (:doc:`slab_one_group`)
- Cost per iteration: one transport sweep
- Works for all geometries

**Krylov (direct operator):**

- Operator: the honest :math:`(L+C-S-N_{2n}-B)` applied matrix-free (its
  :math:`(L+C)` piece via :meth:`StreamingCollisionOperator.apply` — the same
  one-walk discretization the sweep realises; L21 matvec ≡ sweep)
- Iterate: the same typed composite; GMRES additionally stores its
  Krylov basis (``restart`` × the composite's ``n_dof`` — the ERR-053
  sizing family)
- System: :math:`(L+C-S-N_{2n}-B)\,\psi = q_{\rm ext}` — scattering,
  the :math:`(n,2n)` gain and the
  boundary gain live in the operator, not the lagged source;
  :math:`q_{\rm ext}` is the :math:`1/k`-scaled fission source
- Convergence: GMRES with sweep preconditioner, typically ~100
  Krylov iterations at ``tol=1e-4`` (always converges)
- Available for all geometries (Cartesian, spherical, cylindrical)

Wave E Round 2 (Issue #164) replaced the legacy BiCGSTAB FD-operator
path with this Krylov path.  See the Krylov alternative in
:doc:`slab_one_group` for the full discussion.

The two paths share the **one** loss-representation discretization
(matvec ≡ sweep, #206 Phase C), so they solve the same system and
converge to the same **solution set**; the Wave-D-era design in which
they carried different spatial closures — and disagreed on coarse-mesh
:math:`\keff` — is recorded in the two-closure history
(:ref:`loss-rep-history`).

.. note::

   ⛔ **That read "the same fixed point" until 2026-08-15 (#344), and a
   set is not a point.**  On a closed reflective diamond box
   :math:`A = L+C-S-N_{2n}-B` is **exactly singular**, so the two arms
   legitimately return different members of a solution manifold:
   ``[M]`` on an all-reflective 2-D absorber box with a uniform
   isotropic source, source iteration under boundary Gauss-Seidel
   returns a trace carrying :math:`6.08\times10^{-2}` of
   :math:`\ker A` while Krylov carries :math:`4.1\times10^{-14}`, and
   the difference between them lies **entirely** in :math:`\ker A`
   (:math:`\lVert\Pi d\rVert/\lVert d\rVert = 1.000000`).  The **bulk**
   really is arm-invariant — the kernel is pure-trace — and every entry
   now projects the returned trace onto the canonical member, so the
   sentence is true again of what a caller receives.  Derivation:
   :ref:`sn-loss-kernel-gauge`; exit behaviour: :ref:`sn-exit-gauge`.

.. _sn-finalize-one-step:

The returned angular flux — one step of the map the iteration drove
-------------------------------------------------------------------

:class:`~orpheus.sn.solution.Solution` ships **two** flux members, and only
one of them is what the power iteration converged.
:attr:`~orpheus.sn.solution.Solution.scalar_flux` *is* the converged
:math:`\phi`; :attr:`~orpheus.sn.solution.Solution.angular_flux` has to be
*reconstructed*, for two independent reasons:

* the outer iteration's contract is scalar — ``[M]`` all **five** members
  of :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` exchange only
  the method's ``Carrier``, which for S\ :sub:`N` is the bare
  ``(n_g, *spatial)`` scalar flux, so no per-ordinate field crosses the
  boundary at all; and
* on the 2-D Cartesian windowed arm the within-group iterate is not
  per-ordinate at all — it is the harmonic-moment composite
  (:ref:`sn-angular-windowing-honest-scope`), and the user-facing
  :math:`(N, n_g, n_x, n_y)` field has to be built from it.

**The reconstruction is one application of the splitting map — not a
solve.**  A within-group splitting writes the loss operator as
:math:`A = M - N` and iterates

.. math::
   :label: sn-finalize-map

   \psi^{(j+1)} \;=\; G\bigl(\psi^{(j)}\bigr)
   \;=\; M^{-1}\Bigl(q + \sum_i N_i\,\psi^{(j)}\Bigr) ,

.. (vv-status rationale) The governing iteration of the within-group
   splitting, stated so the finalize can be derived from it.  Not a solver
   claim: the map is the DEFINITION of what the SI driver does, and the
   verifiable content — that ONE application at the converged iterate
   reproduces that iterate, and that its angular integral reproduces the
   reported scalar flux — is pinned by
   ``tests/sn/solve/test_eigenvalue_finalize_reconstruction.py``
   (``@pytest.mark.catches("ERR-083")``, seven arms × two orders).
.. vv-status: sn-finalize-map documented

which is the source-iteration bullet above written once.  At a fixed point
:math:`G(\psi^\star) = \psi^\star`, so applying :math:`G` **once** to a
converged iterate returns that iterate — and that is the whole
reconstruction.  Nothing is re-solved, nothing is re-selected: the
finalize reads :math:`M`, :math:`\{N_i\}` and :math:`\psi_{\rm conv}` off
the :class:`~orpheus.sn.solver.InnerSolve` record the last within-group
solve left behind, and evaluates
:func:`~orpheus.numerics.iteration.fixed_point_step` on them.

Two things the identity :math:`G(\psi^\star) = \psi^\star` buys are worth
spelling out, because they are what makes ONE step enough rather than a
convenience:

* :math:`M^{-1}` need not be the *representative* the iteration ran.  On
  the windowed arm the driver's step is :math:`P\,M^{-1}` (project to
  moments after the sweep); the record keeps the **un-wrapped** forward
  :math:`M`, so one step through :math:`M^{-1}` alone comes back
  per-ordinate.  A moment iterate is un-windowed by the reconstruction
  *for free*, because the fixed-point identity does not care which
  right-inverse of :math:`M` you use.
* The right-hand side may legitimately differ from the one the iteration
  last saw — which is exactly what the finalize exploits, below.

**The source is the CONVERGED fission source, and that choice is the exit
balance's question.**  The last inner solve ran with
:math:`q = F\phi_{N-1}/k_{N-1}` — the *penultimate* outer's fission source,
because :func:`~orpheus.numerics.eigenvalue.power_iteration` builds
:math:`q` from the iterate it *enters* the outer with and only then
renormalises :math:`\phi` and updates :math:`k`.  The
finalize re-poses with :math:`q_F(\phi_{\rm conv}, k_{\rm conv})`, the
fission source built from the values the caller is actually handed.  The
reason is not tidiness: a caller who checks the returned object is asking
*does this* :math:`\psi` *solve the equation with this* :math:`k` *and this*
:math:`\phi`?, and only the converged source makes the answer yes.  The
shipped exit-balance diagnostic asks exactly that question
(:ref:`sn-exit-balance-projection`), and it would be measuring a
discrepancy of the finalize's own making if the finalize had used the
last inner's source.  At convergence the two sources agree to the outer
tolerance, so the choice costs nothing; at a truncated exit it is the
difference between a diagnostic that tracks the truncation and one that
does not.

**Everything else in the composite arrives as a gain, not as hand-staged
data.**  The lagged couplings are the record's own —
:math:`(S, N_{2n}, B)` on the Jacobi arm, :math:`(S, N_{2n}, B_{\rm upper})`
when the inner ran under the boundary Gauss-Seidel schedule, and the
coupled gain grid on a carrying (curvilinear) mesh.  Three consequences:

* **The reflective boundary is** :math:`B\,\psi_{\rm conv}`, delivered
  through ``rhs.boundary`` exactly as in every inner iterate.  The
  finalize does **not** reflect the converged trace into its own inflow
  slots by hand; the inflow is a solved unknown carried on
  ``ψ.boundary`` (Wave O #208 O.4a.2), and the reconstruction re-derives
  it from the same operator the iteration used.
* **The** :math:`\ell \ge 1` **emission is present because it is the
  gain's**, not because the finalize remembered it.  This is the whole
  content of the #448 repair — see the retirement note below.
* **The schedule the inner chose is the schedule the finalize inherits.**
  A boundary-G-S inner reconstructs through :math:`(L+C-B_{\rm lower})^{-1}`
  with :math:`B_{\rm upper}` lagged, which is a *different* splitting of
  the same :math:`A`; the converged answer must not depend on it, and that
  is a pinned row
  (``TestTheGaussSeidelArmPosesItsOwnSplitting`` →
  ``test_the_schedule_does_not_move_the_converged_answer``).

**On a carrying mesh the pair is reconstructed as a pair.**  The
eigenvalue right-hand side is built once, by
:func:`~orpheus.sn.solver._eigenvalue_driver_source` (``[M]`` **3** call
sites — the SI inner, the Krylov inner, and this finalize).  Its System-A
member is the fission source lifted per-ordinate through
:meth:`AngularSourceSink.from_isotropic
<orpheus.transport.source_sinks.AngularSourceSink.from_isotropic>` with a
**zero** external boundary; its System-B member is the :math:`\ell = 0`
fold of the **fission source alone** as the :math:`\psi_{1/2}` march's
entry (#282 route (a)).  The seed folds fission alone because the coupled
gain grid carries the rest — the ``Emission`` and :math:`B_b` blocks — so
folding the total source there would double-count it.  That is not a
finalize-specific rule: it is what the two inner drivers pass, and the
finalize passes it because it calls the same constructor.

.. note::

   ⛔ ``max_outer = 0`` is a legal call that runs the power loop zero
   times, so there is no within-group solve to reconstruct from and
   :func:`~orpheus.sn.solver.solve_sn` **raises**.  This is a live refusal
   of a reachable state, not decoration: the finalize is one step of the
   iteration, and there is no cold-solve fallback by design — a fallback
   would be a second reconstruction path, which is the twin this section
   exists to have retired.

.. warning::

   ⛔ **Until 2026-09-06 this block built its own source, and it was P0
   only** (:doc:`ERR-083 </theory/verification/error_catalog>`).  It
   assembled :math:`F\phi/k + \Sigma_{s,0}^{\mathsf T}\phi +
   \nu_{2n}\Sigma_{2,0}^{\mathsf T}\phi` through three now-retired
   solver-side delegators and lifted it **isotropically**, so at every
   ``scattering_order ≥ 1`` the :math:`\ell \ge 1` half of *both*
   collision channels was absent from the reconstruction's right-hand
   side while the loss arm the iterate converged against carried it.  The
   returned :math:`\psi` therefore solved a different equation from the one
   the solve converged, and its own angular moment did not reproduce the
   :math:`\phi` shipped beside it: ``[M]`` on the 421-group Be-reflected
   slab at :math:`L = 2`, the returned flux missed the converged iterate by
   **8.776e-02** and missed its own reported :math:`\phi` by **3.405e-02**;
   the one-step reconstruction reads **1.236e-10** and **3.170e-10** on the
   same solve.  :math:`k` and :math:`\phi` were never affected — they are
   the power iteration's — which is precisely why every eigenvalue-value
   gate in the tree was structurally blind to it.

   Two things the repair also removed, both worth recording because their
   absence looks like a regression until you check:

   * the **hand reflect** of the converged trace into its own inflow slots
     (``ψ.inflow ← B·ψ.outflow`` before the sweep).  ``[M]`` on a converged
     exit it was INERT — skipping it moved the answer by 2.0e-13 / 2.3e-15
     and bit-identically on a vacuum arm — because the converged inflow
     already equals :math:`B\,\psi_{\rm outflow}`.  No value gate in this
     tree could witness its removal, so the honest artefact is the
     measurement plus a *wrong*-:math:`B` mutation arm, not a gate.  The
     whole-trace verb itself survives as the sweep-tier gates' inter-sweep
     helper (``tests/sn/_test_helpers.py::reflect_outflow_into_inflow``);
     it has **no production caller** any more.
   * the ``AngularBoundarySourceSink.prescribed_inflow`` cast the finalize
     used to perform on that reflected trace (the ERR-071 role conversion).
     The finalize passes no trace at all now — its external boundary source
     is zero and :math:`B` is a gain — so that call site is moot.  The
     factory and the rest of the ERR-071 fix are untouched.

.. _sn-convergence-contract:

The convergence contract — a best-effort answer says so
--------------------------------------------------------

Every entry above can stop for two structurally different reasons: it
**converged**, or its **budget ran out** and the returned field is a
best-effort iterate, mid-descent.  Both come back as the same type from
the same call, so the distinction has to be carried explicitly or it is
lost.

**Where the fact comes from.**  The loop that stops knows why it stopped,
and since 2026-08-08 it says so rather than discarding it.  Since #340 it
says so by **measuring**, never by asserting: each level reports the
quantities it stops on — magnitude and tolerance together — as a
:class:`~orpheus.numerics.convergence.StoppingCriterion`, and every
convergence verdict in the stack is *derived* from those trajectories.

* the **inner** (fixed-source) fact is the driver's own
  :class:`~orpheus.numerics.convergence.IterationRecord`, returned by
  :class:`~orpheus.numerics.iteration.SourceIteration` and
  :class:`~orpheus.numerics.iteration.KrylovAcceleration` alongside the
  iterate;
* the **outer** (eigenvalue) fact is
  :attr:`~orpheus.numerics.eigenvalue.PowerIterationOutcome.converged`, a
  property over the outer record that
  :func:`~orpheus.numerics.eigenvalue.power_iteration` assembles from the
  readings :meth:`SNSolver.measure_stopping_criteria
  <orpheus.sn.solver.SNSolver.measure_stopping_criteria>` returns each
  outer — ``dk`` against ``keff_tol``, ``dphi`` against ``flux_tol``;
* the outer record carries the inner records as **children**, so the two
  facts compose rather than collapsing.

.. important::

   The two questions are genuinely different and the difference is the
   point.  ``record.converged`` asks whether THIS level met its own
   criteria; :attr:`~orpheus.numerics.convergence.IterationRecord.fully_converged`
   asks whether it and every level beneath it did.  An
   increment-only outer stop cannot see an upstream throttle — a truncated
   inner suppresses the very increments the outer reads, so the outer
   stalls and calls the stall convergence.  `[M]` on a 20-cell 2-group
   :math:`S_8` slab at ``max_inner=1`` the outer reports
   ``converged=True`` with :math:`\keff` wrong by **11×** its own
   ``keff_tol``; ``fully_converged`` is ``False`` and
   :attr:`~orpheus.numerics.convergence.IterationRecord.first_failure`
   names the starved inner.  **A value gate asserting physics must read
   the fold, not the level.**

⛔ Until 2026-08-09 the inner fact was "one shared predicate over the
residual history" (``orpheus.sn.solver._claims_convergence``).  That
predicate is retired: it read an EMPTY history as *not converged*, so a
Krylov solve that returned on its initial guess was indistinguishable from
a truncation, and reusing it as an audit instrument produced `[M]` 44 of 90
phantom truncations.  A record separates the two —
:attr:`~orpheus.numerics.convergence.IterationRecord.iterated` is the
discriminator.

These surface on :attr:`Solution.history.converged
<orpheus.sn.solution.IterationHistory.converged>`, and
:class:`~orpheus.sn.solution.IterationHistory` is itself a **view over the
record** — every scalar it exposes is DERIVED, so there is one source of
truth and the flat surface cannot drift from the tree it summarises.

⛔ Until 2026-08-09 ``converged`` was a *field*, and this paragraph argued
its honesty from the fact that it was **required**: no default, so a
producer could not claim convergence by omission.  That was the right fix
for the wrong layer — a required field still has to be WRITTEN, by hand,
at every producer, and #342 was five such writes with one of them a literal
``True``.  Deriving it removes the question of who writes it: there is no
argument to pass, so there is nothing to get wrong.

.. tip::

   Read :attr:`Solution.history.record
   <orpheus.sn.solution.IterationHistory.record>` for anything the flat
   readings drop — which is most of it.  ``record.report()`` prints the
   whole tree, level by level, with each criterion's last value, the
   tolerance it was judged against, the observed rate, and the budget that
   rate projects; it is written to be pasted into a bug report unedited.

.. warning::

   **A value gate that does not assert convergence is asserting an
   arbitrary iterate.**  This is not hypothetical.  `[M]`
   ``test_d3_pure_absorber_per_ordinate_psi_exact`` asserted a closed-form
   identity to ``rtol=1e-10`` on an all-reflective 3-D box that needs
   **1631** sweeps, against the **then-default** ``max_inner`` of **1000**
   (hardcoded; the default has been derived from the tolerance since #340
   N3 landed 2026-08-09).  It read
   the 999th iterate, never read the flag the solver had honestly set to
   ``False``, and passed for months because the truncated error happened to
   land inside the tolerance — until a *correct* quadrature change (#337)
   moved it out.  The one-line defence is to assert
   ``sol.history.fully_converged`` **before** reading any value — the
   TREE-wide predicate, because on an eigenvalue solve the flat
   ``converged`` reads ``True`` while an inner starves (see *Loudness*
   below).

   The diagnostic tell is worth memorising, because it points the wrong
   way: the error was **bit-identical at every** ``inner_tol`` **from 1e-9
   to 1e-15**, which reads as a discretization floor.  It is the opposite —
   the running residual never fell below even the loosest tolerance, so
   every run hit the same cap and returned the same bytes.
   Tolerance-insensitivity means the tolerance never *bound*; read the
   iteration count against the budget before concluding anything about the
   discretization.

   ⚠ Read it against
   :attr:`~orpheus.numerics.convergence.IterationBudget.in_iterations`, not
   against the raw knob you passed. The two coincide for source iteration,
   the power outer, CP and MoC, and they do **not** for Krylov: ``max_inner``
   there is scipy's restart-CYCLE cap, while the recorded trajectory counts
   inner Arnoldi steps, and one cycle buys ``restart`` of them (which
   :func:`~orpheus.sn.solver` sizes to the full ``n_dof``). Comparing the raw
   pair is ERR-079 (#349) — it read a healthy converged solve as
   having run out. :meth:`~orpheus.numerics.convergence.IterationRecord.report`
   already prints the honest ceiling, so prefer it to hand arithmetic.

**Loudness.**  A truncated exit emits
:class:`~orpheus.numerics.convergence.ConvergenceWarning`, naming **the level
that failed** and the budget *that level* ran out of, the tolerance *its*
binding criterion missed, and how far its last iterate was — the distance
between "one more sweep" and "diverging" wants opposite responses.  The level
matters because a solve is a tree: on an eigenvalue run it is usually the
*inner* that starved, and until #340 N6 the message quoted the entry's own
``max_outer`` instead — a knob that cannot help, since the outer's stop test
is entirely increments and the starved inner is what suppresses them.

The guard is
:attr:`~orpheus.sn.solution.IterationHistory.fully_converged` — **every**
level, not the top one — so a converged outer standing on a starved inner is
audible.  That case is the whole of the #340 headline defect, and it is not
exotic: `[M]` 2026-08-10, **20 tests in the shipped suite** sat in exactly
that state, at observed rates :math:`\rho` between 0.889 and 0.993, every one
of them silent.

.. note::

   **Ask** :attr:`~orpheus.sn.solution.IterationHistory.fully_converged`,
   **not** ``converged``, before asserting physics against a result.  The two
   differ precisely on the starved-inner solve: the flat ``converged`` reads
   ``True`` there, because the outer really did meet its own criteria — it
   just met them on increments an upstream throttle had suppressed.

⛔ Until 2026-08-10 the guard was ``converged``, the TOP level only, and the
widening was scheduled to ride an *outer residual certificate* that would
separate a truncation which corrupted the answer from one that did not.  `[M]`
that certificate was **refuted by measurement**: the benign and corrupting
populations overlap **634×** and it misses 15 of 16 corrupting cases.  The
ruling that followed was to widen the guard *unconditionally* — a truncation
the caller has not declared is worth saying out loud whether or not we can yet
say what it cost.  The 20 silent tests were adjudicated in the same change —
10 declare the truncation as their fixture and suppress this one category
in-test, 10 are audible on purpose and tracked with measured budgets in
`#352 <https://github.com/deOliveira-R/ORPHEUS/issues/352>`_.

.. note::

   **This machinery is no longer SN's** (#340 N4.7, 2026-08-11).  The emitter
   left ``sn/solver.py`` for
   :func:`~orpheus.numerics.convergence.warn_if_unconverged`, and CP, MoC and
   1-D diffusion now call it from their own public entries.  Nothing about
   SN's behaviour changed — `[M]` the emitted message is character-identical
   across all four advice arms plus the nested and balance-defect cases,
   verified against the pre-move function lifted out of git — but two facts
   about the SHAPE of the diagnostic are worth carrying:

   * The helper was already ~90 % family-agnostic. Every fact it reads off the
     failing level is a generic
     :class:`~orpheus.numerics.convergence.IterationRecord` member; only
     ``balance_defect`` was SN's, and it is now an optional keyword that the
     other three pass as ``None`` (rendering an *absent* clause, never the
     word "unavailable").
   * ⛔ Its closing advice used to name the literal string
     ``solution.history.fully_converged``.  That is a guess at the CALLER's
     local variable name — a fact no library can know — and it was outright
     wrong for the three families whose entries return a ``*Result``.  It now
     names the attribute and its type.  A per-entry spelling passed in as an
     argument was considered and **rejected**: it would re-commit the exact
     defect N6a retired, a fact asserted by the call site and free to drift
     from the object it describes.

.. _sn-exit-balance-projection:

The balance projection
----------------------

The refutation left a real question standing — *how much did this truncation
cost?* — and the answer the warning carries is the **per-group neutron-balance
defect of the returned iterate**, reported on
:attr:`~orpheus.sn.solution.IterationHistory.balance_defect`:

.. math::
   :label: sn-exit-balance-defect

   R_g \;=\; \int_V \int_{4\pi} \bigl(A\psi - q\bigr)_g \, d\Omega \, dV,
   \qquad
   \text{reported as } \;\frac{\lVert R_g \rVert}{\lVert R_g(q) \rVert}

**Why the projection and not the residual norm** — and the reason is
structural, not statistical.  Up to **99.995 %** of :math:`\lVert r \rVert` is
reflective-trace rows, and a reflective inflow-trace defect in a zero-leakage
system carries **no net current**, so a balance-based :math:`\keff` is blind
to it *by conservation*; `[M]` the transfer gain
:math:`\lvert\Delta k\rvert / \text{defect}` spans **1.16 × 10⁵**.  Integrating
over angle and volume annihilates exactly those rows, because that is the
functional :math:`\keff` itself reads.  `[M]` the overlap falls **634× →
4.64×**.

.. warning::

   **4.64× is still an overlap.  This is a diagnostic magnitude, never a
   threshold.**  Do not branch on it, and do not assert on it in a test — a
   gate that did would be the refuted certificate wearing a different name.
   It is reported so a reader who has been told their solve truncated can
   weigh how much that is likely to have cost.

   ⛔ Nor can it be sharpened with a cheap adjoint: `[M]` a spatially-flat
   0-D weight makes it **worse**, 4.64× → **128.95×**, because a signed
   projection against a wrong weight manufactures near-cancellations, i.e.
   false negatives.  The weighting channel already exists
   (:meth:`IntegratedReactionRate.evaluate` takes ``adjoint=``); what a real
   gate needs is the adjoint *solve*
   (`#350 <https://github.com/deOliveira-R/ORPHEUS/issues/350>`_).

It is computed **only on the exit that warns** — the exact complement of the
within-group certificate, which fires when the solve *claimed* convergence and
raises.  One equation, two verbs, complementary guards, so no solve pays for
both forward applies and the converged path costs what it always did.  `[M]`
one residual evaluation is ≈ 3 inner iterations, i.e. **0.72 %** of a
400-iteration truncated solve.

.. warning::

   ⛔ **The complement of a guard is not the same as coverage, and this
   pair left the exit uncovered until 2026-09-06** (#448 /
   :doc:`ERR-083 </theory/verification/error_catalog>`).  The two verbs
   above are complementary *in when they fire* — certificate on the
   converged exit, defect on the truncated one — and they are NOT
   complementary in *what they read*.  The certificate reads the within-group
   **iterate**, inside the inner solves; this projection reads the returned
   flux, but only when the solve did not converge.  So the object a caller
   receives from a **converged** eigenvalue solve was evaluated by neither,
   and the hand-built P0-only reconstruction that produced it drifted
   undetected for the whole life of the anisotropic solve.

   Both halves are now repaired.  The finalize is one step of the
   iteration's own map (:ref:`sn-finalize-one-step`), so the returned flux
   solves the equation the reported :math:`(k, \phi)` pose; and the
   projection's budget response is the property a regression gate asserts
   (``tests/sn/solve/test_eigenvalue_finalize_reconstruction.py`` →
   ``TestTheShippedDiagnostic``).  ``[M]`` 2026-09-06, 2-group A|B|A slab, ``keff_tol =
   flux_tol = 1e-12``, ``inner_tol = 1e-11``, ``max_outer`` 3 → 12: the
   defect falls by :math:`1.43\times10^{7}` at :math:`L = 0` and
   :math:`3.46\times10^{7}` at :math:`L = 1`.  Before the fix the
   :math:`L = 1` column fell by **1.0002 ×** — pinned at a floor set by the
   reconstruction rather than by the truncation, i.e. a diagnostic that
   could not move is a diagnostic that carried no information
   (``vv-principles`` #19).  The full before/after table is in the ERR-083
   entry.

   ⚠ Reading note for anyone comparing the two trees: the absolute defect
   at a given budget is **not** comparable across the fix, because the two
   finalizes construct the returned flux differently — the pre-fix one
   re-solved a source built from :math:`\phi`, the post-fix one steps the
   map from the iterate, and at a truncated exit those differ by the
   truncation itself (compounded by the ERR-052 between-solve
   renormalisation).  What is comparable, and what the diagnostic
   advertises, is the RATIO down the budget.

Two entries report ``None`` rather than a number, and the omission is silent
by design — an empty clause cannot be misread as a measurement:

* **Moment-tailed (LD) schemes**, on both fixed-source arms: the residual mint
  does not admit the trailing :math:`2^d` spatial-moment axis, so no residual
  exists to project.  This is the same un-built widening the within-group
  certificate already exempts (#310's deferred-out list), reached through one
  shared predicate rather than two copies of the test.
* **The daggered eigenvalue entry** ``solve_sn_adjoint``: the rhs
  :math:`F^\dagger\psi^*/k` would have to be assembled for the first time
  here, and N5's population is forward-only — so there is no reference against
  which a plausible number could be checked.  Deferred deliberately as
  `#353 <https://github.com/deOliveira-R/ORPHEUS/issues/353>`_ rather than
  guessed.

It is a warning rather than an exception by the ERR-053
precedent (legitimate callers harvest the residual history of a
deliberately-truncated solve), and it escalates to a hard failure with::

    python -O -m pytest -W error::orpheus.numerics.convergence.ConvergenceWarning

.. warning::

   ⛔ **The category must be DOTTED, and this page said otherwise until
   2026-08-09.**  The recipe was published as ``-W
   error::ConvergenceWarning`` here and at four code sites (one of them the
   emitted warning message itself).  That string **does not parse** —
   Python resolves an undotted ``-W`` category against ``builtins``, so
   pytest exits at startup with ``AttributeError: module 'builtins' has no
   attribute 'ConvergenceWarning'`` and collects **zero** tests.  The CI
   contract was imaginary for exactly as long as it was documented.

   It survived because the gate that appeared to prove it,
   ``test_it_is_escalatable_to_an_error``, installs the filter through
   ``warnings.simplefilter`` — a true claim about the *category* that says
   nothing about the *spelling*.  The doc, the runtime message and the test
   all agreed on a string no interpreter accepts.

   The spelling is now derived from the class as
   :data:`~orpheus.numerics.convergence.ESCALATION_FLAG`, and
   ``test_the_published_escalation_flag_actually_parses`` consumes that
   STRING through pytest's own parser.  General rule this earned: **for
   every recipe a doc publishes as a command, one gate must consume the
   string, not the API.**

**Budget sizing.**  ``max_inner`` is **derived from the tolerance**, and it
had to be: `[M]` an all-reflective box needs ~an order of magnitude more
sweeps per added dimension (d=1 **32**, d=2 **258**, d=3 **1631**), and the
cost scales as :math:`\Sigma_t \cdot n_{\rm inner} \approx` constant.  One
vacuum face collapses the d=3 figure to 208 — the expensive corner is
specifically zero-leakage, weakly-absorbing, and 3-D.  **A constant cannot
track a tolerance it does not know about**, so ``max_inner=None`` — the
default at every SN entry — resolves through
:func:`~orpheus.numerics.convergence.resolve_iteration_budget` to
:func:`~orpheus.numerics.convergence.default_iteration_budget`, which
inverts the geometric budget law at a stated served rate.  An explicit
``int`` is still honoured untouched: a caller who knows their spectral
radius, or who is deliberately starving a solve to measure its truncated
exit, is exercising the API correctly.

⛔ *Until #340 N3 landed (2026-08-09) the default WAS a hardcoded constant*
— five of them in SN plus a sixth in ``KEigenvalue`` — and `[M]` **both**
SN families were short at d=3 zero-leakage: 830 sweeps needed at
``inner_tol=1e-8`` against 200 shipped, 1441 at ``1e-12`` against 1000.
The two families also differed by **5×** where the law puts the factor at
:math:`\ln(10^{-12})/\ln(10^{-8}) = 1.5`; `[M]` the shipped ratio is now
``1308 : 1961 = 1.4992``.

Gates: ``tests/sn/solve/test_convergence_contract.py`` — each honesty claim
is a PAIR (a converging configuration and a deliberately-starved one),
because asserting ``converged is True`` on a solve that converges is
satisfied by the very hardcoded ``True`` the contract forbids; it is the
starved leg that has teeth.  `[M]` a 6-mutation battery, positive control
first, reddens as designed — including re-introducing #342 verbatim.

.. _sn-exit-gauge:

The exit gauge — a converged solve can still be one of many
------------------------------------------------------------

The convergence contract answers *"did the iteration finish?"*.  There
is a second, structurally different way for a returned field to be
unsatisfying, and no convergence certificate can see it: the **equation**
may not have a unique answer.  On a closed reflective Cartesian box under
diamond differencing :math:`A = L+C-S-N_{2n}-B` is **exactly
singular**, so
:math:`A\psi = q` has a solution *manifold* and the iteration freezes an
arbitrary member of it.  The derivation, the counting laws, the evidence
and the remedy hierarchy are at :ref:`sn-loss-kernel-gauge`; what belongs
here is the exit behaviour.

Every entry that returns a trace applies the :math:`G`-orthogonal
projection :math:`\psi \mapsto \psi - \Pi\psi`
(:eq:`sn-loss-kernel-gauge-projection`) and records the magnitude it
removed on
:attr:`~orpheus.sn.solution.IterationHistory.gauge_correction`.  The
gauge is the **sibling** of the balance projection with one sharpening
that changes what verification it owes: :eq:`sn-exit-balance-defect`
*reports*, and this one *mutates*.  A forgotten balance-defect site
loses a diagnostic; a forgotten gauge site silently returns a
non-physical answer.  Coverage is therefore gated by an enumeration
**derived from the module** rather than hand-listed
(``tests/sn/solve/test_every_entry_gauges_its_trace.py``).

Three properties make firing it at a converged exit safe, and each is
asserted rather than assumed:

* **Residual-neutral.**  :math:`A(\psi - \Pi\psi) = A\psi`, so **no
  convergence certificate can move**.  ``[M]`` on a deliberately
  truncated SI solve the balance defect reads ``0.3111434602740818``
  before and after, while the correction goes
  :math:`3.59\times10^{-2} \to 4.9\times10^{-17}`.  It is applied
  **after** the defect is measured, so the reported number describes the
  object the caller receives.
* **Bulk-invariant.**  The kernel is pure-trace (``[M]`` bulk share
  :math:`1.1\times10^{-28}`), so :math:`\keff`, the scalar flux and
  every reaction rate are untouched — ``[M]`` :math:`\keff` reproduces
  the analytic :math:`\kinf = 1.875` on every mesh, gauged or not.
* **Not a universal absorber.**  ``jacobi`` already lands on the
  canonical member, and there the gauge must remove *nothing*: ``[M]``
  :math:`\sim10^{-15}`, and the pre-gauge deviation measures
  :math:`1.0000` **out of** span.

:attr:`~orpheus.sn.solution.IterationHistory.gauge_correction` follows
the :attr:`~orpheus.sn.solution.IterationHistory.balance_defect`
discipline exactly: ``None`` means **not measured**, never *"measured
and zero"*.  A measured :math:`\sim10^{-15}` is the different — and
useful — statement that the freedom is real and the solve landed on the
canonical member anyway.

.. warning::

   :class:`~orpheus.sn.operators.loss_kernel_gauge.GaugeFreedomWarning`
   is **deliberately not** a
   :class:`~orpheus.numerics.convergence.ConvergenceWarning`, and the
   distinction is not cosmetic.  That family means *"an iterative solve
   exhausted its budget; the answer is best-effort"*.  This is the
   opposite situation: ``[M]`` the configuration where it fires hardest
   reports ``fully_converged = True`` and ``balance_defect = None``.
   The solve is fine; the **equation** is degenerate.  Reusing the
   category would also make every caller who escalates
   :data:`~orpheus.numerics.convergence.ESCALATION_FLAG` start failing
   on an unrelated condition.

   It reports an **action taken**, not a configuration property — which
   is what keeps it off the standard all-reflective :math:`\kinf`
   lattice, where it would otherwise fire on every solve.  It escalates
   with :data:`~orpheus.sn.operators.loss_kernel_gauge.GAUGE_ESCALATION_FLAG`,
   whose value is **derived from the class** rather than retyped: the
   category must be DOTTED, for the reason the balance-projection
   section records above.

   The third state is the one a caller must not collapse: an
   **UNDETERMINED** closure — one whose face-mode damping could not be
   classified — warns loudly and is **not** gauged.  ``[M]``
   ``linear_discontinuous`` at :math:`d=3` is exactly that.

.. _sn-consuming-the-frame:

Consuming the frame in SN
=========================

Spatial homogenisation and energy condensation are **discrete-frame
projections** — the Petrov-Galerkin coefficient extraction
:math:`G^{-1}M` of a flux- (or spectrum-) weighted frame. All of that
theory — rate preservation, the source-group / sink-sum matrix rules, the
metric-fold-vs-bilinear adjoint argument, fractional-overlap re-binning,
the condense/homogenize asymmetry law, and the verification gates — is
the frame page's headline **Petrov-Galerkin** consumer; see
:ref:`sn-spatial-homogenization` and :ref:`sn-energy-condensation`
(:doc:`/theory/foundations/frame`). This section keeps only the **SN-layer orchestration**:
how the SN :class:`~orpheus.sn.solution.Solution` drives that machinery
from a converged flux.

Homogenisation: the solve → homogenize → re-solve loop
------------------------------------------------------

:meth:`Solution.homogenize <orpheus.sn.solution.Solution.homogenize>`
takes a coarse mesh (:class:`~orpheus.geometry.mesh.Mesh1D` or
:class:`~orpheus.geometry.mesh.Mesh2D`) and returns a
:class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` — the coarse
geometry already carrying one freshly-homogenised effective
:class:`~orpheus.data.macro_xs.mixture.Mixture` per coarse cell. The SN
:class:`~orpheus.sn.solution.Solution` owns the converged flux, so the SN
layer is what builds the flux-weighted **test** basis the frame consumes;
the frame itself, and the rate-preservation theory that *forces* the flux
weighting (rather than a plain volume average), live in
:ref:`sn-spatial-homogenization` (:doc:`/theory/foundations/frame`). The returned
``MaterialMesh`` is re-promoted to a solvable phase space by
:meth:`SNMesh.from_material_mesh
<orpheus.sn.mesh.augmented_mesh.SNMesh.from_material_mesh>`, closing the
**solve → homogenize → re-solve** loop. The return type is
**mesh-coupled** (geometry and materials born together) — the space half
of the condense/homogenize asymmetry law
(:ref:`sn-condense-homogenize-asymmetry`, :doc:`/theory/foundations/frame`).

Condensation: per-material representative spectra
-------------------------------------------------

:meth:`Solution.condense <orpheus.sn.solution.Solution.condense>` is the
SN-layer orchestration of energy condensation. It condenses **each
material with its own representative spectrum** — the flux·volume-weighted
flux over the cells where the material appears:

.. math::
   :label: energy-condensation-representative-spectrum

   \varphi^{(m)}_g \;=\;
   \sum_{i:\,\mathrm{mat}(i)=m} V_i\,\phi_{i,g},

.. (vv-status rationale) Representational identity: the per-material
   representative spectrum used as the condense test weight — the
   flux·volume-weighted flux over the material's cells (mirrors how
   ``homogenize`` derives its flux weight). A definition consumed by
   :meth:`Mixture.condense`; the end-to-end rate preservation it feeds is
   the L1 gate, not a separate claim.
.. vv-status: energy-condensation-representative-spectrum documented

used as the test weight in :meth:`Mixture.condense
<orpheus.data.macro_xs.mixture.Mixture.condense>` — the data-layer
collapse verb, whose spectrum-weighted-collapse theory is
:ref:`sn-energy-condensation` (:doc:`/theory/foundations/frame`) — mirroring how
:meth:`Solution.homogenize` derives its flux weight from the same solved
flux. The result is a **portable** ``dict[int, Mixture]`` keyed by
material id — few-group cross sections carrying the coarse ``eg``, not
bound to any mesh (the **mesh-decoupled** half of the asymmetry law,
:ref:`sn-condense-homogenize-asymmetry`, :doc:`/theory/foundations/frame`). A material with
no flux in a fine group contributes zero weight there; the condense
frame's Moore–Penrose Gram handles any empty coarse group.
