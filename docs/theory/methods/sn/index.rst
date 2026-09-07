.. _theory-discrete-ordinates:

==========================================
Discrete Ordinates Method (S\ :sub:`N`)
==========================================

.. contents:: Contents
   :local:
   :depth: 3


.. Machine header — the ``nexus-meta`` schema for this page.  Ingestion is
.. PENDING nexus#1 Phase 2: the ``nexus-meta`` directive is NOT yet
.. registered, so the schema is rendered here as a collapsed sphinx-design
.. dropdown and machine-consumed later.

.. dropdown:: Machine header — ``nexus-meta`` schema (module · operators · conventions · invariants)
   :color: muted

   .. code-block:: yaml

      module: sn
      method: discrete-ordinates
      aliases: [SN, discrete ordinates, Sₙ, transport sweep, ordinate method]
      governing_equation: "A ψ = (1/k) F ψ  [eigenvalue] ;  A ψ = q  [fixed source]"
      operators:
        L: streaming, BULK only (Ω·∇) — the boundary law is the sibling B, NOT folded into L
        C: collision / removal (Σ_t)
        S: scattering in-scatter gain (Σ_s0ᵀ φ + anisotropic moments); a ROLE of TransferOperator since #426 step 2, yield y = 1
        B: boundary law as a first-class SIBLING operator (reflective / vacuum / white trace), every geometry
        N2n: (n,2n) emission — first-class since CS4c step 3 (not a passenger inside S), and ANISOTROPIC since #426 step 2 (2026-09-04): the SAME TransferOperator binding as S over the channel's own Legendre stack at the solve's scattering_order, yield y = 2. Until then its ℓ=0-only kernel was a MODEL imposed at the operator tier and never a property of the reaction — a defect worth −413.55 Δk·1e5 on a Be-reflected fast slab, catalogued as ERR-082
        F: fission production (χ ⊗ νΣ_f, rank-1 dyad); TWO bindings of one datum since CS4c step 4 — IsotropicFission (energy, the k-outer's) and FissionOperator (angular, the eigen-M posing's)
      composites:
        A: "L + C - S - N2n - B — the within-group loss operator; the Krylov driver applies it. Every page of this chapter states this member list (issue #425, 2026-09-07); the four-term A = L+C-S-B survives only where it is DECLARED at the site — the 1-D diffusion solver's own composition (its S IS S+N2n), the dated rows of history.rst, and one measured MMS residual table whose fixture is (n,2n)-free by construction. Canonical: eq sn-within-group-with-n2n"
        (L+C): "lower-triangular under the upwind cell ordering; (L+C)⁻¹ IS the transport sweep"
      key_types: [AngularFlux, SNMesh, HarmonicMomentFlux, SweepDependencyGraph]
      entry_points:                    # qualnames; Nexus links via implements edges
        - orpheus.sn.solver.solve_sn
        - orpheus.sn.solver.SNSolver
      conventions:
        sign: "μ>0 is +x / outward-radial outflow; inward inflow at the left / outer boundary"
        scattering: "Mixture.SigS[l][g_from, g_to]; the in-scatter source uses the TRANSPOSE  Q = SigSᵀ @ φ"
        diamond_difference: "ψᵃ = (1+β)ψ_out − β ψ_in; Morel–Montry sets β = 0 (Bailey–Morel–Chang 2010 Eq. 43, unique exact-on-linear-in-μ)"
        quadrature_norm: "GL weights sum to 2; Lebedev / level-symmetric / product sum to 4π; moments carry NO 4π prefactor (Σw normalisation)"
        layout: "angular flux (N, ng, nx, ny); scalar flux and per-cell XS (ng, nx, ny); 1-D keeps ny=1 (singleton, not squeezed)"
        group_ordering: "fast → thermal; downscatter makes SigS upper-triangular"
        starting_direction: "curvilinear half-angle seed ψ_{1/2} is first-class typed state (System B), marched directly (Issue #282 route (a)); only levels with first-ordinate raw τ ∈ (0,1) carry the block (R12a)"
      invariants:
        - "particle balance PER ORDINATE (flat-flux residual = 0) — the strong check, NOT the telescoped scalar balance"
        - "sweep ≡ matvec (one loss representation, two applications: solve vs residual)"
        - "α redistribution dome ≥ 0 (negative → NaN / overflow)"
      depends_on: [transport_methods, operator_algebra, spherical_harmonics, frame]
      verification: [L0, L1, L2]       # authored claim; cross-checked vs the Verification slice (§ below)


.. _sn-synopsis:

Synopsis
========

The discrete ordinates (S\ :sub:`N`) method solves the
:ref:`multi-group eigenvalue problem <mg-eigenvalue-problem>` in
integro-differential form by discretising the direction variable
:math:`\hat{\Omega}` into a finite ordinate set
:math:`\{(\hat{\Omega}_m, w_m)\}`, **retaining the angular flux**
:math:`\psi(\mathbf{r}, \hat{\Omega}, E)` rather than collapsing to the
:term:`scalar flux` (contrast the collision-probability integral form).  It resolves
streaming, anisotropic scattering, and interface angular current directly.
ORPHEUS supports three coordinate systems under one balance framework:
**Cartesian** (slab / 2-D, no inter-ordinate coupling), **spherical** (1-D
radial, a single :math:`\alpha`-redistribution dome coupling all ordinates in
:math:`\mu`), and **cylindrical** (1-D radial, an independent :math:`\alpha`
dome per :math:`\mu`-level).  All three share a geometry factor
:math:`\Delta A / w` that guarantees per-ordinate flat-flux consistency; the
curvilinear formulation follows :cite:`MorelMontry1984` in the
:cite:`BaileyMorelChang2010` Eqs. (42)/(43) form (the Morel–Montry
angular-closure weight — unique exact-on-linear-in-:math:`\mu`),
the general framework :cite:`LewisMiller1984`, and the angular discretisation
:cite:`CaseZweifel1967` / :cite:`Hebert2009` (§3.9.4).

The solver is posed as an **operator algebra** over six operators: streaming
:math:`L` (bulk :math:`\hat{\Omega}\cdot\nabla`), collision / removal
:math:`C`, the scattering gain :math:`S`, the :math:`(n,2n)` gain
:math:`N_{2n}`, the boundary law :math:`B` — a
first-class **sibling** operator, *not* folded into :math:`L` — and the rank-1
fission dyad :math:`F`.  They compose the within-group loss operator
:math:`A = L + C - S - N_{2n} - B` (:eq:`sn-within-group-with-n2n`), so the
eigenvalue problem is
:math:`A\,\psi = \tfrac{1}{k}\,F\,\psi` (fixed source: :math:`A\,\psi = q`).

.. note::

   **Six operators — where the** :math:`(n,2n)` **term is spelled, and
   the three places it is not.**  Since CS4c step 3 (2026-08-30) the
   :math:`(n,2n)` emission is a **first-class** operator
   :math:`N_{2n}` rather than an unnamed passenger inside :math:`S`, so
   the composite the S\ :sub:`N` builder actually composes is
   :math:`A = L + C - S - N_{2n} - B`
   (:eq:`sn-within-group-with-n2n`).  Since #426 step 2 (2026-09-04)
   :math:`S` and :math:`N_{2n}` are also two **instances of one
   binding** — same faces, same arms, same transposes, differing in the
   yield :math:`y` inside :math:`\Lambda_c` alone — so anything this
   chapter derives for :math:`S` holds for :math:`N_{2n}` with
   :math:`y = 2` and the channel's own stack.

   Until issue #425 (2026-09-07) most pages of this chapter *still spelled*
   the four-term :math:`A = L + C - S - B`, on the argument that
   :math:`\Sigma_{2n} \equiv 0` on every fixture the chapter derives
   against.  That simplification is **retired**: a page that states the
   general within-group algebra now states the shipped member list, and
   the four-term form survives only where it is **exact and declared at
   the site** —

   * the **1-D diffusion solver's** :math:`A`, which genuinely *is*
     :math:`L + C - S - B` because it sums the two isotropic energy
     leaves into one :math:`S` at its own composition site; that
     :math:`S` **is** :math:`S + N_{2n}`, so the spelling is a statement
     about the *composition*, not about the member list
     (:ref:`sn-n2n-adjoint`, :ref:`n2n-reactions`);
   * the **dated changelog rows** of :doc:`history`, which keep the
     spelling that was current on their date;
   * one **measured MMS residual table** in
     :doc:`curvilinear_numerics`, whose fixture is built with
     :math:`\Sigma_{2n} \equiv 0` by construction
     (:mod:`orpheus.derivations.continuous.mms.sn`), so the four-term
     operator is exactly the one that produced the tabulated numbers.

   The pass changed **no measured value** — every edit is an algebra
   spelling — and it could not have: ``[M]`` 2026-09-07, all **12**
   ``xs_library`` mixtures (regions A–D :math:`\times` 1/2/4 groups)
   carry a :math:`\Sigma_{2n}` with **zero** non-zeros, and every MMS
   mixture is minted ``Sig2 = csr_matrix(zeros((ng, ng)))``
   (:mod:`orpheus.derivations.continuous.mms.sn`), so the chapter's
   convergence ladders, :math:`\kinf` anchors and SI-rate anchors are
   :math:`(n,2n)`-free by construction.  What changed is that the algebra
   the pages *state* is the algebra the tree *composes*.

   Where the channel **is** live the fixture is *injected* deliberately,
   and there the pages already spelled :math:`N_{2n}` out: the shipped
   anisotropy ladder on a Be-reflected fast slab
   (:ref:`the measured P0-truncation ladder <sn-n2n-p0-truncation-measured>`), the :math:`k`-estimator
   convention leg, and the eigenvalue-finalize reconstruction.  The other
   places the distinction bites — the adjoint chain, the DSA posing, any
   multiplying medium with an :math:`(n,2n)` channel — say so explicitly
   (:ref:`sn-n2n-adjoint`).

The sub-composite :math:`(L+C)` is lower-triangular under the upwind cell
ordering, which is exactly why :math:`(L+C)^{-1}` **is** the transport :term:`sweep`
(:doc:`/theory/methods/sn/loss_representation`).  :class:`SNSolver` satisfies
the :class:`~numerics.eigenvalue.EigenvalueSolver` protocol and
:func:`solve_sn` returns a :class:`~orpheus.sn.solution.Solution`.  Because the protocol places the
scattering source *inside* ``solve_fixed_source``, the inner source iteration
(in-scatter + anisotropic convergence) stays encapsulated in the SN sweep,
while the outer :func:`~numerics.eigenvalue.power_iteration` loop is the one
shared by CP, MoC, diffusion, and the homogeneous solver (see
:doc:`/api/numerics` for the protocol contract).

The spatial closure is **diamond difference** with the Morel–Montry weight
(:math:`\beta = 0`); the :term:`per-ordinate <ordinate>` discrete transport is the **sweep**,
which is byte-identical to the loss-operator **matvec** — one loss
representation, two applications (``solve`` vs residual).  Both the 1-D scan
(:meth:`~orpheus.sn.loss_representation.CumprodScan.sweep`) and the 2-D
wavefront sweep
(:class:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph`,
per-octant batched dispatch over a mesh-time-precomputed DAG) are **bare**:
the reflective coupling :math:`\psi.\text{inflow} = B\,\psi.\text{outflow}`
rides as a sibling :math:`-B` source term rather than a re-applied boundary
condition (:ref:`bare-sweep-extraction`, and the canonical algebra
:ref:`bc-extraction` in :doc:`/theory/foundations/boundary_conditions`).  2-D Cartesian eigenvalue
problems solve through **both** inner drivers —
:class:`~orpheus.numerics.iteration.SourceIteration` (the geometry-agnostic
default) and :class:`~orpheus.numerics.iteration.KrylovAcceleration` —
verified SI ≡ Krylov ≡ closed-form :math:`k_\infty`
(:ref:`bc-extraction-2d-si-krylov-twin`).  Interior cell-face fluxes are typed
as an interior 1-cochain :math:`C^1_{\rm int}` carried in the rolling front
(:ref:`wavefront-flux-cochain`).

Curvilinear redistribution — the geometric :math:`\alpha`-dome, distinct from
Legendre :math:`P_1^+` scattering anisotropy — and its half-angle
**starting-direction seed** are first-class typed state.  The Issue #282 route
(a) design marches :math:`\psi_{1/2}` directly from the true :math:`q_{1/2}`
source through System B's named resolvent
:meth:`RadialCharacteristicOperator.solve <orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator.solve>`
(a single-pass exact inverse on the *full* Legendre fold), only on levels
whose first-ordinate raw :math:`\tau \in (0,1)` (**R12a**); see
:ref:`sn-direct-seed-solve`.  The "#229 floor" resolution
— three distinct curvilinear errors separated by a
volume-weighted-:math:`L_2`-vs-:math:`L_\infty` norm difference — is
:ref:`sn-curvilinear-aniso-norm-reconciliation`.  The storage / index /
scattering / closure **conventions** and the load-bearing **invariants** are
captured structurally in the machine header above and cross-linked below;
verification is L0–L2 plus semi-analytical (Sood, Case singular-eigenfunction)
benchmarks; the traps that hide solver bugs behind green tests are collected in
:ref:`sn-gotchas`.

.. admonition:: Conventions

   - Scattering matrix: :ref:`scattering-matrix-convention` — ``SigS[g_from, g_to]``, source uses transpose
   - **Storage layout**: :ref:`theory-sn-index-convention` — ``(N, ng, nx, ny)`` for ψ, ``(ng, nx, ny)`` for φ / σ
   - Multi-group balance: :eq:`mg-balance` in :ref:`theory-homogeneous`
   - Cross sections: :ref:`theory-cross-section-data`
   - Verification: :ref:`synthetic-xs-library` — regions A/B/C/D
   - Eigenvalue: :ref:`power-iteration-algorithm` shared with all deterministic solvers


Architecture
============

.. note:: **Implementation map — automation pending.**  The auto-generated
   Nexus filtered flow-graph figure (root symbol + traversal depth →
   graphviz) that will head this section is blocked on the nexus#20
   flow-graph directive; until it ships, the architecture below is
   **hand-authored**.  See :doc:`/api/numerics` for the live
   operator-protocol surface and :doc:`/theory/foundations/operator_algebra` for the algebra.

Two-Layer Mesh Pattern
----------------------

The S\ :sub:`N` solver follows the same two-layer pattern as the CP
solver.  This pattern (base :class:`~geometry.mesh.Mesh1D` + augmented
mesh) is shared with :ref:`theory-collision-probability` and
:ref:`theory-method-of-characteristics`.

1. **Base geometry** --- :class:`~geometry.mesh.Mesh1D` or
   :class:`~geometry.mesh.Mesh2D` stores cell edges, material IDs,
   coordinate system, and **boundary condition declarations**.
   Each face carries an optional :class:`~geometry.mesh.BC` field
   (``bc_left``/``bc_right`` for 1-D;
   ``bc_xmin``/``bc_xmax``/``bc_ymin``/``bc_ymax`` for 2-D).
   When ``None`` (the default), the solver applies its own default
   --- for the SN solver, that default is reflective.
   See :ref:`boundary-conditions` for details.

2. **Augmented geometry** --- :class:`SNMesh` pairs the spatial mesh
   with an angular :term:`quadrature`, precomputing the coordinate-specific
   streaming stencil.  Its **primary representation is the per-axis
   tuple** :attr:`SNMesh.axes <orpheus.sn.mesh.augmented_mesh.SNMesh.axes>` (the SN phase space factors as a tensor
   product of per-axis 1-D meshes): a legacy ``Mesh1D`` / ``Mesh2D`` is
   converted to axes **once** at the inbound boundary, and
   :meth:`SNMesh.from_axes` stores the caller's tuple verbatim. After
   C5 (:ref:`sn-axis-primary-c5`) the ``mesh`` attribute is *inbound
   provenance only* — ``None`` for an axis-native :math:`d \ge 3` mesh,
   which carries no legacy mesh at all.  (A literal, not an ``:attr:``
   role: the base ``MaterialMesh`` sets it on the instance, so there is
   no autodoc target to link.)  It also **resolves boundary
   conditions**: each ``BC`` tag
   on the mesh is looked up in :attr:`SNMesh.BOUNDARY_OPERATOR_REGISTRY` and converted
   to a validated kind string (``"vacuum"`` or ``"reflective"``)
   stored in the face-name-keyed :attr:`SNMesh.bc` dict
   (``sn_mesh.bc["xmin"]``, ``sn_mesh.bc["xmax"]``, ... — the dict
   keys are the mesh's true boundary faces; see
   :ref:`bc-face-name-carve`).
   The sweep reads these resolved strings directly --- it never
   inspects the raw :class:`~geometry.mesh.BC` objects.  Precomputed
   stencil contents per coordinate system:

   - **Cartesian**: one per-axis array ``streaming(a)[n,i] =
     2|mu_a|/da[i]`` for every axis ``a < ndim`` (built over
     ``range(ndim)`` from ``quad.axis_cosines(a)`` since C3.6 ---
     no hand-listed x/y pair, no phantom axis on a slab) --- the
     diamond-difference denominator terms, precomputed to avoid
     per-cell division in the sweep hot loop.
   - **Spherical**: ``face_areas`` (:math:`4\pi r^2`) and ``delta_A``.
   - **Cylindrical**: ``face_areas`` (:math:`2\pi r`) and ``delta_A``.

   The **angular** factor is not a streaming-factory output either
   (2026-08-26): the :math:`\alpha`-dome and the starting direction
   :math:`\mu_{\rm start}` are produced once, per :math:`\mu`-level, by
   :func:`~orpheus.sn.angular.redistribution.angular_redistribution` and
   carried on
   :class:`~orpheus.sn.angular.redistribution.AngularRedistribution`.
   ⛔ ``redist_dAw`` / ``redist_dAw_per_level`` retired with them: that
   array was the *fused product* :math:`\Delta A_i \otimes 1/w_n` of a
   geometric with a quadrature factor, and each of its two consumers
   wanted a different one of the two — so both now form it from
   ``delta_A`` and the measure's weights.

   The Morel--Montry angular weight :math:`\tau` is **not** a factory
   output: it is owned by the angular closure
   (:attr:`~orpheus.sn.angular.closure.AngularClosureBase.tau_per_ordinate`),
   since the geometry-side producer was retired in Issue #236 Phase 2
   Step C (see :ref:`sn-tau-c-on-cellvisit-live`).  The geometry
   factories carry **geometry only** — face areas, the
   :math:`\alpha`-dome, the redistribution factor :math:`\Delta A/w`,
   and the level :term:`starting-direction <starting direction>` edge :math:`\mu_{1/2}`.

3. **Solver** --- :func:`solve_sn` creates an ``SNMesh``, builds the
   ``SNSolver``, and runs power iteration. At :math:`d \le 2` the input
   is a ``Mesh1D`` / ``Mesh2D``; at :math:`d = 3` the input is the
   **axes tuple** itself (the only 3-D entry — there is no ``Mesh3D``;
   see :ref:`sn-c5-3d-admission`).

.. code-block:: text

   Mesh1D / Mesh2D (d<=2)   OR   axes tuple (d=3, axis-native)
       |                              |
       |  axes_from_legacy_mesh       |  (stored verbatim)
       +------------+-----------------+
                    v
   SNMesh.axes  (PRIMARY)  -->  SNMesh (stencil + quadrature
                                + alpha coefficients + resolved BCs)
                    |
                    v
   solve_sn() --> Solution

.. _sn-p49b-operator-poses-with-closures:

P4.9b — the hub is a save state, and the operator is posed on it
-----------------------------------------------------------------

:ref:`sn-p49a-closure-owns-the-march` gave the angular march back to its
owner and left the SN *walk* applying it.  That is one level short of the
destination: the walk was still reaching into
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` at apply time for both
method objects, so an operator you had already built could still change
its mind about *how* it discretises if somebody rebound a mesh attribute
underneath it.  P4.9b (2026-08-28) closes that: the streaming operator
is **posed** with the two closures it will use, and from then on it
computes from its own fields.

The subsection is the sequel to P4.9a and states the four things a
reader needs in order not to undo it: what the operator now takes, why
the mesh nevertheless *keeps* the generator, why the constructor carries
no consistency guards, and where the performance weld went.

The three fields, and the absence of a default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~orpheus.sn.operators.streaming.StreamingOperator` is a
dataclass with three **required** fields and no defaults:

.. code-block:: python

   @dataclass
   class StreamingOperator(LinearOperator["FullField"]):
       sn_mesh: "SNMesh"                                # the geometric substrate
       spatial_closure: "DiscretizationSchemeBase"      # required, no default
       angular_closure: "AngularClosureBase"            # required, no default

The absence of a default is the design's first claim, and it is a claim
about *authorship*, not about ergonomics — **the discretization is an
active choice**, so an operator that guessed one would be answering a
question nobody asked.  ``StreamingOperator(sn_mesh)`` is a
``TypeError`` (`[M]` *"missing 2 required positional arguments:
'spatial_closure' and 'angular_closure'"*), which is the loud,
collection-time failure that the illegal-states-unrepresentable pattern
buys here: there is no such thing as an under-specified streaming
operator, so no code path has to check for one.

⚠ Note which half of that pattern is being claimed.  "Make illegal
states unrepresentable" is two-sided — *every admitted value is legal*
**and** *every legal value is admitted* — and only the first half is
asserted above.  The second half is what the next subsection is about:
the constructor admits pairs the hub would never produce, deliberately.

The production surface is the classmethod
:meth:`~orpheus.sn.operators.streaming.StreamingOperator.pose`, whose
whole body is one line:

.. code-block:: python

   @classmethod
   def pose(cls, sn_mesh):
       return cls(sn_mesh, sn_mesh.scheme, sn_mesh.angular_closure)

``pose`` is the migration lever, not the destination.  Every transport
method's streaming operator needs a domain, a codomain, a way to
discretise space and (for the curvilinear ones) a way to discretise
angle; **none of them needs an**
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh`.  The recorded end state
is therefore the cross-method constructor ``(domain, codomain,
spatial-discretization[, angular-discretization])`` with the mesh
argument gone, and ``pose`` retires with the migration that reaches it.
Until then the mesh field is a **declared transitional weld**: the
representation and the walk still read geometry, boundary conditions and
connection coefficients off it, and the operator's ``domain`` /
``codomain`` are still derived from
``sn_mesh.full_field_space``.

.. note:: **Why the mesh field was kept rather than replaced by the
   literal four-argument shape now.**

   Passing ``(domain, codomain)`` *alongside* a mesh would make a
   mismatch between them **spellable** — a Pattern-4 inversion, since the
   spaces are today derived from that very mesh.  One object that
   answers both questions cannot disagree with itself.  The four-argument
   shape becomes reachable when the representation stops needing the
   mesh, which is a different campaign's work.

Why the hub keeps the generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The obvious next move — *the operator has the scheme now, so take it off
the mesh* — is **wrong**, and the reason is worth stating plainly
because the charter originally said to do it.

:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` is a **misnomer**.  It is
not only a mesh: it is the solve's **save state and data hub**, the
object you would dump to disk to reproduce a run.  It keeps the
discretization scheme *not because it needs one to be a mesh*, but
because a scheme is **shared machinery**, and two independent consumers
must be given the same one:

#. **Cross-consumer consistency.**  Diffusion-synthetic acceleration
   must discretise space the way the S\ :sub:`N` sweep does, or the
   correction it computes does not correspond to the error it is
   correcting.  With one generator on the hub, DSA and the transport
   solve read the *same* object; with the generator distributed to each
   operator, keeping them equal becomes a runtime obligation somebody
   has to remember.
#. **Space induction.**  The scheme co-determines the mesh's **spaces**:
   whether the spatial representation is nodal or modal, and hence the
   shape of the spatial axis itself.  A multi-moment scheme such as
   Linear-Discontinuous gives the angular trial space a moment tail;
   ``full_field_space`` — the operator's own domain and codomain — is
   built through that.  The generator is consumed at mesh construction
   *by the space*, which is upstream of any operator posed on it.

So the ruling is a **partition**, not a move.  Method-flavoured
quantities — per-cell kernels, the march, the minted scan constants,
the strategy-selection predicates — arrive through the operator.
Space-and-layout facts stay on the hub, because a layout is a property
of the space the hub induces, not of an operator posed on it.  The
partition is not prose: it is an executable allowlist.

.. list-table:: The hub route after P4.9b — the read-set gate's allowlist
   :header-rows: 1
   :widths: 34 18 48

   * - what the walk reads off the hub's ``scheme``
     - allowed?
     - why
   * - ``spatial_basis_per_axis``
     - ✅ yes
     - the per-axis basis count — a **layout** fact; it is what the
       trace's face-moment layout and the field shapes are built from
   * - ``is_multi_moment``
     - ✅ yes
     - the same fact as a predicate — does the spatial axis carry a
       moment tail
   * - ``residual_kernel_batch`` / ``source_emission`` /
       ``cell_average`` / ``cell_kernel_batch`` / the transposes
     - ⛔ no
     - per-cell **kernels** — method-flavoured, must arrive on
       ``op.spatial_closure``
   * - ``is_affine_scannable`` / ``transverse_coupling_is_facewise`` /
       ``supports_curvilinear``
     - ⛔ no
     - **strategy selection** consumes the handed closure (see
       ``supports`` in :doc:`loss_representation`)
   * - anything at all on the hub's bound ``angular_closure``
     - ⛔ no
     - the allowlist for the angular half is **empty**

``test_hub_route_reads_only_space_facts``
(``tests/sn/operators/test_operator_feeds_the_walk.py``) enforces exactly
that table.  Its instrument is worth knowing, because it is reusable: it
poses the operator (which captures the *real* objects), then rebinds the
hub's two slots to **delegating recording subclasses**.  A read that
arrives through ``mesh.scheme.X`` hits the recorder and is logged; a read
that arrives through the operator's own field hits the real object and
is invisible.  The assertion is then a set difference against the
allowlist.  Two details keep it honest: the recorders are *subclasses*
rather than proxies (the walk type-discriminates on the closure family —
see below), and the gate first asserts that the recorder records at all,
so a silently-inert instrument cannot read as a clean pass.

Why the constructor carries no guards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The constructor does **not** check that its closures came from the hub it
was handed.  That was ruled deliberately, after the position was attacked
four ways; the attacks are recorded here because "add a guard" is the
first thing a reader will want to do.

.. list-table:: The four attacks on the no-guard position
   :header-rows: 1
   :widths: 6 30 64

   * - #
     - the attack
     - outcome
   * - 1
     - the ``pose`` path
     - **Unbeatable.**  ``pose`` reads the hub's own two objects, so on
       the production path a disagreement is not *checked*, it is
       **unspellable**.  Every production construction goes through it.
   * - 2
     - raw constructor, foreign *spatial* scheme (an LD operator over a
       DD-built hub)
     - **Spellable but loud.**  The operator's ``domain`` derives from
       the hub's spaces, so it carries no moment tail while the LD
       kernels index one — a shape error at the first apply.
   * - 3
     - raw constructor, wrong-**family** angular closure (the Cartesian
       identity closure on curvilinear factors)
     - **Constructs, then raises at the first sweep.**  `[M]`
       2026-08-28: typed on the sphere (*"… requires the Morel–Montry
       closure"*), untyped ``IndexError`` on the cylinder, and
       bit-identically inert on the slab — where the identity closure
       IS the default.  The walk's own family dispatch refuses it.
   * - 4
     - cross-hub smuggling — mesh A's closure into mesh B's operator at
       equal ordinate count
     - **The one genuinely silent arm.**  Wrong pairing, plausible-
       looking answers.  It requires two hubs and a deliberate crossing,
       and guarding it would require the closure to remember its mesh —
       re-welding exactly what the closure's un-binding achieved.

.. warning:: **A refuted sentence, kept so it is not re-derived.**

   The design round justified arm 3 as *"silent, plausible-wrong* ``k``\
   *"*.  That was **reasoned, not run**, and it is false: measurement
   (row 3 above) shows the doctored state raises at the first sweep on
   every curvilinear geometry.  The no-guard ruling survives on its other
   grounds — arm 1's unspellability and the seam's legitimate use — but
   the refuted sentence must not be transcribed into the constructor
   docstring, and the constructor's own docstring records the measured
   behaviour instead.

What the raw constructor *is*, then, is a **declared expert seam**.  A
diagnostic probe that wants to switch angular redistribution off and
measure the difference should build such an operator honestly rather
than monkeypatching production; the constructor is how.
``test_the_raw_ctor_is_a_declared_expert_seam``
(``tests/sn/operators/test_streaming_operator.py``) freezes that as a
one-positive-leg test with **no** negative leg, and says so in its own
docstring — the contract being frozen is *that no validation exists*, so
there is nothing to assert raising.

⭐ The consequence for review: with no constructor guard, the pose-identity
gate ``test_pose_reads_the_hub_objects_by_identity`` **is** the production
safety argument, not a nicety beside it.  Its legs are ``is``-identity on
both slots, plus a non-vacuity leg (a second hub's objects differ, and a
pose over it lands on *that* hub's objects) — because an identity
assertion against an accidental singleton would pass for the wrong
reason.  The mutation it exists to catch is a ``pose`` that **mints**
fresh objects instead of reading the hub: that version type-checks,
solves correctly, and silently breaks the one-instance invariant the
DSA-consistency ruling rests on.

`[M]` 2026-08-28, this session — ``pose`` monkeypatched in process to
mint its own objects, mirroring the hub's construction arm for arm so
that only *identity* changes and never the arithmetic (an **in-class**
mutation; the first attempt crashed on the Cartesian arm and reported
four extra reds that were the harness's, not the invariant's — ``vv``
anti-patterns #17 and #18 in one probe).  Over the 65 tests in the three
modules that name the invariant, the mutation reddens exactly **5**
rows, and *every one of them is a structural leg*:

* ``test_pose_reads_the_hub_objects_by_identity`` — the pose-identity
  gate;
* ``test_every_per_cell_consumer_reaches_the_mesh_closure`` — the
  one-instance gate;
* all three **closure** rows of the route gate below, on their
  activation legs.

The other **60 tests pass** — every value assertion, every regression
snapshot in those modules, every ``array_equal`` pin.  A broken invariant
that no number can see is the definition of a claim that needs a
structural gate, and this measurement is what makes "no constructor
guard" a defensible position rather than an omission.

The keystone: a hub mutation after posing must be inert
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The phase's actual claim is a **route** claim — *the walk's
method-flavoured needs come from the operator, not from the mesh* — and a
route claim cannot be gated by asserting an output value (``vv``
anti-pattern #26: a function that does the work and throws it away is
indistinguishable, in its return value, from one that skipped it).  It
needs an instrument that observes the route.

The instrument is a **swap**: pose the operator, then rebind the hub's
slots to deliberately mutant objects, then drive the solve through the
**already-posed** operator.  Before the carve the walk re-read the hub at
apply time, so the answer moved; after it, the operator holds the
pre-swap objects and the answer must be bit-identical.

.. list-table:: `[M]` 2026-08-28 at ``10314dfa`` (pre-carve) — the gate's own red reading
   :header-rows: 1
   :widths: 30 26 22 22

   * - configuration
     - hub slot swapped
     - surfaces mutated (× 1.05)
     - relative deviation
   * - slab, ``gauss_legendre(8)``
     - ``mesh.scheme``
     - ``source_emission`` + ``residual_kernel_batch``
     - 5.000e-02
   * - cylinder, ``folded_product(4, 6)``
     - the bound angular closure
     - ``advance_psi_half`` + ``c_out_per_ordinate``
     - 4.596e-02
   * - cylinder, ``folded_product(4, 8)``
     - the bound angular closure
     - same
     - 5.313e-02
   * - sphere, ``gauss_legendre(8)``
     - the bound angular closure
     - same
     - 1.196e-01

Post-carve every row reads ``np.array_equal``.  The four rows are **not
redundant**, and the reason is a structural asymmetry worth knowing
before designing any gate on this path: the two halves of the carve have
**disjoint activating configurations**.  Counting the two per-cell
entries over whole ``solve_sn`` k-eigenvalue runs (`[M]` 2026-08-28,
this session; 2-group fissile, 8 cells, reflective / vacuum):

.. list-table:: Per-cell dispatch counts over one k-eigenvalue solve
   :header-rows: 1
   :widths: 34 33 33

   * - configuration
     - ``DiamondDifference`` ``residual_kernel_batch`` (the **scheme**)
     - ``MorelMontryAngularSweep`` ``cell_contribution`` (the **closure**)
   * - slab, ``gauss_legendre(8)``
     - 656
     - **0**
   * - sphere, ``gauss_legendre(8)``
     - **0**
     - 5 552
   * - cylinder, ``folded_product(4, 6)``
     - **0**
     - 24 928

The zeros are exact, not small.  A curvilinear fixture therefore cannot
witness the scheme re-plumb and a slab fixture cannot witness the closure
re-plumb — so every step-2 gate must carry both, and a "representative"
single-geometry row would be a gate that structurally cannot fail for
half of what it claims.

⚠ Three ways this gate goes silently green for the wrong reason, all
measured while it was being built, all worth carrying forward to the next
route gate anyone writes:

* **Mutating** ``cell_contribution`` **alone is insufficient, not
  blind.**  The ``.solve`` route consumes ``advance_psi_half`` plus the
  closure's minted scan constants, while ``cell_contribution`` is the
  *matvec's* per-cell arm — a legacy of P4.9a's split of the two forms
  (:ref:`sn-p49a-two-forms`).  A gate that mutates one surface certifies
  one route and reads ``array_equal = True`` on all three curvilinear
  rows.
* **The obvious driver re-poses.**  The shared helper ``sweep_once``
  constructs its own operator internally, so it re-poses *after* the
  swap — post-carve it would still read the mutant and the gate would
  stay red for a reason unrelated to the carve.  The gate builds and
  drives ``(L + C).solve`` itself.
* **The memos mask the swap.**  Without dropping the mesh-attribute
  memos, the cached table survives the swap and the gate passes *because
  of the cache*.  Hence the gate's activation leg, which proves the
  mutant object is consulted at all on the pre-swap route — otherwise
  the assertion is ``X == X``.

⭐ And one that is a property of the **subject**, not of the harness: the
mutants must be **subclasses**.  The walk still type-discriminates on the
angular closure family, so a transparent recording proxy or a duck-typed
stand-in is *refused* with a typed error rather than silently accepted.
That refusal is a live argument for dissolving the ``isinstance``, but it
is not this phase's item; until then, "wrap it in a proxy" is not an
available instrument anywhere on this path.

Algebra eager, performance lazy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The last ruling is the one with the widest reach beyond S\ :sub:`N`, and
it is quoted rather than paraphrased because it is a standing principle:

   *Correctness concerns separated from performance concerns.  So if a
   particular solution method goes through a scan in a certain way or
   welds terms for performance reasons, then this should be done as
   close to the solution strategy as possible.  The algebra should be
   unwelded and highly expressible for as long as possible, and
   performance optimizations should be lazily resolved.*

Applied here, it draws a line straight through the curvilinear scan.
The **operator** owns and exposes the *algebra*: the two closures and the
per-ordinate constants they mint (P4.9a's Form-B pair,
:eq:`sn-p49a-march-forms`).  The **fused, scan-normal table** — the
rearranged coefficients welded with the :math:`(\Delta A/w)\,c_{\rm in}`
spatial-⊗-angular product — is not algebra; it is a *performance weld*,
and it belongs to whoever is doing the solving.  So it is the strategy's
artifact, resolved **lazily**, on first need, from the operator's
objects.

Where the memo lives is then not a taste question but a **lifetime**
question, and the wrong answer is expensive.  A ``cached_property`` on
the operator looks like the tidy home and is not, because the operator's
lifetime is far shorter than a solve.

.. list-table:: `[M]` 2026-08-28, this session — how many operators one solve builds
   :header-rows: 1
   :widths: 42 28 30

   * - configuration (2-group fissile, ``solve_sn`` k-eigenvalue)
     - ``StreamingOperator`` built
     - Stratum-1 table built
   * - slab, ``gauss_legendre(8)``, 8 cells
     - 42
     - **1**
   * - sphere, ``gauss_legendre(8)``, 8 cells
     - 38
     - **1**
   * - cylinder, ``folded_product(4, 6)``, 8 cells
     - 40
     - **1**
   * - slab, ``gauss_legendre(16)``, 200 cells
     - 43
     - **1**

Reproduce it by counting both constructors around a solve:

.. code-block:: python

   import numpy as np
   from scipy.sparse import csr_matrix
   from orpheus.data.macro_xs.mixture import Mixture
   from orpheus.geometry import BC, Mesh1D
   from orpheus.numerics.quadrature import Quadrature
   from orpheus.sn.operators.streaming import StreamingOperator
   from orpheus.sn.solver import solve_sn
   from orpheus.sn.sweep.cache import StreamingCoefficientCache

   S = csr_matrix(np.array([[0.30, 0.10], [0.0, 0.40]]))
   mats = {0: Mixture(SigT=np.array([0.60, 0.80]), SigC=np.array([0.10, 0.20]),
                      SigL=np.zeros(2), SigF=np.array([0.05, 0.10]),
                      SigP=np.array([0.12, 0.25]), SigS=[S],
                      Sig2=[csr_matrix(np.zeros((2, 2)))], chi=np.array([1.0, 0.0]))}
   nx = 200
   mesh = Mesh1D(edges=np.linspace(0.0, 10.0, nx + 1),
                 mat_ids=np.zeros(nx, dtype=int),
                 bc_left=BC("reflective"), bc_right=BC("vacuum"))
   quad = Quadrature.gauss_legendre(16)

   n = {"op": 0, "tab": 0}
   real_init = StreamingOperator.__init__
   real_tab = StreamingCoefficientCache.from_mesh_and_quad.__func__
   StreamingOperator.__init__ = (
       lambda s, *a, **k: (n.__setitem__("op", n["op"] + 1),
                           real_init(s, *a, **k))[1])
   StreamingCoefficientCache.from_mesh_and_quad = classmethod(
       lambda cls, m: (n.__setitem__("tab", n["tab"] + 1),
                       real_tab(cls, m))[1])
   try:
       solve_sn(mats, mesh, quad)
   finally:
       StreamingOperator.__init__ = real_init
       StreamingCoefficientCache.from_mesh_and_quad = classmethod(real_tab)

   assert n["op"] > 10           # dozens of operators per solve
   assert n["tab"] == 1          # ONE table — the interned lazy resolve

`[M]` on the last row of the table above, one Stratum-1 build costs
**8.84 ms** (minimum of five) against a **546.6 ms** whole solve.  A
per-operator memo would therefore add the other 42 builds — **371 ms**,
a **+68 %** wall-clock increase — to a solve whose answer would not
change by one bit.  (The phase's own pre-carve reading, on a different
fixture that converged in fewer outer iterations, was 6–10 operators per
solve and up to 24.65 %; the *operator count scales with the outer
iteration count*, which is why the stable claim is the ratio's sign and
the build cost, not the percentage.)

The ruled home is therefore **the strategy layer**, and specifically
:func:`~orpheus.sn.loss_representation.geometry_cache_for`: a
module-level ``WeakKeyDictionary`` keyed on the hub, **validated against
the handed angular closure's identity** so that a doctored pair gets its
own build rather than silently inheriting a table built for a different
closure.  Three properties follow, and each was a criterion:

#. **The operator stays pure algebra** — nothing is parked on it, so its
   equality and its lifetime stay simple.
#. **The hub stops accumulating computation** — the ``_geom_cache``
   mesh-attribute memo is retired, so a save state is not also a cache.
   (Two sibling memos, ``_coll_cache`` and ``_pole_mirror_cache``,
   deliberately remain: the :math:`\sigma` stratum's re-posing is the
   consumer-side campaign's territory, and moving one of three memos for
   symmetry alone would be churn.)
#. **The mechanism dies with the layer it serves.**  The strategy layer
   is retirement-bound: when the lazy solution strategy it exists to
   serve is built, the interning goes with it, rather than being stranded
   on an operator or a hub that outlive it.  That was the explicit
   selection criterion — *pick the thing that would be best at surviving
   the change to a lazy solution strategy*.

The gate is a **count**, never a wall clock:
``test_geometry_cache_builds_exactly_once_per_mesh``
(``tests/sn/sweep/core/test_cache.py``) pins one build across a whole
solve **and** across two independently posed operators over one hub.  A
timing assertion would be a flaky proxy for the same question; the count
is exact, and it is the only instrument that can see a memo-scoping
regression — which is otherwise a silent tens-of-percent, not a wrong
answer.

What moved, concretely
~~~~~~~~~~~~~~~~~~~~~~~

* **The operator's contract.**  Three required fields; ``pose`` as the
  intermediate posing surface;
  :func:`~orpheus.sn.coupled_system.build_streaming_collision` — the one
  production :math:`L + C` spelling — routes through it.  `[M]` 135 test
  construction sites migrated to ``pose``.  ⭐ The solve entries are
  **unchanged**: ``scheme=`` keeps flowing into the hub, because the hub
  ctor is the active-choice site and its
  :class:`~orpheus.transport.spatial.diamond.DiamondDifference` default
  survives there.  The operator is what never defaults.
* **The representation takes the pair.**
  :func:`~orpheus.sn.loss_representation.default_for` and every
  strategy's base now carry ``mesh`` plus the two closures, and the
  selection predicate is ``supports(mesh, spatial_closure)`` — selection
  consumes the **handed** closure, never ``mesh.scheme``.  The
  representations gained their own ``pose`` classmethod for the test-side
  construction that used to pass a bare mesh.
* **The walk consumes what it was handed.**  The per-cell dispatch reads
  ``self.spatial_closure`` / ``self.angular_closure``; the two remaining
  reach-throughs are handed their objects at posing time.
* **The table is interned in the strategy layer** (above), the eager
  build in the solver's constructor is gone, and the solver's Stratum-1
  slot *is* the interned instance.
* **The Stratum-1 admission contract now raises.**  The table builder
  refused a chain-less (2-D Cartesian) mesh with a bare ``assert`` — a
  no-op under the canonical ``python -O`` runner, and `[M]` with **zero**
  witnesses tree-wide.  It is now a typed ``TypeError`` with a test that
  fires in optimized and debug mode alike: net-new coverage, not a
  migration.
* **The pole misnomer died.**  A cylinder has no pole in the sense a
  sphere does; what matters is that one closure is **spatial** and one is
  **angular**.  The hub attribute ``pole_angular_closure`` became
  ``angular_closure`` and the family base ``PoleAngularClosureBase``
  became
  :class:`~orpheus.sn.angular.closure.AngularClosureBase`, giving the
  operator and the representation one symmetric, greppable pair of slot
  names.  Member names
  (:class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`,
  ``IdentityAngularClosure``) are unchanged — only the family-defining
  spellings moved.  Genuine poles keep the word: the sphere's polar cap,
  the :math:`\mu = -1` starting direction, and Hébert's Carlson
  *coupled-pole* seed are all named correctly.

.. note:: **What did *not* change, and is easy to misread.**

   The hub still carries ``scheme`` and a bound ``angular_closure``: the
   campaign charter's original row (*"the mesh sheds both"*) was
   **revised by ruling**, not deferred, so a future reader who finds
   that row in an archived plan should read this section instead.

   The mesh's pairing predicate is likewise unchanged.  `[M]`
   ``SNMesh.__eq__`` is ``object.__eq__`` — identity, not value — so
   nothing about equality moved; what compares constituents is
   :meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.is_same_phase_space`,
   and it compares the scheme **by type** while deliberately EXCLUDING
   the angular closure (a solve-time sweep strategy changes neither the
   field layout nor the quadrature two solutions contract over, so
   fields from two closures stay contractible).  Do not "strengthen"
   that predicate by adding the closure to it.

   The remaining misnomer is the *name*
   :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` itself, which is
   tracked as its own rename issue; the object's role as save state and
   data hub is settled.


Quadrature Dispatch
-------------------

The geometry-and-quadrature dispatch is a first-class polymorphism:
:func:`~orpheus.sn.loss_representation.default_for` selects the
:class:`~orpheus.sn.loss_representation.LossRepresentation` whose declared
``supports`` predicate matches the mesh — the 1-D chain scan
(:class:`~orpheus.sn.loss_representation.CumprodScan`, any geometry) or the
multi-D anti-hyperplane wavefront — and the operator then calls it
branchlessly.  (This replaced the pre-carve procedural branch on the
``SNMesh.curvature`` string tags and the since-retired operator-free
``transport_sweep`` entry; the full carve is documented in
:doc:`/theory/methods/sn/loss_representation`.)  Boundary conditions are **not** passed as
a parameter to the sweep --- it reads the resolved BC kind strings
directly from the face-name-keyed :attr:`SNMesh.bc` dict
(``sn_mesh.bc["xmin"]``, ``sn_mesh.bc["xmax"]``, ...; see
:ref:`bc-face-name-carve`).

For 1D meshes (``ny=1``):

- **Gauss--Legendre** quadrature takes the fast 1-D chain-scan
  (:class:`~orpheus.sn.loss_representation.CumprodScan`) path (all
  :math:`\mu_y = 0`, so no y-streaming).
- **Lebedev** quadrature falls through to the 2D wavefront sweep.
  Ordinates with :math:`\mu_x \neq 0` stream along *x*; the
  *y*-streaming terms cancel via reflective BCs on the single-cell
  *y*-dimension.  Ordinates with :math:`\mu_x = \mu_y = 0`
  (z-directed) reduce to pure collision:
  :math:`\psi = Q \cdot w_{\text{norm}} / \Sigt{}`.

Both quadratures recover the analytical eigenvalue exactly on
homogeneous problems (verified to machine precision for 1G/2G/4G).


The Transport Equation
======================

The 1-D slab form — the base of the broadening progression — is
posed in :doc:`slab_one_group`; the multi-group energy extension in
:doc:`slab_multigroup`; the 2-D Cartesian form in
:doc:`cartesian_multid`. The curvilinear geometries below extend
them.

Spherical 1D
-------------

In spherical coordinates the transport equation acquires an **angular
redistribution term** that couples ordinates:

.. math::
   :label: transport-spherical

   \mu \frac{\partial \psi}{\partial r}
   + \frac{1 - \mu^2}{r} \frac{\partial \psi}{\partial \mu}
   + \Sigt{} \psi = \frac{Q}{W}

The curvature term :math:`(1 - \mu^2)/r \cdot \partial\psi/\partial\mu`
arises because a neutron streaming radially at angle :math:`\mu` *rotates*
its direction cosine as it moves to a different radius.  Discretising this
term requires :term:`diamond difference` in **both space and angle**.

Cylindrical 1D
---------------

For an infinitely long cylinder with azimuthal symmetry, the transport
equation in the radial variable :math:`r` is:

.. math::
   :label: transport-cylindrical

   \frac{\eta}{r} \frac{\partial(r\psi)}{\partial r}
   - \frac{1}{r} \frac{\partial(\xi\psi)}{\partial\varphi}
   + \Sigt{} \psi = \frac{Q}{W}

where the direction cosines are:

- :math:`\eta = \sin\theta\cos\varphi` --- radial projection (streaming)
- :math:`\xi = \sin\theta\sin\varphi` --- azimuthal component
- :math:`\mu = \cos\theta` --- axial component

The constraint :math:`\eta^2 + \xi^2 + \mu^2 = 1` holds.  The azimuthal
redistribution :math:`-\partial(\xi\psi)/\partial\varphi` couples ordinates
on each :math:`\mu`-level.

The Discrete Balance Equation
=============================

This is the core of the S\ :sub:`N` method.  The balance equations are
presented from simplest to most complex, in the progression chapters:
Cartesian geometries have no angular redistribution — the 1-D slab
balance is derived in :doc:`slab_one_group`, the 2-D Cartesian balance
in :doc:`cartesian_multid`; curvilinear geometries add :math:`\alpha`
coupling and a geometry factor :math:`\Delta A/w` —
:doc:`curvilinear_one_group`.

.. _cell-update-strategies:

Cell update strategies (the strategy contract)
==============================================

The discrete balance equation (the slab DD :eq:`dd-cartesian-1d` of
:doc:`slab_one_group`, the M-M-closed curvilinear update :eq:`dd-solve`
of :doc:`curvilinear_one_group`) yields, for
each cell, a small algebraic system: combine the upstream face flux
(and, for sphere/cylinder, the upstream angular half-flux) with a
local source and the cell's total cross section to produce the
cell-average flux plus the downstream states.  The closure relating
:math:`\overline{\psi}_i` to :math:`\psi_{i-1/2}` and
:math:`\psi_{i+1/2}` is **not unique** — Diamond Difference (DD),
weighted DD, Linear Discontinuous (LD), Step, and Exponential
Characteristic (EC) are all valid choices, each with different
truncation error, positivity, and cost.  Per Cardinal Rule 2
(architecture), the cell-update math is **the same algebra** in slab,
sphere, and cylindrical 1-D — only the populated fields of the
:class:`~orpheus.transport.spatial.scheme.StreamingTerms` packet
change.  Lifting the closure into a strategy contract makes the
sweep driver thin and lets each closure be unit-tested in isolation.

The strategy contract is owned by
:mod:`orpheus.transport.spatial.scheme`.

The Protocol
------------

The contract is a ``@runtime_checkable`` ``typing.Protocol`` —
satisfied by structural typing, not inheritance — exposing two class-
level traits and a single :meth:`update` method:

* :class:`~orpheus.transport.spatial.scheme.DiscretizationScheme`

  - ``is_linear: bool`` — whether the closure is linear in its inputs.
    Diamond Difference is linear; Step's positivity-fixup, weighted-DD
    with a flux-dependent weight, and EC with a clipped argument are
    not.
  - ``is_positivity_preserving: bool`` — whether non-negative inputs
    guarantee non-negative outputs.  Diamond Difference is **not**
    positivity preserving (Lewis & Miller §5.3, where DD's tendency
    to produce negative cell-edge fluxes is exhibited and motivates
    the choice of Step or weighted-DD in stiff cells); Step is.
  - ``update(visit, total_xs, source, upstream_state) ->
    CellResult`` — the cell update itself.  ``visit`` is a
    :class:`~orpheus.transport.spatial.scheme.CellVisit` packet (see
    next subsection) that combines the per-(cell, direction)
    :class:`~orpheus.transport.spatial.scheme.StreamingTerms` with
    sweep-direction-resolved data.

The two helper dataclasses (frozen, slotted) carry the per-cell
state:

* :class:`~orpheus.transport.spatial.scheme.UpstreamState`

  - ``spatial_upstream: np.ndarray`` — shape ``(ng,)``.  Face flux
    entering the cell from the upstream face.  Since P4.9a this is the
    dataclass's **only** field: the state a spatial scheme is handed is
    purely spatial.

* :class:`~orpheus.transport.spatial.scheme.CellResult`

  - ``cell_average_flux: np.ndarray`` — shape ``(ng,)``.  The cell-
    average flux :math:`\overline{\psi}_i = \mathrm{numer}/\mathrm{denom}`
    from the closure's algebraic solve.
  - ``outgoing_spatial_flux: np.ndarray | None`` — shape ``(ng,)`` in
    the typical case; ``None`` for the cylindrical pure-azimuthal
    degenerate case where the cell has no radial face flow (see
    below).

.. note:: **The angular slots left the visit family (P4.9a, 2026-08-28).**

   ``UpstreamState`` carried ``angular_upstream: np.ndarray | None``
   (:math:`\psi_{n-1/2,\,i}`, the upstream half-angle flux) and
   ``CellResult`` carried ``outgoing_angular_state: np.ndarray | None``
   (:math:`\psi_{n+1/2,\,i}`) until 2026-08-28.  Both are **retired**.
   They existed so that a *spatial* scheme could carry the
   Morel--Montry angular thread in and out — the ``None`` on each was
   the slab case.  P4.9a moved the march to its owner (see
   :ref:`sn-p49a-closure-owns-the-march`), which left the two slots with
   nothing to carry, and with them went ``CellVisit``'s closure stamp
   (``tau`` / ``c_in`` / ``c_out``) and the mesh-side
   ``SNMesh._make_cell_visit`` that wrote it.

   What replaces them is *not* a renamed slot but a different kind of
   datum: the two **assembled** contributions
   ``angular_denom_term`` and ``angular_numer_upstream``, passed as
   keyword arguments to
   :meth:`~orpheus.transport.spatial.scheme.DiscretizationScheme.update`
   and ``residual``.  They are already multiplied out — no closure
   constant, no half-angle thread, and nothing for the scheme to
   advance.  A scheme that sees them cannot tell an M-M closure from
   any other; both default to the neutral element (``0.0`` and
   ``None`` → zeros), which is exactly the slab case.

The SN sweep DAG and ``CellVisit``
-----------------------------------

The SN sweep is a **topological sort of a directed cell graph**.
For a given ordinate :math:`\Omega_n`, every face :math:`f` of the
mesh is oriented by the sign of :math:`\Omega_n \cdot \hat n_f` — an
edge from cell :math:`A` to cell :math:`B` if :math:`\Omega_n` points
from :math:`A` into :math:`B` across that face.  The sweep walks
cells in a topological order over this DAG so that, when each cell is
visited, all its upstream face fluxes are already known.  This is
the SN-specific graph-theoretic concept; MoC uses a different
mathematical structure (fiber bundles + solution sheaves over
characteristic curves), and CP / diffusion / MC have no sweep at
all.  Per Cardinal Rule 2 (architecture), no shared
``SweepGraph`` Protocol is hoisted across solvers — the sweep DAG
lives in :mod:`orpheus.sn`.

The contract's :meth:`update` consumes a
:class:`~orpheus.transport.spatial.scheme.CellVisit` packet rather
than a raw
:class:`~orpheus.transport.spatial.scheme.StreamingTerms`.
The :class:`CellVisit` composes:

* ``cell_idx: int`` — the cell being visited.
* ``streaming_terms: StreamingTerms`` — the cell's **evaluation-point**
  primitive: the mesh's metric data (``face_area_inner`` /
  ``face_area_outer`` / ``volume``) together with the ordinate's
  ``abs_mu``.  The two face-area names are *geometric* labels — inner =
  closer to :math:`r=0`, outer = farther — independent of sweep
  direction.  ⛔ This bullet called the packet **purely geometric**
  until 2026-08-28; ``abs_mu`` is the quadrature's, not the mesh's, and
  a spatial closure is legitimately *parameterized* by direction without
  being angular-closure-aware.
* ``face_area_downstream: float | None`` — **sweep-direction-
  resolved**.  For an outward sphere or cylinder sweep
  (:math:`\mu \ge 0`) it equals ``streaming_terms.face_area_outer``;
  for an inward sweep (:math:`\mu < 0`) it equals
  ``streaming_terms.face_area_inner``.  ``None`` for slab (slab DD
  does not read face areas) and for the cylindrical pure-azimuthal
  degenerate case (no spatial flow).

The :class:`CellVisit` packets are produced by
:meth:`SNMesh.dag_walk(*, ordinate_idx=..., direction_sign=...,
mu_level_idx=None)
<orpheus.sn.mesh.augmented_mesh.SNMesh.dag_walk>` — a generator that
yields cells in DAG-topological order.  The method takes EXACTLY ONE
of ``ordinate_idx`` (single-ordinate visits, used by the sweep
driver) or ``direction_sign`` (direction-keyed visits, used by the
apply matvec) as a keyword-only argument.  Both invocation modes
encapsulate the inward / outward branching, the cylindrical
per-:math:`\mu`-level traversal, and the pure-azimuthal degenerate
handling.  The sweep at ``orpheus.sn.loss_representation`` (the dissolved ``sweep.py``) consumes this
generator::

    for visit in sn_mesh.dag_walk(ordinate_idx=n):
        upstream = UpstreamState(spatial_upstream=psi_face)
        # ΔA/w from its two factors (P4.7 — the packet no longer
        # carries the fusion; the closure owns it, the cache interns it).
        # quad here is sn_mesh.quad — a HUB read; see the note below on
        # why the walk's quadrature reads deliberately stay hub-handed.
        dA_w = float(reduced.delta_A[visit.cell_idx] / quad.weights[n])
        result = scheme.update(
            visit=visit,
            total_xs=total_xs,
            source=source,
            upstream_state=upstream,
            angular_denom_term=dA_w * c_out_n,
            angular_numer_upstream=dA_w * c_in_n * psi_angle[:, visit.cell_idx],
        )
        psi_angle[:, visit.cell_idx] = closure.advance_psi_half(
            result.cell_average_flux, psi_angle[:, visit.cell_idx],
            ordinate=n,
        )
        ...

.. note:: **Why the walk still reads the quadrature off the hub, on
   purpose.**

   The P4-remainder (2026-08-29) made the streaming *producer* reach the
   angular geometry **through its bound space factor** — the quadrature
   is the generator of
   :attr:`ReducedStreamingOperator.angular_axis
   <orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator.angular_axis>`,
   recovered by a typed narrow, and the courier field it used to read
   through is gone (:ref:`spaces-generator-route-gate`).  It did **not**
   re-point the walk.  `[M]` there are 19 ``.quad`` reads across
   :mod:`orpheus.sn.loss_representation` (15 in the module body, 2 in
   ``assembly.py``, 2 in ``sweep_schedule.py``), every one of them
   handed the hub, and they stay.

   The reason is a role distinction, not a backlog.  A *producer* that
   is handed a space and then reaches past it for the same data is
   carrying a redundant reference — the defect the re-point removed.
   The walk is not in that position: it is handed an
   :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` and nothing else, so
   spelling its reads as a lookup on the hub's own angular bulk space
   followed by a generator narrow would reach the same object by a
   longer path and record a discipline the layer does not yet have —
   theatre, not architecture.  The reads un-weld when the *strategy*
   layer starts
   receiving spaces rather than a carrier, which is Campaign 2's work
   and a different change to this one.

The cell-update strategy receives only **resolved** data — no
sign-of-:math:`\mu` branching inside the strategy.  This pattern
moves the graph-theoretic concept to where it belongs (the SN
module) and keeps the
:class:`~orpheus.transport.spatial.scheme.StreamingTerms` packet free
of any sweep-frame datum — the sweep-direction concept is
S\ :sub:`N`'s alone: MoC, CP, diffusion and MC have no sweep.  (⛔
This sentence continued "and reusable by future MoC / CP / diffusion
modules that have different mathematical structures" until 2026-08-27.
Withdrawn: geometry-only is not the same claim as shared, and those
families do not form an angular redistribution term at all — see
:ref:`who-needs-a-connection-coefficient`.  ⛔ And the sentence called
the packet "the geometry-layer StreamingTerms, geometry-only" until
2026-08-28, when P4.3 refuted the layer half too: the packet carries
``mu``/``abs_mu`` and :math:`\Delta A/w` — direction-bearing posing,
not geometry — and now lives in ``transport/spatial/scheme.py`` beside
the scheme contract that consumes it.)

Slab vs curvilinear discrimination
-----------------------------------

.. note:: **Superseded mechanism (Issue #196 Phase G Step 2.5 → Issue
   #236 Step C).** The ``alpha_in is None`` slab/curvilinear
   discrimination described below was **retired**: Issue #196 Phase G
   Step 2.5 gave slab *neutral* curvature (``face_area_inner =
   face_area_outer = 1.0``, ``delta_A_over_w = 0.0``) so the unified
   cell-balance helper consumes the same packet regardless of geometry,
   and Issue #236 Step C removed the Morel--Montry ``alpha_in`` /
   ``alpha_out`` / ``tau_mm`` fields from
   :class:`~orpheus.transport.spatial.scheme.StreamingTerms` entirely
   (leaving no closure field on the packet — though "purely geometric"
   it is not, a reading refuted 2026-08-28: it keeps ``abs_mu``, which
   is the ordinate's and not the mesh's; :math:`\tau` is closure-owned —
   see :ref:`sn-tau-c-on-cellvisit-live`.  ⛔ That refutation named
   ``mu`` and :math:`\Delta A/w` alongside ``abs_mu`` until 2026-08-29,
   when P4.7 shed all three of ``mu``, ``chord_length`` and
   ``delta_A_over_w`` — the conclusion is unchanged and now rests on a
   single field, which is the strongest form of it: **one**
   direction-bearing datum is enough to make the packet not
   geometry-only).  ⛔ A second reading died on
   2026-08-28 as well: this note read *"slab is now distinguished at the
   sweep level by* ``upstream_state.angular_upstream is None``\ *"* until
   P4.9a retired that field.  **A spatial scheme no longer distinguishes
   slab from curvilinear at all** — see below.  The prose after this note
   records the historical pre-Step-2.5 convention; the authoritative
   current description is the
   :class:`~orpheus.transport.spatial.scheme.StreamingTerms` docstring.

Since P4.9a there is no slab-vs-curvilinear test in a spatial scheme,
because there is no longer a question to answer.  The curvature data a
scheme consumes arrives already reduced to two numbers whose slab values
are the neutral element of the arithmetic they enter:
``angular_denom_term = 0.0`` adds nothing to the denominator, and a
``None`` ``angular_numer_upstream`` adds a zero array to the upstream
numerator.  What survives inside
:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.update` is
one structural test — ``visit.face_area_downstream > 0.0``, *"is there a
downstream spatial face?"* — which is **not** geometry dispatch: it is
``False`` for exactly one case, the cylindrical pure-azimuthal
degenerate cell, and ``True`` for slab and non-degenerate curvilinear
alike.

The historical convention this section records was a single field test:

* **Slab** — ``visit.streaming_terms.alpha_in is None`` (and the rest
  of the curvature bundle, ``alpha_out``, ``delta_A_over_w``,
  ``tau_mm``, ``face_area_inner`` / ``face_area_outer``, are all
  ``None``).  ``upstream_state.angular_upstream is None``.  The
  strategy returned ``CellResult(outgoing_angular_state=None, ...)``.
* **Sphere or cylinder** —
  ``visit.streaming_terms.alpha_in is not None``; the full curvature
  bundle was populated.  ``upstream_state.angular_upstream`` carried
  :math:`\psi_{n-1/2,\,i}`.  The strategy returned the M-M-closed
  ``outgoing_angular_state``.

That convention was locked in by foundation-tier protocol-conformance
tests in ``tests/sn/sweep/core/test_discretization_scheme_protocol.py``;
those tests now pin the successor contract — the assembled-contribution
keyword arguments and the purely spatial
:class:`~orpheus.transport.spatial.scheme.UpstreamState` /
:class:`~orpheus.transport.spatial.scheme.CellResult`.

Cylindrical pure-azimuthal degenerate case
-------------------------------------------

For cylindrical 1-D radial sweeps, ordinates with axial direction cosine
:math:`|\mu_z| \to 1` have radial direction cosine
:math:`|\eta| = \sqrt{1 - \mu_z^2} \to 0`.  This is a property of the
polar level, not of any one rule family: it holds on the admitted
:math:`\sigma_y`-folded product family exactly as it did on the
full-circle product and level-symmetric rules those replaced (refused at
cylindrical ``SNMesh`` admission since Q5.6.3 —
:ref:`sn-direct-seed-r12a`).  In this limit the cell
has **no radial face flow** — the streaming term
:math:`\mu_x \cdot \partial_r` vanishes — and the cell-update
algebra collapses to the redistribution-only form

.. math::

   \mathrm{denom} = (\Delta A / w)\,c_{\rm out} + \Sigma_t\,V_i,
   \qquad
   \mathrm{numer} = Q_i\,V_i + (\Delta A/w)\,c_{\rm in}\,\psi_{n-1/2,\,i},

with no spatial-flux contribution.  The strategy contract signals
this case by setting ``CellResult.outgoing_spatial_flux = None``: the
sweep driver, on receiving ``None``, skips the face-flux update for
that cell.  The angular M-M closure remains active — angular
redistribution physics is still present — but since P4.9a it is the
**walk**, not the scheme, that applies it: the walk assembles
:math:`(\Delta A/w)\,c_{\rm out}` and
:math:`(\Delta A/w)\,c_{\rm in}\,\psi_{n-1/2,i}` from the closure's own
per-ordinate arrays, hands them to
:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.update` as
the two assembled contributions above, and then advances the half-angle
thread itself through
:meth:`~orpheus.sn.angular.closure.AngularClosureBase.advance_psi_half`.
This is the one production path that visits degenerate cells one at a
time (:ref:`sn-p49a-closure-owns-the-march`).

The numerical threshold is ``streaming_terms.abs_mu < 1e-15``, with
``abs_mu`` populated from the **global ordinate**
:math:`|\eta|` on the streaming-terms packet (resolved through
``level_indices`` for cylindrical geometry — see
:doc:`/theory/foundations/structured_geometry`, "Connection coefficients (reduced
streaming operator)").  In this case
:meth:`SNMesh.dag_walk
<orpheus.sn.mesh.augmented_mesh.SNMesh.dag_walk>` yields visits with
``face_area_downstream = 0.0`` to signal "no spatial flow" to the
strategy (Issue #196 Step 2.5 retired the ``None`` sentinel — the
slab carries ``1.0`` and degenerate cylindrical carries ``0.0`` so
the cell-balance helper consumes one geometry-blind number).

The DD recurrence
------------------

For non-degenerate cells, the closure relation reduces — for slab
geometry, the cell-update math is the DD recurrence
:eq:`dd-recurrence` (derived at :ref:`sweep-cumprod` in
:doc:`slab_one_group`); for curvilinear
geometry, it is the curvilinear DD form combining the
:math:`\Delta A/w` redistribution with the M-M angular closure.  The
sweep driver inlines this math today; Wave D (Issue #159) will
rewrite the driver to dispatch through a
:class:`~orpheus.transport.spatial.scheme.DiscretizationScheme` strategy.

The first concrete strategy — :class:`DiamondDifference` — is shipped
in Round 2 of Wave C (Issue #158) as a bit-identical extraction of
the existing inlined sweep math.  Linear Discontinuous (Lewis &
Miller §5.3 — preview), Exponential Characteristic, and Step
strategies are deferred to a Wave C-extension session, each with its
own MMS spatial-convergence verification.

Diamond Difference
------------------

The first concrete strategy is
:class:`~orpheus.transport.spatial.diamond.DiamondDifference`
(:mod:`orpheus.transport.spatial.diamond`).  It implements the **same**
algebra as the existing inlined sweep — Round 2 of Wave C is a
bit-identical extraction, gated by ``np.array_equal`` hand-calc tests
in ``tests/sn/sweep/core/test_diamond.py`` against the sweep's scalar
formulas at ``orpheus.sn.loss_representation`` (the dissolved ``sweep.py``).

Per Wave C decision **D5** (one geometry-polymorphic class), the
strategy is a single :class:`DiamondDifference` that handles slab,
sphere and cylinder in **one body with no geometry dispatch**.  The
three headings below are therefore *cases of the same formula*, not
branches of the code: they name the values the incoming data takes, and
the reader can check that each collapses out of the general form.

⛔ This paragraph read *"…by branching on two* ``StreamingTerms`` *fields:*
``alpha_in is None`` *(slab vs curvilinear) and* ``abs_mu < 1e-15``
*(cylindrical pure-azimuthal degenerate vs not)"* until 2026-08-28.
Neither test survives: Issue #236 Step C deleted the ``alpha_*`` /
``tau_mm`` fields, and the degenerate case is signalled by the
*geometric* ``visit.face_area_downstream == 0.0``, never by a numerical
threshold on :math:`|\mu|`.

**Slab case** (neutral curvature: ``face_area_inner = face_area_outer =
1.0``, ``delta_A_over_w = 0.0``, ``angular_denom_term = 0.0``).  The flat /
Cartesian DD recurrence reduces to the per-cell scalar form of
:eq:`dd-recurrence`:

.. math::
   :label: dd-slab-scalar

   \overline{\psi}_i \;=\; \tfrac12\bigl(\psi_{i-1/2}
                                         + \psi_{i+1/2}\bigr),
   \qquad
   \psi_{i+1/2} \;=\; \frac{2|\mu_n| - \Delta x_i\,\Sigma_t}
                              {2|\mu_n| + \Delta x_i\,\Sigma_t}\,
                       \psi_{i-1/2}
                   \;+\; \frac{2\,Q_i\,\Delta x_i / W}
                              {2|\mu_n| + \Delta x_i\,\Sigma_t},


.. implements:: dd-slab-scalar
   :by: orpheus.sn.mesh.reduced_operator.slab_streaming

   **Implemented by** 9 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.  P4.9a **migrated** one of the nine rather than dropping it:
   the scalar helper ``cell_balance_terms`` was retired onto
   :func:`~orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`,
   which the slab case reaches through
   ``DiamondDifference``'s ``n_mask=1`` bridge and which the module
   docstring documents as producing the slab denominator
   :math:`2|\mu|\cdot 1 + 0 + \Sigma_t V` from neutral curvature.
   Retiring a declared symbol without migrating its edge would have left
   this equation with eight implementers and one silent hole.

.. implements:: dd-slab-scalar
   :by: orpheus.transport.spatial.cell_balance.cell_balance_for_streaming

.. implements:: dd-slab-scalar
   :by: orpheus.transport.spatial.diamond.DiamondDifference.affine_scan_coefficients

.. implements:: dd-slab-scalar
   :by: orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch

.. implements:: dd-slab-scalar
   :by: orpheus.transport.spatial.diamond.DiamondDifference.update

.. implements:: dd-slab-scalar
   :by: orpheus.transport.spatial.diamond._DD_W

.. implements:: dd-slab-scalar
   :by: orpheus.transport.spatial.scheme.DiscretizationSchemeBase.cell_average

.. implements:: dd-slab-scalar
   :by: orpheus.transport.spatial.scheme.DiscretizationSchemeBase.outgoing_face_from_average

.. implements:: dd-slab-scalar
   :by: orpheus.transport.spatial.scheme.DiscretizationSchemeBase.source_emission

with ``W = Σ_n w_n`` the quadrature weight sum, mirroring the
sweep's vectorised cumprod path
(``_sweep_1d_cumprod`` (the dissolved ``sweep.py``) lines 117–123) and per-
cell solver (``_solve_recurrence`` (the dissolved ``sweep.py``) lines 208–
222) at the operation level.  Per the strategy contract, ``source``
arrives at the cell update **already weight-normalised** by the
sweep — for slab, ``source = Q · Δx / W`` (and slab cell volume is
``V = Δx``).  ⛔ This paragraph closed *"For slab, the strategy sets*
``CellResult.outgoing_angular_state`` *to* ``None`` *— slab geometry has
no angular redistribution"* until 2026-08-28.  The field is retired: a
DD cell update returns no angular state on **any** geometry, and the
reason is structural rather than geometric — a spatial discretization
scheme closes the spatial axis only.

**Curvilinear case** (physical curvature, with a downstream spatial
face).  Sphere or cylinder, away from the cylindrical
pure-azimuthal degenerate case.  Here the M-M angular closure
:eq:`mm-weights` and the WDD spatial closure :eq:`wdd-closure` meet —
but they meet *in the balance*, not inside one scheme: the closure
supplies the redistribution constants

.. math::
   :label: dd-mm-closure-constants

   c_{\rm out} \;=\; \alpha_{n+\tfrac12}/\tau_n,
   \qquad
   c_{\rm in}  \;=\; \frac{1 - \tau_n}{\tau_n}\,\alpha_{n+\tfrac12}
                       + \alpha_{n-\tfrac12},

built from the :math:`\alpha` dome :eq:`alpha-recursion` and the
Morel–Montry weight :eq:`mm-weights`.  The cell-update is then

.. math::
   :label: dd-curvilinear-scalar

   \overline{\psi}_{n,i} \;=\;
       \frac{Q_i\,V_i / W
             + |\mu_n|\,(A_{i-1/2} + A_{i+1/2})\,\psi^s_{n,\,{\rm in}}
             + (\Delta A_i / w_n)\,c_{\rm in}\,\psi_{n-\tfrac12,\,i}}
            {2|\mu_n|\,A^s_{\rm out}
             + (\Delta A_i / w_n)\,c_{\rm out}
             + \Sigma_t\,V_i},


.. implements:: dd-curvilinear-scalar
   :by: orpheus.transport.spatial.cell_balance.cell_balance_for_streaming

   **Implemented by** 5 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.  ⛔ There were **6** until 2026-08-28: P4.9a retired the
   scalar twin ``cell_balance_terms``, whose declaration is **removed**
   rather than migrated, because the survivor
   :func:`~orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`
   was already declared here — the retirement collapsed two
   implementers into one, it did not orphan an edge.

.. implements:: dd-curvilinear-scalar
   :by: orpheus.transport.spatial.diamond.DiamondDifference.affine_scan_coefficients

.. implements:: dd-curvilinear-scalar
   :by: orpheus.transport.spatial.diamond.DiamondDifference.residual

.. implements:: dd-curvilinear-scalar
   :by: orpheus.transport.spatial.diamond.DiamondDifference.update

.. implements:: dd-curvilinear-scalar
   :by: orpheus.derivations.discrete.sn.balance.derive_wdd_solve

mirroring ``_sweep_1d_spherical`` (the dissolved ``sweep.py``) lines
350–355 (and the structurally identical cylindrical branches at
sweep.py:511–531 / sweep.py:548–575) verbatim, with two closures —
**one per axis, and since P4.9a one owner each**:

.. math::

   \underbrace{\psi^s_{\rm out} \;=\; 2\overline{\psi}_{n,i}
                        - \psi^s_{n,\,{\rm in}}}_{\text{spatial — the scheme}},
   \qquad
   \underbrace{\psi_{n+\tfrac12,\,i} \;=\;
       (\overline{\psi}_{n,i}
         - (1 - \tau_n)\,\psi_{n-\tfrac12,\,i})/\tau_n}_{\text{angular — the closure}}.

The left-hand relation is DD's, and
:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.update`
returns its value as ``outgoing_spatial_flux``.  The right-hand one is
the Morel--Montry march; until 2026-08-28 ``update`` evaluated it too, as
a second inline expression, and returned it as ``outgoing_angular_state``.
It is now applied by whoever composes the two axes — the SN walk today,
the ``StreamingOperator`` after P4.9b — through the closure's own
:func:`~orpheus.sn.angular.closure.march_psi_half_step`
(:ref:`sn-p49a-closure-owns-the-march`).

**Cylindrical pure-azimuthal degenerate case**
(``visit.face_area_downstream == 0.0``).
For a level whose axial direction cosine :math:`|\mu_z| \to 1`, the
radial direction cosine :math:`|\eta| \to 0` and the cell has no
radial face flow — the :math:`2|\mu| A_{\rm out}` and
:math:`|\mu|(A_{\rm in} + A_{\rm out})\,\psi^s_{\rm in}`
contributions drop out:

.. math::
   :label: dd-cylindrical-degenerate

   \mathrm{denom} \;=\; (\Delta A / w)\,c_{\rm out} + \Sigma_t\,V_i,
   \qquad
   \mathrm{numer} \;=\; Q_i\,V_i / W
                       + (\Delta A / w)\,c_{\rm in}\,
                          \psi_{n-\tfrac12,\,i},


.. implements:: dd-cylindrical-degenerate
   :by: orpheus.transport.spatial.cell_balance.cell_balance_for_streaming

   **Implemented by** 4 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: dd-cylindrical-degenerate
   :by: orpheus.transport.spatial.diamond._cell_balance_n1

.. implements:: dd-cylindrical-degenerate
   :by: orpheus.transport.spatial.diamond.DiamondDifference.update

.. implements:: dd-cylindrical-degenerate
   :by: orpheus.sn.loss_representation._OneDimScanWalk._run

mirroring ``_sweep_1d_cylindrical`` (the dissolved ``sweep.py``) lines
533–543 verbatim.  The strategy returns
:attr:`~orpheus.transport.spatial.scheme.CellResult.outgoing_spatial_flux`
``= None`` to signal "no face-flux write" to the sweep driver; the
M-M angular closure remains active.

.. note:: **Why** ``update`` **still implements this equation after
   P4.9a — and why the walk now does too (2026-08-28).**

   The obvious reading of the carve is that
   :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.update`
   should lose this declaration along with the march.  It should not,
   and the discriminator is what the **equation states**.
   :eq:`dd-cylindrical-degenerate` is a statement about
   :math:`\mathrm{denom}` and :math:`\mathrm{numer}` — the degenerate
   **cell balance**, whose whole content is that the two
   :math:`|\mu|`-weighted face terms drop out and the redistribution
   pair is all that is left beside collision and source.  ``update``
   still forms that quotient (through
   :func:`~orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`
   at ``n_mask = 1``, where the drop-out happens *geometrically*, via
   ``A_downstream = face_area_total = 0.0``, not by a threshold on
   :math:`|\mu|`).  What ``update`` no longer does is evaluate
   :math:`\psi_{n+1/2,i}` — and that relation is **not written in this
   equation**.  It is :eq:`dd-mm-angular-recurrence`, and that is the
   label whose implementers moved.

   The genuinely new implementer is
   :meth:`!orpheus.sn.loss_representation._OneDimScanWalk._run`, and it
   is owed one because the equation writes the two redistribution
   contributions as **products**, :math:`(\Delta A/w)\,c_{\rm out}` and
   :math:`(\Delta A/w)\,c_{\rm in}\,\psi_{n-1/2,i}`.  Those products are
   no longer formed anywhere below the walk: the walk multiplies the
   closure's own per-ordinate ``c_in`` / ``c_out`` arrays by the visit's
   :math:`\Delta A/w` and passes the results down as
   ``angular_denom_term`` / ``angular_numer_upstream``.  Declaring only
   the balance sites would leave the equation's two most
   geometry-specific factors implemented by nothing.
   ``_cell_balance_n1`` — the single scalar-to-vectorized conversion the
   solve and apply directions share — is declared for the same reason:
   it is where the ``n_mask = 1`` shapes are built and the ``None``
   default becomes the zero array.

**Traits and forward references.**  Diamond Difference has

* :attr:`~orpheus.transport.spatial.diamond.DiamondDifference.is_linear`
  ``= True`` — the cell average and downstream states are affine
  combinations of ``source`` and ``upstream_state``;
* :attr:`~orpheus.transport.spatial.diamond.DiamondDifference.is_positivity_preserving`
  ``= False`` — Lewis & Miller §5.3 exhibits the canonical thin-
  cell / large-source counter-example where DD's
  :math:`\psi_{\rm out} = 2\overline{\psi} - \psi_{\rm in}` produces
  negative outgoing flux from positive inputs.

Of the three planned alternatives, one has landed:
:class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`
(:math:`\mathcal{O}(\Delta x^2)`, better robustness in optically-thick
cells) ships today under the registry key ``"linear_discontinuous"``,
with its own MMS spatial-convergence gates
(``tests/sn/verification/mms/test_mms_ld_slab.py`` and
``test_mms_ld_2d.py``); see :ref:`ld-ubld-multidim` for the derivation
and the multi-dimensional wiring.  The other two are still **reserved,
not yet implemented**, and are therefore written as literals rather than
``:class:`` roles — a live role would assert a class that does not
exist: ``Step`` (positivity-preserving,
:math:`\mathcal{O}(\Delta x)`) and ``ExponentialCharacteristic``
(positivity-preserving by construction).  Each lands with its own MMS
spatial-convergence verification.

References
----------

* Lewis, E. E., & Miller, W. F. (1984). *Computational Methods of
  Neutron Transport*.  §5.3 covers Diamond Difference, weighted-DD,
  Step, and Linear Discontinuous closures and their positivity /
  truncation properties; §4.5 covers the Morel--Montry angular
  closure used for :math:`\psi_{n+1/2,\,i}`.
* Lathrop, K., & Carlson, B. (1966). *J. Comp. Phys.* 1:173 — the
  :math:`\alpha` recursion :math:`\alpha_{n+1/2} = \alpha_{n-1/2} -
  w_n \mu_n` (:eq:`alpha-recursion`); the implemented form is
  Hébert (2009) *Applied Reactor Physics* §3.9.3 (cylinder) /
  §3.9.4 (sphere), Eqs. 3.423-3.424, which is also the authority for
  the cell balance, the :math:`\Delta A / w` factor and the
  :math:`\alpha_{1/2} = 0` seed.
* Morel, J. E., & Montry, G. R. (1984). *Analysis and Elimination of
  the Discrete-Ordinates Flux Dip*.  Transport Theory and Statistical
  Physics 13(5):615--633 — **primary** source for the weighted
  angular closure :math:`\tau` (:eq:`mm-weights`).  The form
  implemented here is :cite:`BaileyMorelChang2010` Eqs. (42)/(43);
  Reed & Lathrop (1970) Eq. (13c) is the same condition, 40 years
  earlier.

.. note:: ⛔ **Retracted citation (2026-08-27).**  This list carried
   *"Bailey, T. S., Adams, M. L., Yang, B., & Zika, M. R. (2009).*
   *A piecewise linear finite element discretization of the diffusion*
   *equation for arbitrary polyhedral grids. JCP 227, 3738--3757.*
   *Eq. 50 (dome recursion), Eq. 74 (Morel--Montry)"* — the
   **wrong Bailey paper**, a piecewise-linear FE *diffusion* paper
   unrelated to curvilinear S\ :sub:`N`.  Issue #168 Phase B retracted
   it in 2026 across ``orpheus.geometry.reduced_operator`` (since
   dissolved into :mod:`orpheus.sn.angular.redistribution` /
   :mod:`orpheus.transport.spatial.scheme`),
   :mod:`orpheus.transport.spatial.diamond` and
   :mod:`orpheus.sn.angular.closure`; this page and
   :mod:`orpheus.transport.spatial.scheme` were missed and kept
   asserting it.  Full account, including the second (Hébert-vs-BMC)
   correction, at :ref:`sn-citation-corrections`.  The two bullets
   above are the authorities that actually cover the two claims the
   retracted entry was cited for.

See also
--------

* :mod:`orpheus.transport.spatial.scheme` — the contract module.
* :mod:`orpheus.transport.spatial.diamond` — the
  :class:`~orpheus.transport.spatial.diamond.DiamondDifference` concrete
  strategy.
* :doc:`/theory/foundations/structured_geometry`, "Connection coefficients (reduced
  streaming operator)" — the upstream side of the contract: where the
  per-cell, per-direction streaming-terms packet is built.


.. _sweep-algorithm:

Sweep Algorithm
===============

Because each cell's outgoing flux becomes the next cell's incoming flux,
the equations must be solved in the direction of neutron travel --- this
is called a **transport sweep**.

The 1-D slab sweep — the cumprod recurrence and the generic affine
outflow reconstruction — is derived in :doc:`slab_one_group`; the 2-D
Cartesian wavefront, its octant dependency graph, and the
multi-dimensional LD (UBLD) system in :doc:`cartesian_multid`; the 2-D
LD stress MMS in :doc:`/theory/verification/sn`; the curvilinear machinery — the
sequential ordinate sweep, the angular closure (#168 Phase B),
the sweep-frame apply matvec (Phase C), and the direct :math:`\psi_{1/2}`
starting-direction solve (#282 route (a)) — in
:doc:`curvilinear_one_group` (the group axis rides that machinery as
data — :doc:`curvilinear_multigroup`). The #168 Phases A/D/F, ERR-058,
and #196
campaign record is preserved in :doc:`curvilinear_numerics`; the
section below preserves the dispatch-consolidation record (Wave D
Round 2, superseded by the ``LossRepresentation`` polymorphism).

.. _unified-sweep-dispatch:

Unified sweep dispatch
-----------------------

.. note::

   **Superseded (coupled-block campaign step 6, R-6.1, 2026-07-12).**
   This section records the Wave-D Round-2 consolidation (#161) — the
   *first* unification of the four sweep paths under a single entry, the
   operator-free ``transport_sweep``.  That entry was itself retired at
   step 6; the dispatch is now the first-class ``LossRepresentation``
   polymorphism (:func:`~orpheus.sn.loss_representation.default_for` selects
   :class:`~orpheus.sn.loss_representation.CumprodScan` for the 1-D scan,
   the anti-hyperplane wavefront for multi-D), documented in
   :doc:`/theory/methods/sn/loss_representation`.  The Wave-D narrative below is preserved as
   the origin of that unification: read ``transport_sweep`` and the
   ``ReducedStreamingOperator`` boolean-dispatch code as the then-current
   implementation, not today's.

Wave D Round 2 of the SN reshape campaign (Issue #161) consolidated
the four pre-existing sweep paths (1-D cumprod / 2-D wavefront /
spherical / cylindrical) under one operator-free ``transport_sweep``
entry point that branched
on a single boolean from the
:class:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator`
primitive (Wave B Issue #6 / Wave D Round 1):

.. code-block:: python

   def transport_sweep(Q, sig_t, sn_mesh, psi_bc, Q_aniso=None):
       reduced = sn_mesh.reduced
       if reduced is not None and reduced.requires_upstream_angular_state:
           return _curvilinear_sweep(Q, sig_t, sn_mesh, psi_bc, Q_aniso)
       return _cartesian_sweep(Q, sig_t, sn_mesh, psi_bc, Q_aniso)

The pre-Wave-D dispatch did string-equality on
``sn_mesh.curvature == "spherical"`` / ``"cylindrical"`` /
``None``.  Wave D replaced it with a geometry-layer boolean,
``ReducedStreamingOperator.requires_upstream_angular_state`` —
``False`` for slab + 2-D Cartesian (no angular redistribution
between successive half-angles), ``True`` for spherical +
cylindrical.  Two-D Cartesian set ``sn_mesh.reduced is None``
(no curvilinear math needed), and the dispatch fell through
to the Cartesian path.

.. note::

   **Both** of those spellings are now retired, so this section is
   two steps of history rather than one.  The boolean was retired on
   2026-08-26: it was exactly ``coord is not CoordSystem.CARTESIAN``
   and had no production reader, the concept having been respelled by
   ``upstream_state.angular_upstream is None`` (what the DD and LD
   cell bodies then branched on) and by
   :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.is_cartesian`.
   ⛔ Two days later P4.9a retired ``angular_upstream`` as well, so that
   respelling is history too: DD branches on nothing, and LD's
   curvilinear refusal was re-keyed onto **value** signals — unequal
   face areas, or a non-neutral assembled angular contribution — which
   is a stronger guard, because it is reachable by calling the scheme
   directly and cannot be dodged by a mesh that forgets to populate a
   field (see :ref:`sn-p49a-closure-owns-the-march`).
   Strategy selection today is
   :func:`~orpheus.sn.loss_representation.default_for`, which picks the
   first :data:`~orpheus.sn.loss_representation.LOSS_REPRESENTATIONS`
   entry whose ``supports`` admits the mesh **and the handed spatial
   closure**, keyed on ``is_1d`` **and** ``is_cartesian`` — neither
   alone is a sufficient discriminator.  Since P4.9b the predicate's
   signature is ``supports(mesh, spatial_closure)``: selection consumes
   the closure the operator was posed with, never ``mesh.scheme``
   (:ref:`sn-p49b-operator-poses-with-closures`).

Why this mattered:

* The :class:`ReducedStreamingOperator` is the primitive that
  already encodes "does this geometry need angular
  redistribution to march the sweep?", so the dispatch reads its
  property directly instead of round-tripping through a
  string tag — Cardinal Rule 2 (architecture).  (⛔ This bullet
  closed with "Consumers outside the SN sweep (MoC, CP) read the
  same property when they need the same dispatch" until
  2026-08-27.  There are none, and there will be none: neither
  method forms an angular redistribution term — see
  :ref:`who-needs-a-connection-coefficient`.  The dispatch-by-property
  argument stands on its own; it never needed a second consumer.)
* The dispatch surface shrinks from four string-equality checks
  to one boolean — a structural simplification that makes the
  control flow easier to reason about and to extend with
  additional cell-update strategies (Wave C-extension).

Cell update strategy parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The curvilinear sweep dispatches per-cell to
:meth:`~orpheus.transport.spatial.scheme.DiscretizationScheme.update`.
Both closures come from the walk's **own fields** — the pair it was
handed when the operator was posed — never from the mesh
(:ref:`sn-p49b-operator-poses-with-closures`); the mesh supplies the
geometry the walk traverses:

.. code-block:: python

   scheme = self.spatial_closure     # the walk's own field, handed at posing
   closure = self.angular_closure    # likewise — never sn_mesh.angular_closure
   # P4b: the constants' one durable home is the closure's read-only cache
   # (the geometry table sheds its copies).
   c_in_n = closure.c_in_per_ordinate[n]
   c_out_n = closure.c_out_per_ordinate[n]

   for visit in self.mesh.dag_walk(ordinate_idx=n, mu_level_idx=p):
       i = visit.cell_idx
       dA_w = geom.delta_A_over_w[n][k]   # the cache-interned ΔA/w row (P4.7)
       result = scheme.update(
           visit=visit,
           total_xs=sig_t[:, i],
           source=QV[:, i],
           upstream_state=UpstreamState(spatial_upstream=psi_face),
           # the WALK assembles the closure's balance contributions …
           angular_denom_term=dA_w * c_out_n,
           angular_numer_upstream=dA_w * c_in_n * psi_angle[:, i],
       )
       psi_face = result.outgoing_spatial_flux  # may be None for cylindrical degenerate
       # … and the WALK advances the half-angle thread, through the owner.
       psi_angle[:, i] = closure.advance_psi_half(
           result.cell_average_flux, psi_angle[:, i], ordinate=n,
       )

The cell-update strategy is chosen at
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` construction — the hub
is the **active-choice site**, and it keeps the generator so that every
consumer of one save state (the S\ :sub:`N` sweep and DSA alike) is
handed the same object.  It reaches the walk through the posed operator,
which is why the block above reads ``self.spatial_closure`` rather than
``sn_mesh.scheme``.  The hub realizes the strategy on its ``scheme``
attribute in its
constructor (introduced in this round as a constructor argument with
default
:class:`~orpheus.transport.spatial.diamond.DiamondDifference`).  The
default reproduces the inlined sweep math bit-identically — every
regression snapshot at ``tests/sn/regression/snapshots/`` was
generated with DD and continues to match bit-for-bit when the
unified sweep dispatches via ``scheme.update(...)``.  See
:ref:`cell-update-strategies` for the strategy contract and
:class:`~orpheus.transport.spatial.diamond.DiamondDifference` for the
DD scalar form.

:class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`
has since landed on exactly this dispatch: a user selects it today by
passing ``scheme=LinearDiscontinuous()`` at
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` construction
(``augmented_mesh.py`` records the same recipe at the ``self.scheme``
default).  ``Step`` and ``ExponentialCharacteristic`` remain
**reserved, not yet implemented** — literals rather than ``:class:``
roles for that reason — and the unified dispatch infrastructure is in
place to receive them.

The 1-D cumprod fast path (DD-only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Cartesian dispatch checks three preconditions before
selecting the 1-D cumprod fast path:

#. ``scheme is DiamondDifference`` — the cumprod recurrence
   :eq:`dd-recurrence` is a DD-specific algebraic identity
   (Lewis & Miller §5.3); LD / EC / Step closures do not admit
   the same recurrence.
#. Quadrature is GL1D (``ny == 1`` and all ``mu_y`` vanish).
#. Source is isotropic (``Q_aniso is None``).

If any precondition fails, the Cartesian dispatch routes
through the 2-D wavefront sweep (which handles 1-D as a
special case).  Preserving the cumprod fast path inside the
unified algorithm is required to keep the 1-D regression
snapshots bit-identical and to retain the historical
sub-millisecond sweep time for typical 1-D problems.

The 2-D wavefront sweep dispatches its DD per-cell algebra
through the storage-free kernel pair
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.cell_kernel_batch`
/ :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.residual_kernel_batch`
on the strategy attached to the
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` (Wave 2 of the SN
performance plan; closes Issue #4 — see
:ref:`sweep-octant-dependency-graph` for the full architecture and
:ref:`sweep-dispatch-relayering` for the S6.4(e) re-layering).
The "inlined DD math" formerly carried inside
:func:`~orpheus.sn.loss_representation._sweep_jacobi` was lifted into
:class:`~orpheus.transport.spatial.diamond.DiamondDifference` as a single
bit-identical extraction, vectorised across the
``(N_oct, n_diag, ng)`` slice — the ordinate axis, anti-diagonal
axis, and group axis simultaneously.  Wave C-extension's LD / EC
/ Step closures override the kernel pair and become drop-in
alternatives at SNMesh construction time:
``SNMesh(mesh, quad, scheme=LinearDiscontinuous())``.  The
open design point of "how to parameterise the 2-D wavefront
without breaking anti-diagonal vectorisation" is now closed: the
storage walk is the scheduler, the level operation owns the
direction fork, and the kernel pair is the closure — the contract
is per-level batched evaluation.


.. _sn-gotchas:

Gotchas
=======

Each gotcha is a **consequence → how it manifests → which test / level
catches it** — a trap that hides a solver bug behind a green test.  They
should *shrink* over time as the code hardens.

.. _sn-symptom-table:

Where to look first — symptom → chapter
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 32 30

   * - Symptom
     - First suspect
     - Go to
   * - :math:`k` wrong on a vacuum-bounded problem (overshoot
       :math:`\approx L/A`)
     - the reported-:math:`k` functional omitting leakage (the #291
       class)
     - :ref:`sn-keff-estimator`
   * - :math:`k` wrong only when :math:`(n,2n)` is present
     - the emission mis-posed as production (the R7 fork)
     - :ref:`sn-keff-estimator`
   * - :math:`k` right on 1-group / reflective, wrong on multigroup
       heterogeneous
     - scattering-matrix orientation drift — a 1-group green proves
       nothing (degeneracy trap below)
     - :ref:`scattering-matrix-convention`, :doc:`slab_multigroup`
   * - flux spike at :math:`r = 0` on a curvilinear fixed source
     - a missing :math:`\Delta A/w` geometry factor
     - :ref:`balance-curvilinear`
   * - NaN / overflow marching through the angular sweep
     - a negative :math:`\alpha`-dome entry (warning below)
     - :doc:`curvilinear_one_group`
   * - negative or oscillating flux on coarse Cartesian cells
     - the DD closure's unboundedness — refine or change closure
     - :doc:`/theory/foundations/discretization`,
       :ref:`ld-ubld-multidim`
   * - SI iteration count blows up as :math:`c \to 1`
     - :math:`\rho = c` physics, not a bug (acceleration is the fix)
     - :ref:`si-within-group-splitting`
   * - sweep and matvec disagree
     - forbidden post-#206 (one walk) — a representation-seam
       regression
     - :ref:`loss-rep-one-walk-one-instance`
   * - Krylov stalls or diverges after a composite-sizing refactor
     - ``restart`` / ``n_dof`` sizing (the ERR-053 family)
     - :ref:`sn-direct-seed-gotchas`
   * - MMS recovers a lower order than theory
     - the ansatz nulls the term (Mode 7) or the regime is degenerate
     - :doc:`/theory/verification/sn`
   * - the adjoint reciprocity gate reds
     - the three-transposes landmine (Euclidean / Hilbert / walk)
     - :ref:`sn-adjoint`

Degeneracy traps — a passing test that proves nothing
-----------------------------------------------------

.. admonition:: Gotcha — 1-group eigenvalue tests are degenerate
   :class: warning

   :math:`k = \nu\Sigma_f / \Sigma_a` is **flux-shape independent**: a
   1-group eigenvalue is a material-property ratio computable *without*
   solving the transport equation, so it cannot detect any error in the
   spatial, angular, or scattering operators.  **Any verification claim
   needs** :math:`\geq 2` **groups** (``vv-principles`` anti-pattern #3).  A
   1-group eigenvalue is still fine for a *rate* or *convergence-order*
   claim — declare the claim layer.

.. _sn-homogeneous-degeneracy-gotcha:

.. admonition:: Gotcha — homogeneous / uniform-rescale invariance hides coefficient bugs
   :class: warning

   Any eigenvalue problem whose target is the flux :math:`\phi` is invariant
   under a uniform rescale :math:`\phi \to C\phi` (the factor :math:`C`
   cancels in the Rayleigh quotient
   :math:`k = \nu\Sigma_f\,\phi / \Sigma_a\,\phi`).  Homogeneous and
   same-material multi-region problems have a **spatially-uniform** rescale,
   so they are blind to factor-of-two coefficient errors that preserve the
   flux shape — and, in curvilinear geometry, blind to redistribution bugs
   (flat flux → the :math:`\alpha` terms vanish identically).  Only a
   genuine **material interface** makes the rescale factor :math:`C(x)`
   position-dependent and breaks the cancellation.

   This is exactly how **ERR-025** hid.  A missing :math:`1/W` normalisation
   in the 1-D diamond-difference recurrence halved the per-ordinate flux, but
   for Gauss–Legendre :math:`W = \sum_n w_n = 2` the missing factor rescaled
   it back — so every homogeneous test passed at machine precision while the
   heterogeneous eigenvalue was :math:`\sim 1.5\,\%` wrong and did **not**
   converge away under mesh or :math:`S_N`-order refinement (the gap
   plateaued in angle).

   **Catcher:** at least one *absolute*-:math:`\phi` test — the fixed-source
   flat-flux diagnostic (:math:`Q/\Sigma_t`), or an absolute eigenvalue
   comparison against a structurally-independent heterogeneous reference.
   The live pins are the L0 symbolic-recurrence check
   :func:`tests.sn.sweep.slab.test_dd_recurrence.test_dd_per_cell_recurrence_matches_symbolic_derivation`
   and the L1 heterogeneous absolute-:math:`k` regression
   :func:`tests.sn.eigenvalue.test_keff_slab.test_heterogeneous_absolute_keff`
   (a 2-region A+B reflective slab pinned against the Case
   singular-eigenfunction reference; the pre-fix :math:`1.48\times10^{-2}`
   error fails it by two orders of magnitude).

.. admonition:: Gotcha — conservation holds even with wrong per-ordinate balance
   :class: warning

   Global particle balance **telescopes** by construction, so a *scalar*
   balance sum can hold to machine precision while the *per-ordinate*
   flat-flux residual is wrong (``vv-principles`` anti-pattern #8; the
   identity :math:`\sum_n w_n(\alpha_{n+1/2} - \alpha_{n-1/2}) = 0`
   annihilates per-ordinate redistribution errors that cancel in the sum).
   The load-bearing invariant is the **per-ordinate** flat-flux residual
   (= 0), not the telescoped scalar balance.

Curvilinear redistribution
--------------------------

.. admonition:: Gotcha — the α redistribution dome must stay non-negative
   :class: warning

   The curvilinear :math:`\alpha` dome must be non-negative; a negative entry
   drives NaN / overflow through the angular sweep.  The fixed-source
   flat-flux diagnostic (:math:`Q/\Sigma_t`) is the single most powerful
   curvilinear bug detector — a spike at :math:`r = 0` localises a missing
   :math:`\Delta A / w` geometry factor.

Solver-coordination traps
-------------------------

* **Renormalise-then-report ordering.**
  :func:`~orpheus.numerics.eigenvalue.power_iteration` renormalises
  :math:`\phi` to unit production **between**
  :meth:`~orpheus.sn.solver.SNSolver.solve_fixed_source` and
  :meth:`~orpheus.sn.solver.SNSolver.compute_keff`.  So
  ``compute_keff`` sees the *renormalised* :math:`\phi`, while the stored
  ``_inner.iterate.boundary`` is the *un-renormalised* trace — the scale
  bridge (:ref:`sn-keff-estimator`) is what makes the leakage term
  consistent across that boundary.  Reordering the two (report before
  renormalise) would break the bridge's ``1.0`` shortcut.
* **The outer iterate must stay a bare** ``np.ndarray``.  The Mode-11
  live-arm sentinel in
  ``tests/sn/operators/test_fission_kernel_crosscheck.py`` proves that
  ``power_iteration`` feeds a **bare** :class:`numpy.ndarray` flux to
  :meth:`~orpheus.sn.solver.SNSolver.compute_fission_source`, so the
  bare-``np.ndarray`` dispatch arm of
  :meth:`IsotropicFission.apply
  <orpheus.transport.operators.isotropic_transfer.IsotropicFission.apply>`
  — the fission **energy** binding the k-outer has held since CS4c step 4
  — is the *live production arm* (the sentinel wraps that leaf in-process
  and asserts the counter advances).  The *angular* binding
  ``FissionOperator`` refuses a bare scalar carrier outright, which is
  the same contract stated as a type instead of as a sentinel.  The estimator's
  :class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`
  evaluations read the same bare array.  Routing the outer iterate
  through a typed carrier would dark the arm (redding the sentinel) and
  break the estimator's evaluate path — the bare-array outer iterate is
  a load-bearing contract, not an implementation accident.

.. seealso::

   **Sweep-machinery gotchas** — the Krylov ``restart`` sizing bug
   (ERR-053 family), the product-cylinder edge-extrapolation data-flow
   invariant, and the Mode-12 / ERR-067 :math:`G`-reciprocity metric catch
   — are documented alongside the sweep at :ref:`sn-direct-seed-gotchas`.


.. _sn-chapters:

Chapters in this sub-book
=========================

This page is the S\ :sub:`N` sub-book's index: the synopsis, the
architecture map, the transport equation, and the shared cell-update
and dispatch contracts; the chapter decomposition is tracked as issue
`#231 <https://github.com/deOliveira-R/ORPHEUS/issues/231>`_.

Several orders through the book serve different jobs (tracks, not one
sequence):

* **Newcomer** — :doc:`placement` first (*why* discrete ordinates —
  the trade-space against CP/MoC/P\ :sub:`N`/diffusion/MC), then the
  broadening progression in toctree order:
  :doc:`slab_one_group` (the whole machine at its simplest) →
  :doc:`slab_multigroup` (energy and the eigenvalue) →
  :doc:`cartesian_multid` (space) → :doc:`curvilinear_one_group` →
  :doc:`curvilinear_multigroup`, with :doc:`angular_quadrature` and
  :doc:`boundary_conditions` as on-ramp references.
* **Modifying the sweep** — :doc:`loss_representation` (the
  representation catalog and the one-walk theorems,
  :ref:`loss-rep-one-walk-one-instance`) → the strategy contract on
  this page (:ref:`cell-update-strategies`) → the multi-D schedule
  (:ref:`sweep-octant-dependency-graph`) → the curvilinear sequential
  sweep (:doc:`curvilinear_one_group`) → the sweep-machinery gotchas
  (:ref:`sn-direct-seed-gotchas`).
* **Porting an equation from the literature** — the machine header at
  the top of this page (sign / normalization / layout / ordering
  conventions) → :doc:`angular_quadrature` (weight normalization) →
  the scattering-matrix convention
  (:ref:`scattering-matrix-convention`) and :ref:`pn-scattering` →
  :doc:`/theory/verification/sn` for the gate the new equation ships with.
* **Debugging a wrong answer** — start at the symptom table
  (:ref:`sn-symptom-table`), then the gotcha catalog below it.

The chapters:

.. toctree::
   :maxdepth: 2

   placement
   slab_one_group
   slab_multigroup
   cartesian_multid
   curvilinear_one_group
   curvilinear_multigroup
   curvilinear_numerics
   angular_quadrature
   loss_representation
   boundary_conditions
   solver
   acceleration
   adjoint
   history
