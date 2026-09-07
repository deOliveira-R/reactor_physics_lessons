.. _boundary-conditions:

Boundary Conditions
===================

Infrastructure
--------------

Boundary conditions are declared on the **geometry mesh** and resolved
by the **solver's augmented mesh** at construction time.  This two-stage
design separates physics intent (what condition to apply) from solver
mechanics (how to enforce it in the :term:`sweep`).

**Stage 1 --- Geometry declaration.**
:class:`~geometry.mesh.Mesh1D` carries ``bc_left: BC | None`` and
``bc_right: BC | None``; :class:`~geometry.mesh.Mesh2D` carries
``bc_xmin``/``bc_xmax``/``bc_ymin``/``bc_ymax: BC | None``.
:class:`~geometry.mesh.BC` is a frozen dataclass with two fields:

- ``kind: str`` --- an identifier such as ``"vacuum"``, ``"reflective"``,
  or ``"white"``.
- ``params: dict[str, float]`` --- optional numeric parameters
  (e.g. ``{"albedo": 0.7}``).

Convenience instances are available for the common cases:
:attr:`BC.vacuum <orpheus.geometry.mesh.BC.vacuum>`,
:attr:`BC.reflective <orpheus.geometry.mesh.BC.reflective>`, and
:attr:`BC.white <orpheus.geometry.mesh.BC.white>`.
When a face is left as ``None``, the solver applies its own default
(reflective for the SN solver, matching the infinite-lattice /
eigenvalue convention).

**Stage 2 --- Solver resolution via the BC realizer.**
:class:`SNMesh` owns a class-level
:attr:`~SNMesh.BOUNDARY_OPERATOR_REGISTRY` mapping kind strings to
:class:`~orpheus.geometry.boundary.BoundaryTraceLaw` **subclasses**
(post Wave 8 of the trace-law refactor in
``.claude/plans/transient-giggling-cake.md``)::

    BOUNDARY_OPERATOR_REGISTRY = {
        "vacuum":     VacuumInflow,
        "reflective": ReflectiveBoundary,
    }

The registry values are the law classes themselves, not factory
functions. The pre-refactor ``_sn_vacuum_boundary_operator`` /
``_sn_reflective_boundary_operator`` factories were retired; their
job is now done by :meth:`SNMesh.realize_boundary_law <orpheus.sn.mesh.augmented_mesh.SNMesh.realize_boundary_law>`, which dispatches
through :class:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer`
**uniformly** for every supported mesh (1-D Cartesian, 1-D
spherical, 1-D cylindrical, 2-D Cartesian) — see
:ref:`bc-sn-resolution-table` below. Issue #188 (curvilinear support
in the trace space — then named ``InflowTraceSpace``, now the unified
:class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`) and Issue
#176 (drop 2-arg ``apply`` + simplify shim) collapsed the pre-cleanup
Cartesian-vs-curvilinear bypass into a single realizer-routed
path; details at :ref:`bc-curvilinear-realizer-unification`.

During ``SNMesh.__init__``, each face's :class:`~geometry.mesh.BC`
is looked up in the registry.  If the kind is not found, a
``ValueError`` lists the supported kinds.  For curvilinear
geometries (spherical, cylindrical), only ``"reflective"`` and
``"vacuum"`` are currently supported --- requesting any other kind
on a curvilinear mesh raises ``ValueError``.

The two-arg legacy interface ``bc.apply(angular_flux_outgoing,
quadrature)`` is **retired entirely** post Issue #186 / B3 + β2
(2026-05-11). Concrete BC :meth:`apply` methods no longer exist;
:class:`~orpheus.geometry.boundary.BoundaryTraceLaw` is a **pure
descriptor** with no callable interface. Rank-N composition uses
the descriptor-tree algebra
(:class:`~orpheus.geometry.boundary.LawSum` /
:class:`~orpheus.geometry.boundary.LawScaled`) with
:func:`~orpheus.geometry.boundary.realize_recursively` as the
sole descriptor→operator type transformer. See
:ref:`bc-trace-law-descriptor-model` for the design rationale.

The resolved BCs at ``sn_mesh.bc["xmin"]`` etc. expose the uniform
1-arg contract through the
:class:`~orpheus.geometry.boundary._bound_compat._BoundBoundaryOperator`
shim — internal to the package, not in
:attr:`orpheus.geometry.boundary.__all__`. Post Issue #186 the
shim is a **strict 1-arg passthrough** (extra args raise
:class:`TypeError`); since campaign phase **B2.0** it carries the
originating **law** itself, so a resolved BC can be asked what its law
*does* (``bc[face].law.geometry_map``,
``bc[face].law.response_kernel``) and not merely what it was declared
as. Its ``kind`` tag now reads that law's registry key, so the
``sn_mesh.bc["xmin"] == "vacuum"`` diagnostic
comparison continues to evaluate True iff the underlying law is
:class:`VacuumInflow`. (C4 / #220 re-keyed this surface from the
per-attribute ``sn_mesh.bc_left`` to the face-name-keyed
:attr:`SNMesh.bc` dict — see :ref:`bc-face-name-carve`.)
See :ref:`bc-tensor-decompositions` below
for the operator-algebra view and
:ref:`theory-boundary-conditions` for the full trace-law /
realizer architecture.

**Backward compatibility.**
:func:`solve_sn_fixed_source` still accepts a ``boundary_condition: str``
parameter (default ``"vacuum"``).  Internally it calls
``_apply_default_bcs(mesh, boundary_condition)``, which applies the
string to **all faces** that lack explicit :class:`~geometry.mesh.BC`
declarations.  When the mesh already carries explicit BCs, the parameter
is silently ignored --- mesh-level declarations always take precedence.
:func:`solve_sn` (the eigenvalue entry point) does not expose a
``boundary_condition`` parameter; eigenvalue problems use whatever the
mesh declares (defaulting to reflective on all faces).

.. note::

   Before this infrastructure existed, the SN solver hardcoded
   reflective BCs on all faces and the then-production ``transport_sweep``
   entry accepted a ``boundary_condition: str`` parameter.  That parameter
   has been removed --- BCs now flow exclusively through the mesh → SNMesh
   resolution path described above.

Supported Types
---------------

**Reflective** (specular reflection).
At the outer boundary :math:`r = R` (or :math:`x = L`), the incoming
flux for :term:`ordinate` :math:`n` is set to the outgoing flux of its reflected
partner:

.. math::
   :label: reflective-bc

   \psi_n^{\rm in} = \psi_{n'}^{\rm out}

where :math:`n'` is the reflected partner ordinate (negating the
appropriate direction cosine).  The pairing is derived from the mirror
motion by each :term:`quadrature`'s
:meth:`~orpheus.numerics.quadrature.Quadrature.ordinate_permutation`
(:eq:`quadrature-ordinate-permutation`).  This is the
default for eigenvalue problems (infinite lattice / infinite medium).
The CP solver uses white (isotropic) BCs instead; see
:ref:`white-bc-quality` for a comparison showing the ~1% gap between
the two approaches.

.. warning::

   ⚠ **Two reflective axis pairs plus diamond differencing make the
   within-group loss operator EXACTLY SINGULAR** (#344).  A specular
   face is a *closed* boundary for the diamond closure's undamped face
   sawtooth, so with :math:`\ge 2` axes reflective at both ends
   :math:`A = L+C-S-N_{2n}-B` (:eq:`sn-within-group-with-n2n`) has a
   non-trivial kernel — the kernel is a *pure-trace* object, so the two
   bulk gains :math:`S` and :math:`N_{2n}` annihilate it and neither
   enters the count — ``[M]``
   :math:`\dim\ker A = 12` on a 2-D ``level_symmetric`` :math:`S_4`
   2-group box, :math:`138` at :math:`d=3` :math:`(3,4,5)` — and the
   returned boundary trace is one member of a solution manifold.  This
   is the DEFAULT for :func:`~orpheus.sn.solver.solve_sn`, which has no
   ``boundary_condition`` parameter, and for any bare
   :class:`SNMesh`.  Nothing a user normally checks reveals it — every
   mirror-even functional is blind by theorem — and the solver projects
   the trace onto the canonical member and says so.  ⚠ ``_apply_default_bcs``
   fills only when **all** faces are ``None``, so a *partial* declaration
   silently leaves the rest reflective.  Full treatment:
   :ref:`sn-loss-kernel-gauge` in :doc:`cartesian_multid`.

**Vacuum** (zero incoming flux).
All incoming angular fluxes at the face are set to zero:

.. math::
   :label: vacuum-bc

   \psi_n^{\rm in} = 0

.. (vv-status rationale) definition: Definitional / notation introduction. Operational rule ψ_n^in = 0 for vacuum boundary; semantics exercised by every vacuum-BC test (test_boundary_conditions, MMS suite); no isolated identity to verify.
.. vv-status: vacuum-bc documented


In the 1-D cumprod path, this means the recurrence starts from zero
instead of the reflected outgoing flux.  In the 2-D wavefront sweep,
the reflective-partner copy is skipped, leaving incoming-face angular
fluxes at their zero initialisation.  Vacuum BCs are the natural
choice for fixed-source MMS verification on finite slabs (see
:ref:`sn-mms-verification`).

.. _bc-tensor-decompositions:

Boundary conditions as tensor decompositions
---------------------------------------------

The boundary conditions used by :class:`SNMesh` are concrete instances
of a more general tensor-decomposed framing, defined in
:mod:`orpheus.geometry.boundary`. A boundary condition is a linear
operator :math:`B` mapping the outgoing angular flux at a face to the
incoming angular flux:

.. math::
   :label: bc-tensor-decomposition

   \psi_{\rm in}(\Omega)
   = (B\,\psi_{\rm out})(\Omega)
   = \sum_\alpha \bigl(G_\alpha\,\psi_{\rm out}\bigr)(\Omega) \cdot A_\alpha,

.. (vv-status rationale) Definitional/literature-transcribed framing (a BC
   as R = sum_alpha G_alpha A_alpha, Lewis & Miller 1984 §3.4). The concrete
   rank-N primitives (vacuum/reflective/white) carry their own verification.
.. vv-status: bc-tensor-decomposition documented

where :math:`G_\alpha` is a **per-term operator** (permutation,
pushforward, angular average, spatial wrap) and :math:`A_\alpha` is a
**scalar amplitude** (typically an :term:`albedo` :math:`\in [0, 1]`).

.. warning::

   **Two decompositions, and the letters collide.** This is the §15.2
   **rank-N expansion** :math:`B = \sum_\alpha G_\alpha \otimes
   A_\alpha` — a sum over *terms*. The affine trace law
   :eq:`affine-bc-form` is a **factorisation** of one term,
   :math:`\gamma_-\psi = R\,G\,\gamma_+\psi + q`, where

   * :math:`G` is the **deck transformation** — the composition
     operator of a measure-preserving bijection, and specifically one
     that is the deck transformation of an **actual quotient** of the
     domain. Multiplicativity is necessary but NOT sufficient: a
     specular *kernel* is a permutation, hence multiplicative, and is
     still constitutive (:ref:`bc-factor-roles`);
   * :math:`R` is the **constitutive response kernel** — an amplitude
     or an angular kernel.

   Whence **exactly one of** :math:`G`, :math:`R` **is non-trivial**,
   and **whichever one that is carries the crossing**
   :math:`\Gamma_+ \to \Gamma_-`: geometrically for a quotient law,
   and by the physics for a constitutive one, which has no ambient
   isometry to inherit it from. (This bullet pair typed :math:`G` as
   :math:`\Gamma_+ \to \Gamma_-` and :math:`R` as living "on
   :math:`\Gamma_-`" until 2026-08-04. That pairing is the
   **classifying** typing, not the realized one — see
   :ref:`bc-taxonomy-vs-factorization`; every realized response in the
   tree, including the two this page's table builds, is typed
   :math:`\Gamma_+ \to \Gamma_-`.)

   So the :math:`G_\alpha` of this section is **not** the :math:`G` of
   the affine form: it is the whole per-term map, whose honest name is
   the composite :math:`R \circ G` — never :math:`R` alone. Campaign
   phase **B3.0** fixed one consequence of the collision in this very
   table: the cosine-weighted hemispheric (Lambertian) average had
   been filed as a *geometric* operator, and an average is not
   multiplicative and not a bijection, so it is a **response kernel**
   (:class:`~orpheus.geometry.boundary.LambertianReemission`). See
   :ref:`bc-factor-roles` for the criterion and the rank-one theorem
   that explains why the misfiling had no observable consequence.

This is the same algebra Lewis & Miller (1984) §3.4 use to introduce
boundary conditions in transport: every BC of practical interest is
either rank-1 (one :math:`G_\alpha \otimes A_\alpha` term) or a finite
linear combination of rank-1 primitives (rank-N). The implemented
primitives are, with each law's affine factors alongside:

.. list-table:: Implemented :class:`~orpheus.geometry.boundary.BoundaryTraceLaw` primitives (Wave 7 vocabulary)
   :widths: 18 24 12 24 22
   :header-rows: 1

   * - Class
     - :math:`G_\alpha` (the per-term map :math:`R \circ G`)
     - :math:`A_\alpha`
     - affine factors :math:`(G,\;R)`
     - Rank / wired into ``solve_sn``
   * - :class:`~orpheus.geometry.boundary.VacuumInflow`
     - the zero map
     - 0
     - ``SelfPairedDeck.identity()``, ``ScalarResponse(0.0)``
     - 0 / yes
   * - :class:`~orpheus.geometry.boundary.ReflectiveBoundary`
     - permutation under reflection axis
     - albedo (1 = perfect)
     - ``SelfPairedDeck.mirror(axis)``, ``ScalarResponse(α)``
     - 1 / yes
   * - :class:`~orpheus.geometry.boundary.WhiteBoundary`
     - cosine-weighted hemispheric average — a **response**, not a
       geometry (B3.0)
     - albedo
     - ``SelfPairedDeck.identity()``, ``LambertianReemission(α, …)``
     - 1 / no (Wave C)
   * - :class:`~orpheus.geometry.boundary.PeriodicBoundary`
     - spatial wrap along ``axis``; the realizer derives the partner
       face from the installation face
     - 1
     - ``PairedDeck.wrap(axis)``, ``ScalarResponse(1.0)``
     - 1 / no (Wave C/D)
   * - :class:`~orpheus.geometry.boundary.AlbedoBoundary`
     - the **re-emission closure**'s pairing — specular or diffuse;
       with no closure, unstated (see below)
     - albedo
     - ``SelfPairedDeck.identity()`` always, and ``SpecularReemission(α, a)``
       / ``LambertianReemission(α, …)`` / ``ScalarResponse(α)`` by closure
     - 1 / no (building block)
   * - :class:`~orpheus.geometry.boundary.PrescribedInflow`
     - 0
     - 0
     - ``SelfPairedDeck.identity()``, ``ScalarResponse(0.0)``, plus
       :math:`q \in \Gamma_-`
     - 0 with :math:`q \neq 0` / not ``solve_sn`` (a k-eigenvalue
       problem has no external inflow), but **yes**
       ``solve_sn_fixed_source`` since campaign phase P2′ — the
       declared source is read by
       :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.from_mesh_laws`
       and delivered through the boundary-source channel
       (:ref:`bc-affine-source-channel`)

Note vacuum's and prescribed inflow's :math:`G`: it is the **identity
deck element**, not zero. The zero map is not a bijection, so it cannot
be a geometry map at all; the vanishing belongs entirely to :math:`R`.
Writing ":math:`R = G = 0`" spelled one fact twice, once in the wrong
tier — corrected at B3.0.

.. _bc-albedo-reemission-closure:

Albedo's re-emission closure — and why SN refuses the bare law
--------------------------------------------------------------

A surface's re-emission law is a product of two **independent** degrees
of freedom: *how much* of the arriving flux returns (:math:`\alpha`)
and *in what angular shape*. Every shape admits every amplitude, so
:class:`~orpheus.geometry.boundary.AlbedoBoundary` carries them as two
parameters rather than as a menu of their product:

.. code-block:: python

   AlbedoBoundary(0.7, SpecularReturn(axis="x"))                     # polished
   AlbedoBoundary(0.7, IsotropicReturn(axis="x", outward_sign=+1))   # matte
   AlbedoBoundary(0.7)                                               # unstated

The closure is **amplitude-free** by design: ``amplitude`` is a member
of the response-kernel Protocol that the diffusion realizer reads, so
the kernel must carry :math:`\alpha`; a closure carrying it too would
give one number two homes. The law instantiates the shape at its own
:attr:`~orpheus.geometry.boundary.AlbedoBoundary.albedo`.

**The third spelling is complete for one method and under-determined
for another** — the first bite of the
:ref:`angular-resolution axis <bc-method-realizability>`:

* A **scalar** trace has one boundary degree of freedom,
  :math:`J^- = \alpha J^+`. There is no angular distribution to fix, so
  the bare amplitude IS the whole law; the diffusion realizer reads it
  and stops, and ``BC("albedo", albedo=…)`` means exactly what it
  always meant.
* An **angular** trace resolves the full hemisphere. There
  :math:`R = \alpha\,I` is a map :math:`\Gamma_+ \to \Gamma_+`, and
  :math:`G = \mathrm{id}` supplies no crossing to :math:`\Gamma_-`.
  Nothing in the law says which outgoing direction feeds which incoming
  one, and composing it anyway inside :math:`\iota_-\circ\text{law}\circ\gamma_+`
  pairs incoming ordinate :math:`j` with outgoing ordinate :math:`j` —
  an artefact of **array position**, not geometry. The SN realizer
  therefore raises, naming the two completions, rather than choosing a
  default.

.. admonition:: Why the positional pairing is worse than meaningless
   :class: warning

   It is a **configuration-dependent accident**. Measured `[M]` on the
   ``xmax`` face, comparing the positional pairing's index map against
   the specular one:

   .. list-table::
      :header-rows: 1
      :widths: 40 30

      * - quadrature
        - positional :math:`=` specular?
      * - ``product(2, 4)``
        - **True**
      * - ``level_symmetric(6)``
        - **True**
      * - ``gauss_legendre(4)`` / ``(8)``
        - False (the slab mirror reverses order)
      * - ``lebedev(17)``
        - False

   So before the closure existed, a bare albedo behaved *exactly like a
   mirror* on two of the tree's quadratures and like nothing in
   particular on the others — silently, with no seam to notice. A user
   who validated on ``level_symmetric`` and ran production on
   ``lebedev`` got different physics from the same law object.

   This is the strongest form of the argument for refusing: the old
   answer was not a defensible default that the closure merely makes
   explicit. It was a coincidence of index order that *looked*
   defensible on the half of the fixture set where the coincidence
   held.

That is a refusal of an *incomplete spelling*, not of the law: both
completions are fully built, and each routes through the same
realization body as its geometry-tier twin —
``AlbedoBoundary(α, SpecularReturn(a))`` through
:class:`~orpheus.geometry.boundary.ReflectiveBoundary`'s, and
``AlbedoBoundary(α, IsotropicReturn(a, s))`` through
:class:`~orpheus.geometry.boundary.WhiteBoundary`'s.

.. note::

   **Where the pairing's invariants live.** The specular pairing
   :math:`\pi` (derived from the axis-mirror motion via
   ``ordinate_permutation``) carries three independent
   invariants — measure preservation (ERR-042), involution (ERR-044)
   and inflow :math:`\to` outflow (ERR-045). They were methods on
   :class:`~orpheus.geometry.boundary.ReflectiveBoundary`, correct
   while that was the only law standing on the pairing. With a specular
   closure available on albedo they moved to the pairing itself
   (``orpheus.geometry.boundary._specular``), so **both** carriers fire
   the same certification. Leaving them where they were would have
   meant a wrong table caught on one route and silently realized on the
   other.

The pre-Wave-7 names ``VacuumBoundaryOperator`` /
``SpecularBoundaryOperator`` / ``WhiteBoundaryOperator`` /
``PeriodicBoundaryOperator`` / ``AlbedoBoundaryOperator`` were
**retired in Wave O step O.4a.1**; their canonical successors
(:class:`~orpheus.geometry.boundary.VacuumInflow` /
:class:`~orpheus.geometry.boundary.ReflectiveBoundary` /
:class:`~orpheus.geometry.boundary.WhiteBoundary` /
:class:`~orpheus.geometry.boundary.PeriodicBoundary` /
:class:`~orpheus.geometry.boundary.AlbedoBoundary`) are the sole
live names in :mod:`orpheus.geometry.boundary`.
``MixedBoundaryOperator`` was **retired in Wave 11**; rank-N
(Marshak, partial-current) boundaries are now expressed via the
**descriptor-tree algebra**
(:class:`~orpheus.geometry.boundary.LawSum` /
:class:`~orpheus.geometry.boundary.LawScaled`) on the unrealised
laws:

.. code-block:: python

   tree = 0.3 * spec + 0.7 * white            # LawSum of LawScaled
   # The walker is method-blind: pass the method's own realizer.
   op = realize_recursively(tree, method_space, SNBoundaryRealizer())
   psi_in = op.apply(psi_out)                 # OperatorSum of ScaledOperator

See :ref:`bc-rank-n-algebra` for the closed algebra and the
:ref:`bc-realize-recursively` walker that lowers a descriptor
tree to a Wave-0 operator tree.

The abstract base :class:`~orpheus.geometry.boundary.BoundaryTraceLaw`
is a **pure descriptor** post Issue #186 / B3 + β2 (2026-05-11)
— it has **no** :meth:`apply` method. The :class:`LinearOperator`
inheritance that historically supplied ``apply`` was removed; the
concrete laws likewise carry no ``apply`` / ``apply_transpose``
methods. The §16A.3 three-layer architecture (descriptor /
realizer / operator) is now enforced by the type system: a static
type checker rejects ``law.apply(...)`` at the linter level
without running the program. The full retrospective on the
predecessor Option A and β1 forms (and why each was rejected) is
at :ref:`bc-trace-law-descriptor-model`.

The tensor framing pays off architecturally because partial-current
Marshak boundaries (:math:`B = c_1 \, B_{\rm refl} + c_2 \, B_{\rm
diff}` — a specular term plus a Lambertian term, Bell & Glasstone 1970
§1.5) and multi-region interface couplings are all instances of the
same algebra: pick the per-term maps, pick the amplitudes, sum. Each
term still *classifies* as :math:`R \circ G` internally (a taxonomy of
the term's content, not a recipe for evaluating it —
:ref:`bc-taxonomy-vs-factorization`), and what
distinguishes the two terms of a Marshak mix is their **response**: a
:class:`~orpheus.geometry.boundary.ScalarResponse` behind a mirror deck
element (:meth:`~orpheus.geometry.boundary.SelfPairedDeck.mirror`) for
the specular term, a
:class:`~orpheus.geometry.boundary.LambertianReemission` for
the diffuse one — whose rank-one structure makes its own :math:`G`
unobservable, which is why the diffuse term declares the identity deck
element rather than a second mirror. New BCs are one
:class:`BoundaryTraceLaw` subclass + one
``BOUNDARY_OPERATOR_REGISTRY`` entry away — no sweep edits per BC.

.. _bc-sn-resolution-table:

SN BC resolution table
----------------------

The :meth:`SNMesh.realize_boundary_law <orpheus.sn.mesh.augmented_mesh.SNMesh.realize_boundary_law>` dispatch is summarized below.
Each row maps the user-facing :class:`~orpheus.geometry.mesh.BC`
kind string to (a) the resolved :class:`BoundaryTraceLaw`
subclass and (b) the :class:`SNBoundaryRealizer.realize` output
operator. The realizer dispatch is **uniform** across every
supported mesh — 1-D Cartesian / spherical / cylindrical and 2-D
Cartesian — post Issue #188 + #176.

There is deliberately **no sweep-cycle column**. Whether a
configuration sweeps in one pass is a property of the whole face
configuration, not of the law on any single face — a reflecting
face opposite a vacuum is acyclic, two reflecting faces are not —
so it cannot be tabulated per row. The per-law
``creates_sweep_cycle`` ``ClassVar`` that once occupied this column
was retired 2026-07-30 for exactly that reason; the configuration-
level criterion, and the record of why the flag could not work, are
at :ref:`bc-sweep-cycle`.

.. list-table:: BC.kind → law class → realized SN operator
   :header-rows: 1
   :widths: 18 26 42 14

   * - ``BC.kind``
     - Law class
     - Realized SN operator
     - α
   * - ``"vacuum"`` — **narrowed** :math:`\Gamma_+ \to \Gamma_-`
     - :class:`~orpheus.geometry.boundary.VacuumInflow`
     - the **zero map**: a
       :class:`~orpheus.numerics.operator.ZeroOperator` whose two
       space hooks emit :math:`|\Gamma_-|` rows forward and
       :math:`|\Gamma_+|` rows on the transpose
     - —
   * - ``"reflective"`` — **narrowed** :math:`\Gamma_+ \to \Gamma_-`
     - :class:`~orpheus.geometry.boundary.ReflectiveBoundary`
     - ``PermutationOperator(local_perm) & IdentityOperator()`` on the
       **reduced** ordinate axis, with ``local_perm =
       Γ₊(f).to_local(π⁻¹[inflow])`` (the half-trace SPACE owns the
       local↔global remap — G6.5; a mirror is self-paired, so the domain
       face IS the installation face) — :math:`\pi` the mirror's derived
       ordinate permutation (:math:`\pi^{-1} = \pi` for a mirror)
     - 1 (fast path)
   * - ``"reflective"``
     -
     - ``α * <that TP>``
       (:class:`~orpheus.numerics.operator.ScaledOperator`)
     - α ≠ 1
   * - ``"white"`` — **narrowed** :math:`\Gamma_+ \to \Gamma_-` (B3.4a),
       **factored** (G6.3 step 3b)
     - :class:`~orpheus.geometry.boundary.WhiteBoundary`
     - ``(IsotropicEmissionOperator(...) @ PartialCurrentOperator(...))
       & IdentityOperator()`` — the Lambertian kernel as a two-link
       chain: :class:`~orpheus.sn.boundary.angular.PartialCurrentOperator`
       contracts :math:`\Gamma_+` to the outgoing partial current, and
       :class:`~orpheus.sn.boundary.angular.IsotropicEmissionOperator`
       re-emits it on :math:`\Gamma_-`. The law's
       declared ``axis`` / ``outward_sign`` is cross-checked against the
       installation face's :math:`\Gamma_+` before construction.
     - 1 (fast path)
   * - ``"white"``
     -
     - ``α * <that TP>``
     - α ≠ 1
   * - ``"periodic"`` — **narrowed** (B3.4c); arrow derived (G6.3 step 7)
     - :class:`~orpheus.geometry.boundary.PeriodicBoundary`
     - ``PermutationOperator(arange) & IdentityOperator()``, bound
       :math:`\Gamma_+(f') \to \Gamma_-(f)` and fed the PARTNER
       face's :math:`\Gamma_+`. The crossing lives in the channel
       (:meth:`PairedDeck.domain_face <orpheus.geometry.boundary.PairedDeck.domain_face>` names the partner; the composite
       supplies it); the angular factor is the ordinate permutation the
       wrap MOTION induces — the identity relabelling between two DISTINCT
       index sets, EARNED by the kernel's certified identification
       :math:`\Gamma_+(f') \equiv \Gamma_-(f)`, not assumed. (Until step 7
       this was an unbound ``IdentityOperator() & IdentityOperator()`` —
       the one link of the five that was not a typed arrow.)
     - 1
   * - ``"albedo"`` — **narrowed** (B3.4b); the realized body is chosen
       by the law's **re-emission closure**, not by its class
     - :class:`~orpheus.geometry.boundary.AlbedoBoundary`
     - with :class:`~orpheus.geometry.boundary.SpecularReturn`: the SAME
       narrowed permutation the reflective row above builds (one
       construction, not two agreeing transcriptions)
     - any
   * - ``"albedo"``
     -
     - with :class:`~orpheus.geometry.boundary.IsotropicReturn`: the SAME
       narrowed Lambertian the white row above builds
     - any
   * - ``"albedo"`` — closure-free spelling
     -
     - **REFUSED**. On an ANGULAR trace the law is under-determined:
       :math:`R = \alpha I` is an endomorphism of :math:`\Gamma_+` and
       :math:`G = \mathrm{Id}` supplies no crossing, so nothing says which
       outgoing direction feeds which incoming one. Until B3.4b this arm
       returned the full-face endomorphisms
       :class:`~orpheus.numerics.operator.ZeroOperator` /
       :class:`~orpheus.numerics.operator.IdentityOperator` /
       ``α·(I & I)``, which the composite then paired by ARRAY POSITION.
       A **scalar** method needs no closure, which is why the diffusion
       realizer takes the same object unchanged
     - n/a
   * - ``"prescribed_inflow"`` — the rank-0 **affine** law;
       **narrowed** :math:`\Gamma_+ \to \Gamma_-` (B3.4a),
       **collapsed** onto the zero morphism (P3)
     - :class:`~orpheus.geometry.boundary.PrescribedInflow`
     - the same :class:`~orpheus.numerics.operator.ZeroOperator` the
       vacuum row builds. This tier realizes the law's LINEAR factor
       :math:`L`, and for prescribed inflow :math:`L = 0`; the source
       :math:`q` travels the boundary-source channel
       (:ref:`bc-affine-source-channel`), assembled from the declared
       law by
       :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.from_mesh_laws`.
       Until **P3** this arm returned an ``IncomingSourceOperator``
       whose ``apply`` ignored the outgoing flux and asked the source
       spec to fill ``(|Γ₋|,) + psi_out.shape[1:]`` — affine, in a
       linear slot. The inflow **mask** had dissolved with the codomain
       at B3.4a, so :math:`q \in \Gamma_-` holds by TYPING rather than
       by an erasure (ERR-047).
     - —

.. note::

   **Since campaign phase B3.2 a realized SN law is typed**
   :math:`\Gamma_+ \to \Gamma_-` — it consumes the outflow half-trace
   and produces the inflow half-trace, and the consumer composes
   :math:`\iota_-\circ\text{law}\circ\gamma_+`. B3.2 landed it for
   ``vacuum`` and ``reflective``; **B3.4a** added ``white`` and
   ``prescribed_inflow``. What remains is ``albedo`` — at *every*
   :math:`\alpha`, since the :math:`\alpha = 0` and :math:`\alpha = 1`
   fast paths are endomorphisms too — and ``periodic``, whose
   :math:`G` must read the PARTNER face's :math:`\Gamma_+`. Those rows
   still emit full-\ :math:`N` endomorphisms, are **unreachable from
   this registry** (which admits only ``{vacuum, reflective}``), and
   are pinned by strict xfails until **B3.4b** / **B3.4c** land. A
   shape assertion cannot tell the two typings apart —
   :math:`|\Gamma_+| = |\Gamma_-|` on every quadrature × face in the
   tree — so read the *declared spaces*, never the output shape. Full
   derivation at :ref:`bc-domain-narrowing`.

The :meth:`SNMesh.realize_boundary_law <orpheus.sn.mesh.augmented_mesh.SNMesh.realize_boundary_law>` dispatch constructs the resolved
operator via :meth:`SNBoundaryRealizer.realize(law, method_space)`
where the ``method_space`` is built by
:meth:`SNMethodSpace.for_face` carrying the precomputed unified
:class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace` (built
once at :class:`SNMesh` construction for every supported mesh).
The reflective branch derives its reflection axis from the face's
own :class:`~orpheus.transport.mesh.axis.FaceLabel` —
``AXIS_NAMES[label.axis_index]`` — so the partner is correct at any
dimension by construction (C4 / #220; see
:ref:`bc-face-name-latent-d3-bug`). The
:class:`~orpheus.geometry.boundary._bound_compat._BoundBoundaryOperator`
shim pairs the result back with the law it was realized from; its
``kind`` tag reads that law's registry key, serving the
``sn_mesh.bc["xmin"] == "vacuum"`` string-equality surface.

.. note::

   **The 1-D y-face placeholders were retired in C4 / #220.** Pre-C4,
   a slab :class:`SNMesh` carried a pair of realized no-op
   ``ReflectiveBoundary(axis="y")`` operators at ``bc_ymin`` /
   ``bc_ymax`` so cross-dimensional code could read them without
   coord-system gating — but **no production code ever read them**
   (a 1-D mesh's ``trace.layout.faces`` is ``("xmin", "xmax")``).
   C4 makes them unrepresentable: a slab has no y-axis in its
   :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.axes` tuple, so
   :func:`~orpheus.transport.mesh.axis.face_labels` emits no y-label and
   :attr:`SNMesh.bc` has no y-entry — ``slab.bc["ymin"]`` is a
   :class:`KeyError`, not a no-op. See
   :ref:`bc-face-name-carve-what-retired` for the full retirement
   record (the pre-C4 "why the placeholders were once safe"
   rationale is preserved there).

**Pre-cleanup history.** Before Issue #188 + #176 (closed
2026-05-11), curvilinear ``Mesh1D`` bypassed the realizer because the
trace factory (then ``InflowTraceSpace.from_mesh_and_quadrature``, since
C5.3 the geometry-blind
:meth:`AngularTraceSpace.from_quadrature_and_layout
<orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.from_quadrature_and_layout>`)
raised :class:`NotImplementedError` on those coord systems; the
``_BoundBoundaryOperator`` shim carried a dual mode where the
``quadrature=`` kwarg, when non-``None``, bound an
``AngularQuadrature`` and forwarded ``inner.apply(psi,
bound_quad)`` to the legacy 2-arg :class:`BoundaryTraceLaw` body.
The bypass and dual-mode are gone; details and the algebraic
sequence ("Issue #188 unblocks Issue #176") at
:ref:`bc-curvilinear-realizer-unification`.

Inner Boundary (Curvilinear)
-----------------------------

At :math:`r = 0`:

- The face area :math:`A(0) = 0`, so **no spatial flux crosses the
  origin**.  The spatial incoming flux for outward-sweeping ordinates is
  zero.
- The **angular redistribution provides the inward-to-outward
  transition**: flux entering as an inward-directed ordinate
  (:math:`\mu < 0` or :math:`\eta < 0`) is redistributed to outward
  ordinates (:math:`\mu > 0` or :math:`\eta > 0`) through the
  :math:`\alpha` coupling.

This means the curvilinear sweep does not need an explicit boundary
condition at :math:`r = 0` --- the geometry handles it naturally.
Curvilinear sweeps currently only support reflective BCs on the outer
face; this is enforced by the validation in
:meth:`SNMesh.realize_boundary_law <orpheus.sn.mesh.augmented_mesh.SNMesh.realize_boundary_law>`.


