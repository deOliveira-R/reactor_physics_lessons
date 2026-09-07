.. _sn-curvilinear-numerics:

The curvilinear seed-strategy campaign (#168 → #282)
====================================================

This chapter is Part B's **campaign record** — the investigation
history behind the curvilinear machinery of
:doc:`curvilinear_one_group`.  It preserves, verbatim, the #168
Phases A, D, and F, the ERR-058 closeout (#195), and the #196
eigenvalue-equivalence verification: what was tried, why each attempt
fell short, and the diagnoses that narrowed a wrong-fixed-point family
to two closure seeds.  Phases B (the angular closure) and C (the
sweep-frame matvec) live with the production machinery in
:doc:`curvilinear_one_group`, and the terminal resolution — route (a)
(#282), which retired the whole ``PsiHalfAngleSeed`` strategy family
by making :math:`\psi_{1/2}` first-class marched state — is
:ref:`sn-direct-seed-solve`.

Read this chapter for the *why*: why anyone tried a Carlson inward
:term:`sweep`, an apply-vs-sweep twin audit, a Krylov default flip — and how
every gate stayed green while the fixed point was wrong (the blindness
analysis inside the ERR-058 closeout).  Present-tense design claims in
the preserved sections are historical; each section carries its
supersession banner.

.. note:: **Reading the period vocabulary and the period file paths.**

   Because the chapter preserves its sections verbatim, it names things
   as they were named at the time, and two of those names have since
   moved.  The angular-closure family was called the *pole* angular
   closure until 2026-08-28 (P4.9b): the family base is
   :class:`~orpheus.sn.angular.closure.AngularClosureBase` today, the
   hub attribute is ``angular_closure``, and the reason for the change
   is that a cylinder has no pole in the sense a sphere does — what
   matters is that one closure is *spatial* and one is *angular*
   (:ref:`sn-p49b-operator-poses-with-closures`).  Its module moved too:
   the file this chapter cites as ``orpheus/sn/sweep/pole_angular_closure.py``
   is :mod:`orpheus.sn.angular.closure`, and line numbers quoted against
   the old path are frozen at the commit that diagnosed them.  Genuine
   poles keep the word throughout — the sphere's polar cap, the
   :math:`\mu = -1` starting direction, and the Carlson *coupled-pole*
   seed are named correctly wherever they appear.

.. admonition:: Key Facts
   :class: tip

   * The terminal diagnosis (ERR-058, #195): **two independent closure
     seeds** in the curvilinear within-group operator — the angular
     half-angle thread seed and the spatial pole-face seed — were
     wrong on every non-flat field, and both were **flat-field-exact**,
     so every flat-flux gate stayed green while every non-flat fixed
     point was wrong.
   * Phase D put the Hébert §3.9.4 Eqs. (3.432)–(3.435) inward-sweep
     seed into the M-M angular recurrence on the apply path; Phase F
     backported it to the SI/sweep path — the same defect shipped
     twice, the structural twin-path pattern.
   * The Carlson proxy source :math:`\bar Q = \Sigma_t\phi_0/\sum w`
     was the dominant defect; ``AngularEdgeExtrapolation`` replaced
     it — and was itself the #282 walk-order back edge that route (a)
     finally removed.
   * #196 pinned **SI ≡ Krylov** as the permanent gate (bit-identical
     for fixed-source, iteration-floor-equivalent for eigenvalue); the
     curvilinear inner default returned to ``source_iteration`` on
     speed alone.

ERR-026 closure status (partial through Wave E)
===============================================

.. note:: **Superseded (2026-06-12, Issue #195).**

   This subsection records the Wave-E-era reading: ERR-026 PARTIAL,
   the curvilinear ``"krylov"`` default "would regress MMS to
   :math:`\mathcal{O}(h)`", the open second-order follow-up.  That
   reading was the best available *then* and is preserved as history.
   It is now **superseded**: ERR-058 (#195) showed the wrong fixed
   point was the *closure-seed* family, not a boundary-truncation
   order; the curvilinear default returned to ``"source_iteration"``
   (SI :math:`\equiv` Krylov bit-identical post-unification); and the
   isotropic MMS is :math:`\mathcal{O}(h^2)`-consistent.  See
   :ref:`sn-err-058-closure-seed-closeout` for the mechanism,
   structural obstruction, and production decision.  The numerical
   values below stay as bug-era evidence; their *interpretation* is
   carried by the close-out.

The curvilinear sweep's one-directional WDD closure
:math:`\psi_{n+1/2} = (\overline{\psi} - (1 - \tau_{mm})\,
\psi_{n-1/2})/\tau_{mm}` is preserved bit-identically by
:class:`DiamondDifference` (Wave C extracted it from the
inlined sweep verbatim).  ERR-026 (catalogued in
:doc:`/development` and the V&V matrix at
:doc:`/theory/verification/matrix`) lives in this closure: the
solver's source-iteration path converges to a non-flat
fixed point even though the matrix-free ``apply`` path with
the symmetric closure is exact for **constant** sources.

Wave D's gating contract was bit-identity for ``scheme =
DiamondDifference`` — the bug is preserved by construction so
the regression snapshots stay green.  Wave E (Issues #98 #99 #164)
took two passes at the closure:

* **Wave E Round 2** wired
  :func:`~orpheus.sn.solver.solve_sn_fixed_source` to route
  through Krylov-on-:meth:`StreamingCollisionOperator.apply` (the
  symmetric closure) with the sweep-as-``solve`` as preconditioner.
  This closes ERR-026 on **constant-source reflective-BC
  problems** — the canonical
  :file:`tests/sn/test_sweep_operator_inconsistency.py` regression
  suite confirms the krylov path gives the analytical flat flux
  to round-off where the sweep does not.
* **Wave E Round 3** (Issue #98 follow-up) closed the BC-faithfulness
  gap that Round 2 identified: the FD operator's
  then-``solution_to_angular_flux*`` codec and the
  matvec helpers consumed the
  :class:`~orpheus.geometry.boundary.BoundaryTraceLaw` instances on
  the :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` (Wave B Issue 7
  tensor-decomposed BC algebra), dispatching boundary fills via the
  realiser-routed 1-arg :meth:`apply` on the resolved
  :class:`~orpheus.numerics.operator.LinearOperator`. Vacuum,
  reflective, white, periodic, albedo, and mixed BCs are now
  plumbed uniformly through the FD operator; bit-identity to the
  pre-Round 3 hard-coded reflective fill is preserved for
  :class:`ReflectiveBoundary(axis=…, albedo=1.0)` (the standard
  ``BC.reflective`` case), which is the load-bearing condition for
  the 11 frozen regression snapshots to stay green. (Note:
  post Issue #186 / B3 + β2 the law itself is a pure descriptor;
  ``BoundaryTraceLaw.apply`` no longer exists. The realiser
  produces the 1-arg :class:`LinearOperator` whose :meth:`apply`
  the matvec calls; the Wave-E Round-3 prose describes the
  contract as it existed at the time, but the architectural
  conclusion — uniform BC consumption through a 1-arg
  ``apply`` — is the same.)

What is **still** open after Round 3: empirically the symmetric-
closure FD operator at the curvilinear outer face uses cell-center
as a face-flux approximation (``psi_right = fi[:, n, i, 0]`` at
``i = nx-1`` for outgoing :math:`\mu > 0`).  This is exact for
constant solutions but only first-order accurate on non-constant
solutions like the manufactured ``A(r) = sin(πr/R)`` ansatz used
by the curvilinear MMS test suite.  Switching the
``solve_sn_fixed_source`` curvilinear default from
``"source_iteration"`` to ``"krylov"`` would *regress* the MMS
convergence rate from the WDD sweep's
~:math:`\mathcal{O}(h^{1.3})` (ERR-026-affected, but a benign
volumetric-error mode for these MMS) to
~:math:`\mathcal{O}(h^{1})` (FD operator's boundary truncation).
Round 3 therefore *keeps* ``inner_solver="source_iteration"`` as
the default for all geometries; ``"krylov"`` is opt-in and
correct for constant-source problems but not the right default
for MMS.

The two ``xfail-strict`` tripwires at
``tests/sn/verification/mms/test_curvilinear_aniso_convergence.py``
remain ``xfail`` through Round 3 with updated reason strings
reflecting the partial closure.  Full ERR-026 closure on MMS
depends on a follow-up that extrapolates the curvilinear
outer-face flux at second order (DD diamond relation at the
boundary, or analogous ghost-cell technique).

Adams & Larsen 2002 §III.B's "preconditioner correctness vs
operator correctness" frame is the right lens: the sweep's WDD
fixed-point bias is the wrong answer for a *primary solve*, but
as a *preconditioner* the same fixed point is just an effective
scaling of the residual — it does not poison the converged
solution determined by the operator.  The operator must be
correct *and* second-order-accurate; Round 3 closed the
correctness piece (BC-faithfulness), the second-order piece is
the open follow-up.

.. _sn-boundary-face-flux-protocol:

Boundary face-flux strategies — Phase A (RETIRED in Phase C)
============================================================

Issue #168 empirical investigation (recorded at
``.claude/agent-memory/numerics-investigator/issue_168_three_defects.md``)
found **three independent O(1) boundary truncation defects** in the
historical curvilinear FD operator.  Phase A (2026-05-10) addressed
**Defects 1 + 2** via a ``BoundaryFaceFlux`` strategy Protocol — a
one-sided second-order DD diamond extrapolation
:math:`\psi^{\text{face}}_{N-1/2} = \tfrac{3}{2}\,\psi_{N-1} -
\tfrac{1}{2}\,\psi_{N-2}` plus a structural decoupling of
cell-centre storage from BC face-value storage in
the then-``solution_to_angular_flux_spherical`` codec
(returning a ``(fi, boundary_face_flux)`` tuple where ``fi`` was
pure cell-centre storage and the BC face flux lived in its own
companion array).

**Phase C retired the Phase A Protocol entirely.** The sweep-frame
apply matvec subsumes the boundary-face closure into the WDD
propagation chain — the BC trace law owns the boundary edge per
the §16A.3 contract, no separate algebraic extrapolation is needed.
The retired symbols are:

* ``orpheus.sn.spatial.boundary_face_flux.BoundaryFaceFlux`` (Protocol)
* ``orpheus.sn.spatial.boundary_face_flux.DDExtrapolation`` (default)
* ``orpheus.sn.spatial.boundary_face_flux.CellCenter`` (ablation)
* ``orpheus.sn.spatial.boundary_face_flux.BoundaryFaceFluxBase`` (ABC)
* The ``boundary_face_flux`` field on
  :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh`
* The 21 foundation tests at
  :file:`tests/sn/sweep/test_boundary_face_flux.py`

See :ref:`sn-sweep-frame-apply-matvec` for the replacement
architecture. The Phase A subsection is preserved as historical
context for the empirical-defects-investigation reasoning chain.

.. _sn-phase-d-carlson-coupled-pole-sweep:

Phase D Carlson coupled-pole sweep (Issue #168 Phase D)
=======================================================

.. attention:: **Superseded by Issue #282 route (a) (2026-07-04).**

   The swappable ``PsiHalfAngleSeed`` strategy family
   (``ZeroSeed`` / ``CarlsonInwardSweep`` / ``AngularEdgeExtrapolation``)
   whose design this section — and the Phase F and ERR-058 sections that
   follow — build up was **retired** by Issue #282 route (a).  The
   :term:`starting-direction <starting direction>` half-angle flux :math:`\psi_{1/2}` is now
   **first-class typed state** the sweep marches *directly* from the true
   within-group source, not a functional of the previous iterate.  Any
   "current default / retained strategy / the seed lives as a strategy
   field" claim in the three sections below is **historical** — read them
   for the *why* (what was tried and the diagnoses that narrowed the
   defect), but for the CURRENT design see
   :ref:`sn-direct-seed-solve`.  In particular the
   ``AngularEdgeExtrapolation`` "iterate extrapolation" seed those
   sections land on was itself the #282 walk-order back edge that route
   (a) removes.

.. admonition:: Key Facts
   :class: important

   * Phase D (commit landed 2026-05-12 on
     ``refactor/sn-operator-algebra``) closes the structural bug
     in :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
     by replacing the hardcoded ``psi_half_left = 0`` seed with
     the canonical Hébert §3.9.4 Eqs. (3.432)–(3.435) inward
     :math:`\mu = -1` sweep output.
   * The seed lives in the **M-M angular recurrence**
     (:func:`~orpheus.sn.angular.closure.compute_psi_half_per_level`
     in ``orpheus/sn/angular/closure.py``), **NOT**
     in the WDD spatial pole-face initial condition the
     :ref:`Phase C plan <sn-curvilinear-trajectory-resolvent-crosscheck-section>`
     proposed.  The diagnostic memo at
     ``.claude/agent-memory/numerics-investigator/phase_d_gate_1_1_sphere_mms_diagnosis.md``
     empirically falsified intervention ``[A]`` (WDD pole-face
     replacement) and confirmed intervention ``[B]`` (M-M
     half-angle seed replacement).
   * Architectural choice is **Option α (composition)**: the
     seed lives as a
     ``PsiHalfAngleSeed``
     strategy field on :class:`MorelMontryAngularSweep`, not as a
     sibling Protocol on :class:`SNMesh`. The Legacy / Bailey
     closures have no ``psi_half_left`` variable to seed; a
     sibling Protocol would force every consumer to handle an
     irrelevant Protocol.
   * The **L = 0 isotropic-only** assumption is load-bearing: the
     apply matvec's :math:`L` operator currently carries only
     :math:`\Sigma_t \psi` (scattering is composed externally
     via a separate operator).  A future refactor that moves
     scattering INTO :math:`L` MUST extend the moment-folded
     source in :eq:`hebert-3-432-source` to include
     :math:`\ell \ge 1` terms.

The Hébert §3.9.4 equations
----------------------------

Hébert §3.9.4 (pp. 141–144 of :cite:`Hebert2009`) opens the sphere
difference relations at Eq. (3.418) (angularly-integrated
divergence form), introduces the :math:`\alpha`-recursion
:eq:`alpha-dome-recursion` and the cell-balance with
redistribution divisor :math:`\Delta S_i / (2\,\mathcal{W}_n)` at
Eq. (3.428), and then specialises to the auxiliary starting
direction :math:`\mu = -1`.  At this direction the
angular-redistribution coefficient :math:`(1 - \mu^2)` is
**identically zero**, so the streaming–collision balance
decouples from the :math:`\alpha`-cascade and reduces to a plain
DD inward recurrence in radius.

The continuous form at :math:`\mu = -1` (Hébert Eq. (3.432)) is

.. math::
   :label: hebert-3-432

   -\frac{\partial}{\partial r}\,\phi_{-1/2}(r)
   \;+\; \Sigma(r)\,\phi_{-1/2}(r)
   \;=\; \sum_{\ell=0}^{L}
         \frac{2\ell + 1}{2}\,Q_\ell(r)\,P_\ell(-1).


.. implements:: hebert-3-432
   :by: orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator

   **Implemented by** 3 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: hebert-3-432
   :by: orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source

.. implements:: hebert-3-432
   :by: orpheus.sn.sweep.psi_half_angle_seed.radial_characteristic_forward_residual

The subscript :math:`-1/2` is Hébert's half-integer index for the
auxiliary starting ordinate — it labels the **inward
zero-weight** direction that sits one half-step above
:math:`\mu = -1` in the :math:`\alpha`-cascade, not a physical
:term:`ordinate` at :math:`\mu = -0.5`.  The right-hand side is the
Legendre expansion of the scattering source :math:`Q` evaluated
at :math:`\mu = -1`, where :math:`P_\ell(-1) = (-1)^\ell`.

For an **isotropic** operator (:math:`L = 0`, the current ORPHEUS
apply matvec) the source collapses to

.. math::
   :label: hebert-3-432-source

   \bar Q_i \;=\; \sum_\ell \frac{2\ell+1}{2}\,Q_\ell(r_i)\,(-1)^\ell
   \;\;\xrightarrow{L=0}\;\;
   \frac{1}{2}\,\Sigma_t(r_i)\,\phi_0(r_i),

where :math:`\phi_0` is the scalar-flux Legendre :math:`\ell = 0`
moment of the input :math:`\psi`.  The :math:`L = 0` collapse is the
*apply matvec's* reach (isotropic scattering); the **source** side is
NOT collapsed — Issue #282 route (a) folds **all** Legendre moments of
the true within-group source, because streaming manufactures angular
structure an isotropic flux does not have (an :math:`\ell = 0`-only
fold floored the anisotropic curvilinear MMS).  See
:ref:`sn-direct-seed-source-fold` for the full fold and the load-bearing
:math:`\ell = 1` term.

Discretising Eq. :eq:`hebert-3-432` on a sub-mesh of cell width
:math:`\Delta r_i` gives the DD cell-balance Hébert Eq. (3.433):

.. math::
   :label: hebert-3-433

   -\bigl(\bar\phi_{i+1/2} - \bar\phi_{i-1/2}\bigr)
   \;+\; \Delta r_i \cdot \Sigma_i \cdot \bar\phi_i
   \;=\; \Delta r_i \cdot \bar Q_i,

.. (vv-status rationale) Literature-transcribed derivation step: the Hébert
   Eq. (3.433) DD cell-balance for the μ=−1 starting-direction sweep — a
   verbatim reference definition en route to the Carlson seed.  Its terminal
   result (the seed producing the correct flat-flux / cold-start solution) is
   tested downstream; the hebert-3-43X family's optional explicit wiring is
   tracked on `Issue #194 <https://github.com/deOliveira-R/ORPHEUS/issues/194>`_
   (see :ref:`sn-phase-f-test-wiring`).
.. vv-status: hebert-3-433 documented

with Hébert's typographic conventions

.. math::

   \bar\phi_i \;\equiv\; \phi_{1/2,\,i}, \qquad
   \bar Q_i \;\equiv\; Q_{1/2,\,i}, \qquad
   \Delta r_i \;=\; r_{i+1/2} - r_{i-1/2}.

The negative sign on the streaming jump comes from
:math:`\mu = -1 < 0` — particles travel **inward**, so the
discrete jump is :math:`-(\phi_{i+1/2} - \phi_{i-1/2})`.
**Critically**, no :math:`\alpha`-redistribution divisor appears
in this balance because :math:`(1 - \mu^2) = 0` at the endpoint.
This is the entire reason Hébert can solve the :math:`\mu = -1`
sweep in closed form with a plain DD recurrence: the coupled
angular cascade is decoupled at the starting direction.

Combining the DD auxiliary relation
:math:`\phi_{n,i} = \frac{1}{2}(\phi_{n,i-1/2} + \phi_{n,i+1/2})`
specialised to the :math:`-1/2` ordinate with the balance and
solving for :math:`\bar\phi_i` in terms of the known
outgoing-face value :math:`\bar\phi_{i+1/2}` (further from the
centre — known because we sweep **inward** from the outer BC)
yields Hébert Eq. (3.434):

.. math::
   :label: hebert-3-434

   \bar\phi_i \;=\; \frac{\Delta r_i \cdot \bar Q_i
                            + 2 \cdot \bar\phi_{i+1/2}}
                          {\Delta r_i \cdot \Sigma_i + 2}.


.. implements:: hebert-3-434
   :by: orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source

   **Implemented by** 3 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: hebert-3-434
   :by: orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_transpose

.. implements:: hebert-3-434
   :by: orpheus.sn.sweep.psi_half_angle_seed.radial_characteristic_residual_march

Stepping inward to the next face uses the textbook DD auxiliary
relation rearranged (Hébert Eq. (3.435)):

.. math::
   :label: hebert-3-435

   \bar\phi_{i-1/2} \;=\; 2 \cdot \bar\phi_i - \bar\phi_{i+1/2}.


.. implements:: hebert-3-435
   :by: orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source

   **Implemented by** 6 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: hebert-3-435
   :by: orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_transpose

.. implements:: hebert-3-435
   :by: orpheus.sn.sweep.psi_half_angle_seed.radial_characteristic_residual_march

.. implements:: hebert-3-435
   :by: orpheus.transport.spatial.diamond._DD_W

.. implements:: hebert-3-435
   :by: orpheus.transport.spatial.scheme.DiscretizationSchemeBase.outgoing_face_from_average

.. implements:: hebert-3-435
   :by: orpheus.transport.spatial.scheme.DiscretizationSchemeBase.outgoing_face_from_average_transpose

The pair :eq:`hebert-3-434`–:eq:`hebert-3-435` IS the spatial
recurrence.  Together they realise a tridiagonal-style inward
sweep on the radial mesh: outer face :math:`\rightarrow` cell
centre :math:`\rightarrow` inner face :math:`\rightarrow` next
cell centre :math:`\rightarrow \ldots \rightarrow` pole face
:math:`\bar\phi_{1/2}` at :math:`r = 0`.

.. note::

   The three labels :eq:`hebert-3-432-source`,
   :eq:`hebert-3-434`, :eq:`hebert-3-435` are also declared in the
   :mod:`~orpheus.sn.sweep.psi_half_angle_seed` module docstring
   (the canonical algebra-of-record).  Each :math:`:label:` is
   unique across the documentation graph; the Sphinx page is the
   **presentation layer** for the equations the code module owns
   as source-of-truth.  The
   ``@pytest.mark.verifies("hebert-3-43X")`` wiring on the L0
   algebraic-identity tests in
   :file:`tests/sn/sweep/curvilinear/test_psi_half_angle_seed.py` is tracked
   at Issue #194; without that wiring the labels appear in the V&V
   audit as "documented but not tested" (orphan labels).

Why :math:`\mu = -1` is the natural starting direction
-------------------------------------------------------

The M-M angular closure on sphere is a per-cell
:math:`\alpha`-cascade that **couples** the :term:`angular flux` across
ordinates within one spatial cell: ordinate :math:`n` reads
:math:`\alpha_{n-1/2}` from the previous (more-inward-:math:`\mu`)
ordinate.  To start the cascade at the smallest-:math:`\mu`
ordinate, one needs a value for :math:`\alpha_{1/2}` AND for the
angular-edge flux :math:`\phi_{1/2,i}` at that seed half-integer.

The :math:`\alpha_{1/2} = 0` seed is **free**: it comes from
:math:`1 - \mu^2` evaluated at :math:`\mu = -1`, i.e.,
"The first value :math:`\alpha` is equal to :math:`1 - (-1)^2 = 0`"
(text below Hébert Eq. (3.422)).  That handles the *angular*
half of the problem.

The flux value :math:`\phi_{1/2,i}`, however, is NOT free.  It
is the **spatial flux profile** at the auxiliary starting
direction, and it must be solved for as a function of position
:math:`i` along the radial mesh.  Eqs. :eq:`hebert-3-432` through
:eq:`hebert-3-435` provide exactly that spatial solve.

At :math:`\mu = -1` the sphere streaming operator collapses to
pure radial divergence **without** the angular-redistribution
coupling.  As Hébert writes (p. 143):

   *"We observe that these directions correspond to particles
   entering the external surface and moving toward the central
   axis with* :math:`\mu = -1`. *The angular redistribution term
   vanishes on these points so that Eq. (3.164) simplifies to
   [Eq. (3.432)]."*

This is the **only** direction on the unit interval :math:`[-1, 1]`
where the spatial 1D-sphere problem reduces to a closed-form
linear recurrence in radius alone, without an inner angular
solve.  Picking any intermediate :math:`\mu` would leave the
coupling term active and re-introduce the cascade
chicken-and-egg.  See also
:ref:`sn-phase-d-pomraning-structural-singularity` for the
deeper structural reason :math:`\mu = \pm 1` is the only
admissible starting direction in any curvilinear geometry.

Why "zero-weight"
-----------------

In an :math:`N`-point Gauss–Legendre :term:`quadrature` on :math:`[-1, 1]`
the endpoints :math:`\mu = \pm 1` are **not** base points (the
polynomial is approximated by interior nodes only).  They have no
quadrature weight, hence "zero-weight" — the flux value at
:math:`\mu = -1` does NOT contribute to any
:math:`\sum_n \mathcal{W}_n \phi_n` integral that builds the scalar
flux moments.

The :math:`\mu = -1` ordinate is therefore a **purely auxiliary
numerical construct**: its flux values exist for the sole purpose
of seeding the :math:`\alpha`-cascade for the finite-weight
ordinates that follow.  After the cascade is initialised, the
angular-edge values :math:`\bar\phi_{i\pm 1/2}` are discarded;
only the **cell-centred values** :math:`\bar\phi_i \equiv
\phi_{1/2,i}` are kept (Hébert, p. 143, between Eqs. (3.435) and
(3.436)).  Those cell-centred values feed the finite-weight
ordinates' cell-balance Eq. (3.436) via the
:math:`(\alpha_{n-1/2} + \alpha_{n+1/2})\,\phi_{n-1/2,i} /
(2\,\mathcal{W}_n)` redistribution term, with
:math:`\phi_{n-1/2,i} = \phi_{1/2,i}` at the first
finite-weight ordinate :math:`n = 1`.

The flat-:math:`\psi` algebraic verification trace
--------------------------------------------------

The Phase D hypothesis is: *for a flat angular flux*
:math:`\psi_{\text{cell}} = C` *across all cells, the
inward sweep returns* :math:`\bar\phi_{1/2} = C`.  The algebra
verifies this in closed form.

Take a homogeneous problem with constant :math:`\Sigma_t = \Sigma`
and source :math:`\bar Q_i` constructed so the consistent fixed
point is :math:`\bar\phi_i = C` everywhere.  Specialising
Eq. :eq:`hebert-3-432-source` to :math:`L = 0` and applying the
flat-:math:`\psi` ansatz gives the consistent source
:math:`\bar Q = \frac{1}{2} \Sigma \cdot 2C = \Sigma \cdot C` (the
:math:`\phi_0` integral over flat unit-:math:`\psi` against GL
weights summing to 2 returns :math:`2C`; lumped into the discrete
:math:`\bar Q_i = \Sigma \cdot C`).

Substituting into Eq. :eq:`hebert-3-434` with inductive hypothesis
:math:`\bar\phi_{i+1/2} = C`:

.. math::

   \bar\phi_i
   \;=\; \frac{\Delta r \cdot \Sigma \cdot C + 2C}
              {\Delta r \cdot \Sigma + 2}
   \;=\; C \cdot \frac{\Delta r \cdot \Sigma + 2}
                     {\Delta r \cdot \Sigma + 2}
   \;=\; C.

Eq. :eq:`hebert-3-435` then gives
:math:`\bar\phi_{i-1/2} = 2C - C = C`.  The recurrence is
self-similar: every face and cell value stays at :math:`C`.  Hence
:math:`\bar\phi_{1/2}(r = 0) = C` for flat :math:`\psi` on the
consistent flat source — **the hypothesis holds**.

This trace establishes the Phase D fix as a **closed-form
analytical reference** in the
:doc:`algebra-of-record </development>` State-1A pillar sense: the
identity :math:`(L \cdot \psi_{\text{flat}})_{n,i,g} = \Sigma_t
\cdot \psi_{n,i,g}` is verifiable by exact algebra on the discrete
operator, no numerical quadrature required.  The L0 foundation test
:func:`tests.sn.sweep.curvilinear.test_psi_half_angle_seed.TestCarlsonFlatPsiAlgebraicIdentity.test_carlson_flat_psi_identity_reflective`
pins this identity at machine precision (``rtol=1e-13``).

The corrected injection-point story
-----------------------------------

The single largest **architectural correction** of Phase D is
where the canonical inward-sweep output is injected.  The Phase D
plan (and the literature memo's §7 implementation note) routed
the inward-sweep result :math:`\bar\phi_i` into the **WDD
spatial pole-face initial condition** at the then-production
``transport_operator_matvec_spherical`` matvec's (since deleted)
``psi_face_in`` initialisation — the very same site the
:ref:`sn-curvilinear-trajectory-resolvent-crosscheck-section` discussion
identified as the Phase C Carlson seed location.

The numerics-investigator diagnostic
(:file:`tests/sn/diagnostics/gate_1_1_sphere_mms_failure.py`)
falsified that hypothesis empirically.  Four interventions tested
against the M-M failing configuration on the flat-:math:`\psi`
probe:

.. list-table:: Phase D injection-point intervention sweep (Σ_t = 0.5)
   :header-rows: 1
   :widths: 8 40 30 22

   * - Probe
     - What it changes
     - Site
     - max\|residual\|
   * - ``[A]``
     - Carlson seed for WDD ``psi_face_in``
     - ``operator.py:738``
     - **1.89e+01 FAIL** (unchanged)
   * - ``[B]``
     - Carlson seed for M-M half-angle ``ψ_{1/2,i}``
     - ``pole_angular_closure.py:411``
     - **1.78e-15 PASS**
   * - ``[C]``
     - BOTH ``[A]`` + ``[B]``
     - both
     - **1.78e-15 PASS** (no extra effect)
   * - ``[D]``
     - M-M half-angle ``ψ_{1/2,i}`` = cell-centre value
     - ``pole_angular_closure.py:411``
     - **1.78e-15 PASS** (degenerate)

Reading the table:

* ``[A]`` confirms the WDD spatial pole-face IC is **not** what's
  wrong.  The Phase C
  ``psi_face_in = fi[:, outgoing_mask, 0, 0]`` Lewis–Miller
  cell-centre seed is already structurally equivalent to the
  Carlson inward-sweep output **on flat ψ** — both equal
  :math:`\psi_{\text{cell}}[0]` in that limit.  Replacing the
  WDD seed changes nothing.
* ``[B]`` is the canonical Carlson intervention: feeding
  :math:`\bar\phi_i` into the M-M recurrence's ``psi_half_left``
  closes the residual to machine precision.
* ``[D]`` is the **falsification check**: on the flat-:math:`\psi`
  reflective probe ``[B]`` and ``[D]`` coincide because the
  inward sweep returns :math:`\bar\phi_i \equiv \psi_{\text{cell}}`
  exactly.  The probe **cannot distinguish** the two.

To prove the Carlson seed is canonical (not merely coincidentally
correct), the diagnostic includes a vacuum-BC structural
independence cross-check.  On a vacuum-BC probe the inward sweep
returns a non-trivial spatial profile

.. math::

   \bar\phi_i \;=\; (0.613, 0.572, 0.527, 0.478, 0.423, 0.362,
                     0.295, 0.220, 0.138, 0.048),

distinctly **not** equal to the cell-centred flat
:math:`\psi_{\text{cell}} = \mathbf{1}`.  The two seeds differ by
up to 0.95 in absolute value, and the resulting operator residuals
differ by max-abs 7.31 — the Carlson seed ``[B]`` is mathematically
distinct and quantitatively superior to the degenerate
broadcast-cell-centre seed ``[D]``.  This is the
**structural-independence evidence** that pins the Phase D fix as
canonical, not as a coincidental match on a degenerate probe.

The pinning test for this structural distinction is
:func:`tests.sn.sweep.curvilinear.test_psi_half_angle_seed.TestCarlsonFlatPsiAlgebraicIdentity.test_carlson_vacuum_BC_flat_source_nx_3`
— a vacuum-BC hand calculation on the Carlson inward sweep
(``rtol=1e-13``) whose values are distinct from the degenerate
broadcast-cell-centre seed.  Without this test a future
regression that replaced the Carlson sweep with a naive
broadcast-cell-centre would pass every flat-:math:`\psi`
reflective test silently.

The bug Phase B baked in
------------------------

The pre-Phase-D production code at
``orpheus/sn/sweep/pole_angular_closure.py:411`` carried the
hardcoded zero seed:

.. code-block:: python

   psi_half_left = np.zeros((ng, nx), dtype=psi_level.dtype)
   for m in range(M):
       tau_m = tau_level[m]
       psi_half_right = (
           psi_level[:, m, :] - (1.0 - tau_m) * psi_half_left
       ) / tau_m
       redist[:, m, :] = (
           dAw_level[:, m].reshape(1, nx)
           * (alpha_level[m + 1] * psi_half_right
              - alpha_level[m] * psi_half_left)
           / volume.reshape(1, nx)
       )
       psi_half_left = psi_half_right

The Phase B docstring justified the zero seed as: *"for the
forward apply matvec we adopt* :math:`\phi_{1/2,i} = 0`, *the
unique choice that makes the recursion's seed consistent with*
:math:`\alpha_{1/2} = 0` *and that the sweep converges to under
fixed-point iteration."*  This reasoning is wrong — the
:math:`\alpha_{1/2} \psi_{1/2}` product vanishes regardless of
:math:`\psi_{1/2}` because :math:`\alpha_{1/2} = 0`, but the seed
ALSO enters the **denominator-propagation chain**: every
subsequent half-angle face flux
:math:`\psi_{m+1/2,i,g}` depends on :math:`\psi_{m-1/2,i,g}`
recursively, and the chain inherits the seed through the M-M
weighting :math:`(1 - \tau_m)`.  Setting the seed to zero when
Hébert's structural form says :math:`\psi_{1/2,i,g} =
\bar\phi_{1/2,i}` (the inward-sweep output) is a **wrong term
initialisation** — Mode 3 in the
``vv-principles`` 6-failure-mode taxonomy (see ``error_catalog.rst``
ERR-026 entry).

How the wrong seed survived Phase B
------------------------------------

The zero seed survived Phase B's L1 flat-flux-identity test
(``tests/sn/l1_analytical/test_pole_closure_flat_flux_identity.py``,
deleted with the Legacy/BFF closures it compared — a path literal, not a
live file)
because that test compared the three closures (Legacy / BFF /
M-M) **against each other on flat ψ**, NOT against the closed-form
fixed-point identity :math:`L \cdot \psi = \Sigma_t \cdot \psi`.
All three closures collapse to the same wrong-but-internally-
consistent value on flat :math:`\psi`, so cross-comparison passes
while the absolute closed-form check would have caught it
immediately.

The cylindrical case ALSO carries the zero seed in production but
:ref:`Cylindrical Gate 1.1 <sn-phase-d-gate-1-1-empirical>`
**passes** empirically.  The mechanism is the **dead first-ordinate
seed** of the level-symmetric quadrature exercised here: the
first-swept ordinate's seed weight is zero
(:math:`c_{\rm in}[m_0]=(1-\tau)/\tau=0` at raw :math:`\tau=1`), so
the wrong ``psi_half_left = 0`` seed is annihilated at source per
level.  (This was originally read as per-:math:`\mu`-level
:math:`\alpha`-dome telescoping "cancelling" the seed; #280 Phase
2.5b corrected that — it is a dead weight, level-symmetric-only, and
**false for a product quadrature**, where the seed is a live
self-coupling and the cold solve was seed-lagged until the
direct-seed fold.  Both regimes are historical since Q5.6.3:
cylindrical ``SNMesh`` admission refuses every non-carrying rule, and
the fold was retired with its subjects — see
:ref:`sn-direct-seed-r12a`.)  The sphere cascade has no equivalent dead-seed
weight — a wrong seed propagates directly to a wrong fixed point.  Phase D's fix
updates the cylindrical path too for **structural alignment with
the canonical form** (architectural correctness), but cylindrical
behaviour is empirically a regression-stability check, not a new
PASS.

.. _sn-phase-d-pomraning-structural-singularity:

Pomraning structural-singularity cross-reference
------------------------------------------------

Pomraning (1989) :cite:`Pomraning1989` frames the curvilinear pole
problem as **geometric**: :math:`r = 0` is structurally singular
in any curvilinear streaming operator because the
angular-derivative coefficients in the streaming term (the
:math:`(1 - \mu^2)/r` factor in the sphere streaming operator)
contain :math:`1/r`.  At :math:`r = 0` the coefficient diverges;
the natural discretisation must somehow handle this.  In his words
(p. 339, right column):

   *"It was pointed out that if the bounding surface of the
   system is used as one of the coordinate surfaces and one
   considers a family of nonintersecting surfaces that starts
   with the bounding surface and progresses inward to fill the
   system, then these surfaces will eventually shrink to a
   surface with a zero area, namely a line or a point. ... A
   special case of this elliptical example is a sphere, where
   the innermost surface is simply a point.  Hence, in general
   there will exist points on the innermost surface where the
   coefficients of the angular derivatives in the streaming term
   are infinite, since these coefficients contain the reciprocal
   of the radii of curvature ... Prime examples of such singular
   points are found in the usual spherical and cylindrical
   geometry formulations where* :math:`1/r` *terms are extant and
   the attendant difficulties are well known, particularly in
   numerical treatments."*

The naive engineering response would be **extrapolation**: pick
:math:`\psi_{\text{face}}(r = 0)` by fitting a polynomial in
:math:`r` through nearby interior cells.  This is what an
incautious starting heuristic does; it is also what produces
the M-M wrong fixed point ERR-026 diagnoses.

The Carlson coupled-pole response is **canonical** because it
sidesteps the singularity entirely: at the auxiliary direction
:math:`\mu = -1` the singular :math:`(1 - \mu^2)/r` term is
**identically zero** (the numerator vanishes), so the spatial
sweep at this direction sees **no singularity at all**.  The
equation tells the discretisation what
:math:`\bar\phi_{1/2}(r = 0)` should be — there is no need to
guess.  The cost is that the :math:`\mu = -1` sweep must be
solved first, then its result used as the seed for the cascade
at finite-weight ordinates (where :math:`(1 - \mu^2) > 0` and
the singularity would otherwise be felt).  This is exactly the
price Pomraning warns about: "difficulties must be dealt with".
The Carlson construction deals with it by **exploiting** the
singularity's vanishing at :math:`\mu = \pm 1` rather than
trying to regularise it at intermediate :math:`\mu`.

Option α: composition over sibling Protocol
-------------------------------------------

The seed is **M-M-specific**: only
:class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
carries a ``psi_half_left`` variable to seed.  The Legacy and
Bailey closures don't have one — their half-angle face flux
evaluation collapses to cell-centre values unconditionally.  Two
architectures were considered:

* **Option α (composition, shipped)** — the seed strategy lives as
  a ``PsiHalfAngleSeed``
  field on
  :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`.
  The abstraction stays local to the closure that consumes it.

* **Option B (sibling Protocol on SNMesh, rejected)** — the seed
  would be a separate Protocol attribute on
  :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh`, applied by the matvec
  before calling the pole closure.  This would force every
  consumer (Legacy / BFF / M-M) to handle a Protocol that is a
  **no-op** for the non-M-M strategies, violating the
  single-responsibility principle and forcing unrelated tests to
  thread the Protocol through call signatures.

The
``CarlsonSweepContext``
dataclass bundles the four inputs the Carlson sweep needs that
are NOT in the
:class:`~orpheus.sn.angular.closure.AngularClosureBase`
strategy's ordinary per-cell call signature (``sigma_t``, ``dr``,
``mu_quad``, ``weights``, ``bc_outer_value``), keeping the
call-signature expansion to a single new optional keyword — a minimal
blast-radius extension that Legacy and Bailey closures ignore by
documented closure contract.

Linear-operator preservation
----------------------------

Both seed strategies — ``ZeroSeed`` and
``CarlsonInwardSweep`` — are **linear in the input** ``psi_cells``
(verified by the ``is_linear: ClassVar[bool] = True`` trait, pinned
by foundation tests).  Linearity is the load-bearing property:
the apply matvec must be a linear operator, otherwise the
operator-algebra operations of
:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator`
(apply, apply_transpose, dense matrix probing) break.  The
``CarlsonInwardSweep`` is linear because:

* The :math:`\phi_0` moment is a linear projection of input
  :math:`\psi` (Legendre integration is linear).
* :math:`\bar Q = \frac{1}{2} \Sigma_t \cdot \phi_0` is linear
  in :math:`\psi` (:math:`\Sigma_t` is constant).
* The recurrence Eqs. :eq:`hebert-3-434`–:eq:`hebert-3-435` is
  an affine function of :math:`(\bar Q, \bar\phi_{i+1/2})` with
  constant coefficients depending only on
  :math:`(\Sigma_t, \Delta r)`.
* The ``bc_outer_value`` is constructed in the matvec by applying
  the realised BC operator to the cell-centred outer-cell
  :math:`\psi`, then extracting the most-inward ordinate's value
  — both operations are linear in the input :math:`\psi`.

The foundation test
:func:`tests.sn.sweep.curvilinear.test_psi_half_angle_seed.TestSeedLinearity.test_carlson_inward_sweep_is_linear`
pins the linearity directly; the operator-level linearity gate
in :file:`tests/sn/test_streaming_operator.py` pins it transitively
at the matvec boundary (``rtol=1e-12`` — relaxed from the
pre-Phase-D ``rtol=1e-13`` to absorb ~10×ULP non-associativity
drift, justified by the three principled-relaxation criteria of
the ``vv-principles`` bit-identity-vs-principled-equivalence
framework).

The L = 0 isotropic-only limitation
------------------------------------

The current
``CarlsonInwardSweep``
evaluates only the :math:`\ell = 0` (isotropic) Legendre moment
when building the moment-folded source in
:eq:`hebert-3-432-source`.  This is **consistent with the apply
matvec's structure**: the
:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` apply matvec
carries only an isotropic collision term :math:`\Sigma_t \psi`;
anisotropic scattering (P\ :sub:`1`\ +) is composed externally via
a separate scattering operator, not included in :math:`L`.

.. warning::

   **The L = 0 isotropic-only assumption is load-bearing for the
   Phase D fix.**  If a future refactor moves scattering INTO
   :math:`L` (e.g., to enable a "monolithic" SN apply that
   includes within-group scattering), the Carlson seed becomes
   WRONG: the source at :math:`\mu = -1` (Eq. :eq:`hebert-3-432`)
   needs the full Legendre-moment sum

   .. math::

      \bar Q_i \;=\; \sum_\ell \frac{2\ell+1}{2}\,Q_\ell(r)\,(-1)^\ell,

   not just :math:`\Sigma_t \phi_0`.  This is a Mode-6
   convention-drift risk per the ``vv-principles`` skill (the
   definition-site assumption disagreeing with the usage-site
   intention).  A foundation test pinning the isotropic-only
   assumption (e.g., asserting the apply matvec does NOT couple
   to ``self_scattering``) would catch a future drift; in its
   absence, this WARNING block and the module docstring's
   matching admonition are the only safeguards.  Track the
   future-refactor case under a fresh GitHub issue when the
   monolithic apply work is scheduled.

.. _sn-phase-d-default-flips:

Default flips
-------------

Phase D ships **two default flips** that activate the full
canonical curvilinear closure path:

#. The ``pole_angular_closure`` constructor argument of
   :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` — realized onto the
   like-named instance attribute — had its default
   flipped from
   ``LegacyTauSymmetricInterpolation``
   to
   :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`.
   :class:`MorelMontryAngularSweep`'s own constructor default for
   ``psi_half_seed`` is
   ``CarlsonInwardSweep``,
   so the single :class:`SNMesh` flip activates the full Phase D
   fix (canonical M-M closure + canonical Carlson seed) without
   requiring downstream call sites to thread the new strategy
   explicitly.

#. :class:`~orpheus.sn.solver.SNSolver`'s ``inner_solver`` default
   flipped from ``"source_iteration"`` to ``"krylov"`` for
   **curvilinear geometries** (spherical, cylindrical); Cartesian
   stays at ``"source_iteration"``.  The rationale: the Phase D
   fix lives in the apply matvec, and the Krylov path is the one
   that uses
   :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply`.  The
   sweep path (``"source_iteration"``) uses the spatial WDD
   recurrence and is unaffected by the Phase D fix — leaving its
   ERR-026-affected curvilinear behaviour in place would be wrong
   for the production default.

   .. note:: **Reverted (2026-06-12, Issue #195).**

      This curvilinear ``"krylov"`` default was undone by the ERR-058
      fix.  The premise — that the sweep path was ERR-026-affected
      while Krylov-on-apply was not — held only because the sweep and
      matvec were *distinct* discrete systems at Phase D time.  After
      the Depth-B/Wave-T unification they are ONE system, and the
      ERR-058 closure-seed fix makes that system O(h²)-consistent: SI
      :math:`\equiv` Krylov bit-identical, SI :math:`\sim 10^2\times`
      faster.  The curvilinear default returned to
      ``"source_iteration"``.  See
      :ref:`sn-err-058-closure-seed-closeout`.

.. _sn-phase-d-gate-1-1-empirical:

Empirical Gate 1.1 outcome (Phase D — full 12-cell crosstab)
------------------------------------------------------------

The Phase D acceptance gate is Gate 1.1 on **all three** pole
closures across both curvilinear geometries and both :math:`\Sigma_t`
values.  The parametrised test
:func:`tests.sn.sweep.core.test_phase_c_gates.test_apply_curvilinear_per_ordinate_flat_flux_residual`
produces the 12-cell crosstab:

.. list-table:: Gate 1.1 outcome under Phase D Carlson seed (2026-05-12)
   :header-rows: 1
   :widths: 18 30 26 26

   * - Geometry
     - Pole closure
     - :math:`\Sigma_t = 0`
     - :math:`\Sigma_t = 0.5`
   * - Sphere
     - ``LegacyTauSymmetricInterpolation``
     - PASS
     - PASS
   * - Sphere
     - ``BaileyFlatFluxRedist``
     - PASS
     - PASS
   * - Sphere
     - ``MorelMontryAngularSweep``
     - **XPASS**
     - **XPASS**
   * - Cylinder
     - ``LegacyTauSymmetricInterpolation``
     - PASS
     - PASS
   * - Cylinder
     - ``BaileyFlatFluxRedist``
     - PASS
     - PASS
   * - Cylinder
     - ``MorelMontryAngularSweep``
     - **XPASS**
     - **XPASS**

All 12 cells PASS or XPASS.  The 4 XPASS cells under M-M closure
are the ERR-026 markers — they now flip from FAIL to XPASS on
xfail-strict=False, unblocking the marker-removal commit that
Phase D Step 5 will execute (deferred per the closeout memo's
acceptance gate item 6).

This is the load-bearing **empirical evidence** for the
ERR-026 identity-and-rate scope closure.  The asymmetry between
the Phase C (cylinder PASS / sphere FAIL) and Phase D
(both PASS) crosstabs is the diagnostic mark of the Phase D
intervention: the sphere case required the Carlson seed because
its single-cascade structure has no telescoping; the cylinder case
already passed under Phase C because — for the **level-symmetric**
quadrature exercised here — the first-swept ordinate's seed weight
is exactly zero (:math:`c_{\rm in}[m_0] = (1-\tau)/\tau = 0` at raw
:math:`\tau = 1`), so the zero-seed inconsistency was annihilated at
source (a **dead** seed), not "absorbed" by any telescoping of the
solve.

.. note::

   **Correction (#280 Phase 2.5b, 2026-07-05).**  An earlier reading
   of this crosstab attributed the cylinder's Phase-C pass to
   ":math:`\alpha`-dome telescoping absorbing the zero-seed
   inconsistency" and generalised it to "the cylinder solve is
   seed-insensitive / was already exact."  That is a **level-symmetric-
   only** artefact and is **false for a product quadrature**: there the
   starting direction coincides with the first-swept ordinate
   (:math:`\mu_{\rm start} \equiv \mu_{m_0}`, :math:`t = 0`, #229), so
   :math:`c_{\rm in}[m_0] \ne 0` and the seed is a **live per-ordinate
   self-coupling** that contributes :math:`O(1)` to the :math:`m_0`
   cell diagonal.  The product-cylinder cold ``(L+C).solve`` was in
   fact seed-**lagged** (cold error :math:`\approx 0.57`) until the
   #280 2.5b direct-seed fold folded that self-coupling into the
   :math:`m_0` diagonal (:math:`c_{\rm out} \to c_{\rm out} -
   c_{\rm in}`), making the cold solve a single-pass direct inverse.
   The augmented :math:`(L+C)` is block-lower-triangular because the
   seed contribution lands **on the block diagonal** (forward
   substitution resolves it) — *not* because the seed "telescopes
   away."  Distinct claim, still valid: the :math:`\alpha`-dome
   telescopes under the angular weight sum
   :math:`\sum_n w_n \psi_n`, which is why **scalar / balance** V&V
   gates are blind to a wrong per-ordinate seed (anti-pattern #8) —
   that blindness statement is unaffected by this correction.

   **Closure (Q5.6.3, ``1689faf4``).**  The fold this correction
   documents was itself retired: cylindrical ``SNMesh`` admission now
   refuses every non-carrying rule
   (:func:`~orpheus.sn.angular.closure.assert_carrying_quadrature`),
   so neither the dead-weight regime nor the live self-coupling the
   fold absorbed is constructible.  Every admitted cylinder level
   carries an independent seed resolved by route (a)'s forward
   substitution — the correction's *content* (which mechanism operated
   on which rule class, and the blindness statement) stands as the
   record of why the refused classes were refused.

.. _sn-phase-d-gate-1-5-capture-and-compare:

Gate 1.5 strengthening — capture-and-compare BC apply input
------------------------------------------------------------

Phase C's Gate 1.5
(:ref:`bc-trace-contract-respected-by-matvec`) was a "round-trip"
check: invoke ``bc.realize().apply(...)`` independently and
compare against the matvec's observable output.  Phase D
strengthens this to a **capture-and-compare** check that pins the
exact value the matvec passes into the BC trace law:

#. Patch ``sn_mesh.bc["xmax"].apply`` (the outer radial face —
   a sphere's ``"outer"`` endpoint renders as ``"xmax"`` since
   C4 / #220, see :ref:`bc-face-name-carve`) to capture every input
   array passed to it during one matvec call.
#. Independently reconstruct the WDD-propagated outflow trace via
   a reference implementation
   (:func:`tests.sn.sweep.core.test_phase_c_gates._outflow_at_boundary_for_sphere_from_bulk`).
#. Assert the captured BC apply input matches the reference to
   ``rtol=1e-14`` — exactly bit-equal up to FP non-associativity.

The strengthening matters because the Phase D matvec now calls
``bc["xmax"].apply`` **twice** per matvec:

#. **Phase D Carlson context call** — applied to cell-centred
   outer-cell :math:`\psi` to build ``bc_outer_value`` for the
   ``CarlsonSweepContext``.  See the BC companion section
   :ref:`bc-two-bc-applies-per-matvec`.
#. **Phase C BC trace law call** — applied to the WDD-propagated
   outflow face value at the boundary edge, per the
   :ref:`affine-bc-form` contract.

The capture-and-compare test
:func:`tests.sn.sweep.core.test_phase_c_gates.test_bc_trace_contract_capture_and_compare_sphere`
(parametrised over ``vacuum`` and ``reflective``) **locates the
Phase C call by shape and content matching**: of the two captured
inputs, the one whose shape matches ``(N, ng)`` and whose values
match the independent reference is the Phase C trace law call.
Both vacuum and reflective parametrised cases pass; the test is
foundation-tagged because it pins a software invariant (the
matvec's two-application sequence) rather than a math claim.

.. _sn-phase-d-err-026-closure-narrative:

ERR-026 PARTIAL → PARTIAL (narrowed scope)
-------------------------------------------

.. note:: **Retraction (2026-06-12, Issue #195).**

   The sub-claim table below classified the residual MMS magnitude as
   a benign "pre-asymptotic transient" that finer :math:`n_x` would
   clear (status OPEN, tracked at #195).  **That classification is
   wrong.**  The curvilinear isotropic MMS error did not shrink under
   refinement — it PLATEAUED mesh-independently (orders :math:`\to 0`),
   because the dominant defect was the *angular* closure seed (the
   Carlson proxy source), not a spatial-truncation constant.  The
   "rate :math:`[3.33, 2.46]` already correct, only the constant is
   large" reading was an artefact of the
   :math:`\alpha`-dome-telescoping blindness of the scalar residual.
   ERR-058 (#195) replaced both seeds; the isotropic MMS is now a
   clean :math:`\mathcal{O}(h^2)` ladder.  The table is preserved as
   bug-era evidence — its STATUS / Closed-by interpretation is
   superseded by :ref:`sn-err-058-closure-seed-closeout`.

Phase D **narrows** ERR-026's open scope.  The bug ERR-026
originally diagnosed — *"curvilinear sweep WDD angular closure
converges to wrong fixed-source solution"* — had three
sub-claims, each addressed by a different Wave:

.. list-table:: ERR-026 sub-claim closure tracking
   :header-rows: 1
   :widths: 35 35 30

   * - Sub-claim
     - Status
     - Closed by
   * - Operator identity:
       :math:`(L \cdot \psi_{\text{flat}})_{n,i,g} = \Sigma_t \cdot \psi_{n,i,g}`
       on per-ordinate flat-flux probe
     - **CLOSED**
     - Phase D Carlson seed (Gate 1.1 XPASS)
   * - Convergence rate:
       :math:`\mathcal{O}(h^2)` MMS rate at fixed :math:`N`
     - **CLOSED (rate)**
     - Phase D Carlson seed (empirical rate [3.33, 2.46] across
       refinements; both above the L1 acceptance floor of 1.9)
   * - Convergence magnitude: pre-asymptotic absolute MMS error
       below quadrature floor at practical ``nx`` (:math:`\le 160`)
     - **OPEN**
     - Tracked at `Issue #195
       <https://github.com/deOliveira-R/ORPHEUS/issues/195>`_;
       requires either finer ``nx`` or a higher-order spatial
       closure refinement to fully close

The convergence-rate evidence ``[3.33, 2.46]`` is the slope
sequence measured at successive refinement levels; both values are
above the L1 acceptance floor of 1.9 (second-order accuracy
demonstrated robustly), satisfying the rate sub-claim.  However,
the **absolute magnitude** at the largest tested ``nx`` (=160)
remains above the L1 tolerance that the test architect specified
for full closure on the pre-asymptotic regime.  This is **NOT** a
violation of the Phase D fix — the rate is correct, the asymptotic
regime is the right shape, but the **constant-coefficient** in
front of the :math:`\mathcal{O}(h^2)` term is larger than the
test's pre-asymptotic-magnitude budget at practical mesh
resolutions.

The pre-asymptotic regime is the consequence of the Carlson
sweep's L0-truncated source: at coarse :math:`nx` the Legendre
:math:`\phi_0` moment is computed from the cell-centred input
:math:`\psi` against the GL quadrature — an integration whose own
truncation contributes to the constant in
:math:`\bar Q = \frac{1}{2} \Sigma_t \phi_0`.  Refining
:math:`nx` reduces this contribution, but the rate at which it
reduces is set by the WDD spatial closure's own truncation order,
not by the Carlson sweep itself.  See Issue #195 for the candidate
follow-up paths (higher-order pole-face spatial closure, or a
:math:`\phi_0` recomputation that uses the M-M angular recurrence
output rather than the cell-centred input).

The 4 ``xfail-strict`` ERR-026 tripwires
(:file:`tests/sn/verification/mms/test_curvilinear_aniso_convergence.py`,
sphere + cylinder × isotropic + anisotropic ansatz) therefore
**stay xfail** through Phase D Step 3.  They will ``xpass`` under
the Phase D defaults (which is what triggers the deferred Step 5
marker-removal commit); the pre-asymptotic-magnitude regression
that prevents `strict=True` flipping is Issue #195's domain.  The
narrative for ``error_catalog.rst`` therefore reads:

   ERR-026 status: **PARTIAL CLOSURE** (was PARTIAL through Phase
   C, narrowed scope through Phase D).  The structural bug (M-M
   recurrence hardcoded ψ\ :sub:`1/2,i` = 0 seed) is closed by the
   Phase D Carlson coupled-pole sweep; Gate 1.1 sphere MMS PASS
   confirms the operator identity and the second-order
   convergence rate is recovered.  The pre-asymptotic-magnitude
   open question (Issue #195) is what keeps the status at PARTIAL
   rather than CLOSED.

.. _sn-phase-d-files-touched:

Files touched by Phase D
------------------------

The full Phase D footprint (per the closeout memo at
``.claude/agent-memory/method-implementer/issue_168_phase_d_step3_closeout.md``):

**New modules**

* :mod:`orpheus.sn.sweep.psi_half_angle_seed` — Protocol family
  + ABC + 2 strategies (``ZeroSeed`` + ``CarlsonInwardSweep``)
  + ``CarlsonSweepContext`` dataclass.
* :file:`tests/sn/sweep/curvilinear/test_psi_half_angle_seed.py` — 25
  foundation + L0 + L1 tests covering Protocol conformance,
  registry/self-registration, immutability, shape contract,
  bit-identity for ``ZeroSeed``, L0 algebraic identities
  (flat-:math:`\psi` at varying C, vacuum-BC nx=3 hand
  calculation, multi-region :math:`\Sigma_t` step), linearity, and
  L1 structural-independence (Carlson vs Zero on vacuum-BC probe).

**Modified files**

* :mod:`orpheus.sn.angular.closure` —
  :class:`MorelMontryAngularSweep` gains a
  ``psi_half_seed: PsiHalfAngleSeed`` field; the per-level M-M
  recurrence (then ``_mm_weighted_angular_recurrence_single_level``,
  now :func:`~orpheus.sn.angular.closure.compute_psi_half_per_level`)
  accepts an
  optional ``psi_half_seed`` array; Protocol signatures extended
  with an optional ``carlson_context`` kwarg (Legacy + Bailey
  ignore it).
* :mod:`orpheus.sn.sweep` ``__init__`` re-exports the new
  symbols.
* :mod:`orpheus.sn.operators.streaming` — spherical + cylindrical matvecs
  build the
  ``CarlsonSweepContext``
  before calling ``pole_angular_closure``.
* :mod:`orpheus.sn.mesh.augmented_mesh` — :class:`SNMesh` default flipped to
  :class:`MorelMontryAngularSweep`.
* :mod:`orpheus.sn.solver` — curvilinear default ``inner_solver``
  flipped to ``"krylov"``.
* :file:`tests/sn/sweep/core/test_phase_c_gates.py` (``tests/sn/`` at the
  time; moved by the taxonomy reorg ``105ce125``) — Gate 1.5 strengthened
  with capture-and-compare.
* :file:`tests/sn/test_streaming_operator.py` (post-D-K successor
  to the retired ``test_snstreamingoperator.py``) — 3 tests updated
  (one test docstring rewritten to pin the Phase D fix; two
  bit-identity tests threaded with ``sn_mesh.pole_angular_closure``;
  one linearity tolerance relaxed ``rtol=1e-13 → 1e-12``).

The agent-memory trail for Phase D session reproducibility:

* Literature memo:
  ``.claude/agent-memory/literature-researcher/phase_d_carlson_coupled_pole.md``
  — Hébert §3.9.4 derivation + flat-:math:`\psi` algebra +
  architecture-shape correction + open questions.
* Diagnostic memo:
  ``.claude/agent-memory/numerics-investigator/phase_d_gate_1_1_sphere_mms_diagnosis.md``
  — empirical evidence + 4 plan corrections + structural-
  independence cross-check.
* Step 3 closeout:
  ``.claude/agent-memory/method-implementer/issue_168_phase_d_step3_closeout.md``
  — what shipped + 3 deviations + V&V evidence chain.
* Diagnostic script:
  :file:`tests/sn/diagnostics/gate_1_1_sphere_mms_failure.py`
  — self-contained CLI probe reproducing the diagnostic table.


.. _sn-phase-f-carlson-sweep-path-backport:

Phase F Carlson seed sweep-path backport (Issue #168 Phase F)
=============================================================

.. admonition:: Key Facts
   :class: important

   * Phase F (commit chain landed 2026-05-12 on
     ``refactor/sn-operator-algebra``, atop Phase E ``6708a4a``)
     backports the Phase D Carlson coupled-pole seed
     (``CarlsonInwardSweep``,
     Hébert §3.9.4 Eqs. (3.432)–(3.435)) from the apply-matvec path
     (the then-production ``transport_operator_matvec_spherical``
     / ``_cylindrical`` matvec — since deleted, #197 / #280 campaigns —
     fixed in Phase D Step 3) into the SI/sweep
     path
     (``_sweep_1d_spherical`` (the dissolved ``sweep.py``) and
     ``_sweep_1d_cylindrical``).
   * The bug is the **structural twin** of the Phase D defect: the
     SI loop in :file:`orpheus/sn/sweep.py` initialised
     ``psi_angle = np.zeros((nx, ng))`` at the spherical sweep
     entry (line 474, pre-Phase-F) and at the cylindrical per-level
     loop entry (line 634, pre-Phase-F) — the same hardcoded
     zero seed Phase D diagnosed as wrong-term-initialization on
     the apply-matvec twin
     (``orpheus/sn/sweep/pole_angular_closure.py:411``).
   * Phase F factors a **NEW free function**
     :func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`
     ``(Q_bar, sigma_t, dr, bc_outer_value) -> (ng, nx)`` that runs
     :eq:`hebert-3-434`–:eq:`hebert-3-435` driven by the SI
     within-group source ``Q_1d`` rather than by an apply-path
     :math:`\psi` Legendre fold.
     ``CarlsonInwardSweep.__call__`` is refactored to delegate
     to the same helper after folding ``psi_level → Q̄ = 0.5 ·
     Σ_t · φ_0``.  **One helper, two consumers** — Cardinal Rule 2
     (architecture) enforced via reuse without duplication.
   * Empirical result on ``sphere_2g_3reg`` n=40
     (heterogeneous A|B|A reflective 2-group sphere):
     ``sf[0]/sf[1]`` ratio at the pole was **0.522** (DIVERGING
     to **0.473** under refinement to n=320); post-Phase-F it is
     **0.778** and STABLE under refinement (still 0.777 at n=320).
     The outer-cell reflective-face defect ``sf[-1]/sf[-2]`` was
     **0.887** → **0.997** (essentially CLOSED).
     :math:`\psi(r=0)` quasi-isotropy: ``cv(ψ@i=0)``
     **0.520** → **0.404**, ``max/min(ψ@i=0)`` **6.4×** →
     **1.16×** (Pomraning 1989 prediction substantially approached).
   * **What was logged as open** *(now CLOSED, #196)*: the residual
     O(h) per-cell WDD spatial-closure asymmetry between SI and Krylov
     paths was logged as **manifestation #7 of ERR-026**.  It is now
     **CLOSED** — ERR-058 (#195) showed the gap was a shared
     closure-seed defect, not a discretisation asymmetry; #196 verified
     SI :math:`\equiv` Krylov to the iteration floor on the
     heterogeneous eigenvalue path and added the permanent regression
     gate (see :ref:`sn-phase-f-residual-o-h-open` and
     :ref:`sn-issue-196-eigenvalue-equivalence`).  The Phase E
     flux-shape sentinel
     (:func:`tests.sn.verification.analytical.test_phase_c_crosscheck.test_phase_e_trajectory_resolvent_flux_shape_crosscheck`)
     **no longer xfails** — it runs as a plain L1 test, the
     structurally-independent Variant-α anchor.

The twin-path bug Phase D left open
------------------------------------

Phase D's fix lived entirely in the **apply-matvec path**.  The
Phase D Carlson seed is invoked by
:meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply` via the
``MorelMontryAngularSweep.psi_half_seed`` composition; that
covers every Krylov-driven call.  But ORPHEUS's curvilinear
production default is **source iteration**, which (pre-Phase-F) dispatched
through the then-production ``transport_sweep`` entry rather than
through the apply matvec, and the two paths ran **different code**
to seed the M-M half-angle recurrence:

.. list-table:: Apply vs SI/sweep dispatch divergence (pre-Phase-F)
   :header-rows: 1
   :widths: 24 38 38

   * - Path
     - Carlson seed site
     - Pre-Phase-F state
   * - Apply matvec (Krylov)
     - ``_mm_weighted_angular_recurrence_single_level``
       :math:`\to`
       ``CarlsonInwardSweep``
       via ``MorelMontryAngularSweep.psi_half_seed``
     - **CORRECT** — Phase D Carlson seed installed; Gate 1.1
       XPASS on the per-ordinate flat-flux residual probe
       (residual :math:`\le 10^{-15}`).
   * - SI/sweep (Source Iteration)
     - ``_sweep_1d_spherical`` (the dissolved ``sweep.py``) line 474
       (spherical) and ``_sweep_1d_cylindrical`` line 634
       (cylindrical per-:math:`\mu`-level loop)
     - **WRONG** — hardcoded ``psi_angle = np.zeros((nx, ng))``,
       the very same Phase B zero seed Phase D diagnosed and
       replaced on the apply-matvec twin.  The bug survived
       Phase D's regression suite untouched.

The cylindrical site has its own per-:math:`\mu`-level twin —
each level's azimuthal recurrence enters with the same hardcoded
zero.  Cylindrical Gate 1.1 passed empirically pre-Phase-D
because — for the **level-symmetric** quadrature exercised here — the
first-swept ordinate's seed weight is the **dead** first-ordinate weight
(:math:`c_{\rm in}[m_0]=(1-\tau)/\tau=0` at raw :math:`\tau=1`), which
annihilates the wrong seed at source.  (This is **not**
:math:`\alpha`-dome ':math:`\alpha=0`' level-edge cancellation
*absorbing* the seed — a level-symmetric-only reading, false for a
product quadrature; see the #280 Phase 2.5b correction at
:ref:`sn-phase-d-gate-1-1-empirical`.)  Cardinal Rule 2 (architecture)
nonetheless demands the structural fix on the sister path even when the
empirical signature is invisible there.

Phase F-Step-2 mesh-refinement evidence (sphere)
------------------------------------------------

The Step 2 numerics-investigator probe ran SN on
``sphere_2g_3reg`` (A|B|A reflective 2-group sphere, R=2.0 cm,
GL-8) at :math:`n_{\text{total}} \in \{40, 80, 160, 320\}` and
Variant α (composite-GL trajectory-resolvent reference) at
:math:`n_r \in \{24, 36, 48, 72, 96\}` matching effective
refinements.  The full table from
``.claude/agent-memory/numerics-investigator/phase_f_step2_mesh_refinement.md``:

.. list-table:: SN sphere pre-Phase-F mesh refinement (g=0 ratios)
   :header-rows: 1
   :widths: 10 18 18 18 18 18

   * - :math:`n_{\text{total}}`
     - :math:`k_{\text{eff}}`
     - :math:`\bigl|\text{sf}[0]/\text{sf}[1] - 1\bigr|`
       (pole)
     - :math:`\bigl|\text{sf}[N{-}1]/\text{sf}[N{-}2] - 1\bigr|`
       (outer)
     - log-log slope (pole)
     - log-log slope (outer)
   * - 40
     - 1.3578153066
     - 4.78e-01
     - 1.13e-01
     - —
     - —
   * - 80
     - 1.3576649296
     - 4.94e-01
     - 9.75e-02
     - −0.049 (**DIV**)
     - +0.21
   * - 160
     - 1.3576295736
     - 5.11e-01
     - 6.59e-02
     - −0.049 (**DIV**)
     - +0.57
   * - 320
     - 1.3576226569
     - 5.27e-01
     - 3.88e-02
     - −0.043 (**DIV**)
     - +0.76

A linear-in-:math:`h` extrapolation of the pole ratio gave
``ratio = 0.473 + 1.06·h`` — a **fixed structural asymptote
at 0.473**, not 1.  The outer ratio converged toward 1 at
:math:`\sim \mathcal{O}(h^{3/4})`, slower than the
:math:`\mathcal{O}(h^2)` DD interior, consistent with a
first-order BC-trace truncation that *vanishes* under
refinement.  Variant α at all five refinements gave inner /
outer ratios **monotonically → 1** (1.001949 → 1.000010 inner;
1.027508 → 1.001004 outer), confirming SN as the outlier and
ruling out the BC-interpretation alternative.

Per the
``vv-principles`` Step 2 decision matrix, the pole cell fires
**Branch 3 (DIVERGENT, high urgency)** and the outer cell fires
**Branch 1 (O(h^p), file follow-up)**.  This made the dispatch
to Step 3 (deep diagnostic) mandatory.

Phase F-Step-3 isolation: SI vs Krylov split
---------------------------------------------

The Step 3 diagnostic ran the **same problem** through
:func:`~orpheus.sn.solver.solve_sn` with the only variable
changed being the ``inner_solver`` kwarg:

.. list-table:: SI vs Krylov on ``sphere_2g_3reg`` n=40 (pre-Phase-F)
   :header-rows: 1
   :widths: 26 18 18 19 19

   * - Inner solver
     - :math:`k_{\text{eff}}`
     - :math:`\text{sf}[0]/\text{sf}[1]`
     - :math:`\text{sf}[N{-}1]/\text{sf}[N{-}2]`
     - cv(ψ\@i=0)
   * - ``"source_iteration"`` (sweep)
     - 1.38069560
     - **0.5223**
     - 0.8871
     - **0.520**
   * - ``"krylov"`` (apply matvec)
     - 1.38464040
     - **1.0288**
     - 0.9745
     - 0.445

The Krylov path **eliminates the pole anomaly entirely** at
n=40, and Krylov's pole ratio converges to 1 cleanly under
refinement (1.029 at n=40 → 1.002 at n=80 → 1.0018 at n=160 —
:math:`\sim\mathcal{O}(h^2)` consistent with second-order DD).
Same materials, same quadrature, same mesh, same
:class:`MorelMontryAngularSweep` pole closure with the Phase D
Carlson seed installed on its ``psi_half_seed`` field — the
**only** difference is which inner-solver dispatch is used.
The Krylov path went through
:meth:`StreamingCollisionOperator.apply` (which consumes the Phase D
Carlson seed correctly); the SI path went through the then-production
``transport_sweep`` entry (which carried the **legacy zero
seed** untouched by Phase D).

This split is the smoking gun that pins the bug to
:file:`orpheus/sn/sweep.py:474` (and :file:`orpheus/sn/sweep.py:634`
for the cylindrical per-level twin).  See
``.claude/agent-memory/numerics-investigator/phase_f_step3_diagnostic.md``
for the full empirical trail.

Source-driven Hébert (3.434)–(3.435) — the math
------------------------------------------------

The apply-matvec's Phase D Carlson seed consumes a
:math:`\psi`-current array shaped ``(ng, M, nx)`` and folds it
to :math:`\bar Q = \frac{1}{2} \Sigma_t \phi_0` via the
Legendre :math:`\ell = 0` projection (Eq. :eq:`hebert-3-432-source`
with :math:`P_0(-1) = 1`).  The SI/sweep path has **no such
current :math:`\psi` array at sweep start** — the entire point
of one SI iteration is to *produce* the updated angular flux.
What the SI loop **does** carry at sweep start is the
within-group source

.. math::
   :label: phase-f-q-1d-decomposition

   Q_{\text{1d}}(i, g) \;\equiv\;
   Q^{\text{scatt}}_{\text{within}}(i, g)
   \;+\; \frac{1}{k_{\text{eff}}}\,Q^{\text{fiss}}(i, g)
   \;+\; Q^{\text{ext}}(i, g),

.. (vv-status rationale) Notation definition: the decomposition of the
   within-group SI source (scatter + fission-moment + external).  Not a solver
   claim; verified transitively by the SI convergence infrastructure (the
   inner tolerance enforces the fixed-point identity to machine precision), as
   spelled out in the :ref:`sn-phase-f-test-wiring` note.
.. vv-status: phase-f-q-1d-decomposition documented

i.e. the **isotropic** within-group source from the previous
power-iteration's :term:`scalar flux` + fission moment + external
source.

On the fixed-point solution of the SI loop, the operator
identity :math:`L \cdot \psi = Q_{\text{1d}}` is satisfied
ordinate-by-ordinate, with :math:`L` carrying only
:math:`\Sigma_t \psi` for the current isotropic ORPHEUS scope.
The scalar-flux Legendre moment satisfies
:math:`\phi_0 = \sum_n \mathcal{W}_n \psi_n`.  Combining
gives, on the fixed point,

.. math::
   :label: phase-f-source-eq-sigt-phi0

   \Sigma_t(r) \cdot \phi_0(r) \;=\; Q_{\text{1d}}(r),

.. (vv-status rationale) Derivation step: the SI fixed-point identity
   (Σ_t·φ₀ = Q_1d) that makes the sweep-path Carlson seed canonically
   equivalent to the apply-path form.  Not a standalone claim; verified
   transitively by the SI convergence infrastructure — off the fixed point the
   two forms differ by the SI residual, which vanishes at convergence (see
   :ref:`sn-phase-f-test-wiring`).
.. vv-status: phase-f-source-eq-sigt-phi0 documented

so the cell-averaged source at :math:`\mu = -1` (Eq.
:eq:`hebert-3-432-source` collapsed to :math:`L = 0`,
isotropic) admits two equivalent expressions:

.. math::
   :label: phase-f-q-bar-twin-forms

   \bar Q_i
   \;=\; \tfrac{1}{2}\,\Sigma_t(r_i) \cdot \phi_0(r_i)
   \quad\text{(apply path: builds }\phi_0\text{ from input }\psi\text{)}

   \bar Q_i
   \;=\; \tfrac{1}{2}\,Q_{\text{1d}}(r_i)
   \quad\text{(sweep path: takes }Q_{\text{1d}}\text{ directly).}

**The two are identical on the fixed point** by Eq.
:eq:`phase-f-source-eq-sigt-phi0`.  Off the fixed point they
differ by the SI residual :math:`r_k = Q_{\text{1d}} -
\Sigma_t \phi_0^{(k)}`, which vanishes as the SI loop
converges.  The sweep path's source-driven Carlson seed is
therefore the **canonically equivalent** invocation of the
same Hébert §3.9.4 math, packaged for a code path that has
:math:`Q_{\text{1d}}` available but not the per-ordinate
:math:`\psi`.

The factor :math:`\tfrac{1}{2}` is the Legendre fold weight
:math:`(2\ell + 1)/2` at :math:`\ell = 0` times
:math:`P_0(-1) = 1`.  For an :math:`L \ge 1` anisotropic
operator (not currently in scope for ORPHEUS's apply
matvec, but flagged in the
``CarlsonInwardSweep``
class docstring's L=0 WARNING block), additional terms
:math:`(2\ell + 1) Q_\ell \cdot (-1)^\ell / 2` for
:math:`\ell \ge 1` would enter — the source-driven helper
would need a moment vector ``Q_ell[ell, i, g]`` rather than
the present ``Q_bar[i, g]`` to recover the canonical
construction.

With :math:`\bar Q_i` from either formula, the inward DD
recurrence Eqs. :eq:`hebert-3-434`–:eq:`hebert-3-435`
proceed identically to the apply path:

.. math::
   :label: phase-f-carlson-seed-source-driven

   \bar\phi_i \;=\;
   \frac{\Delta r_i \cdot \tfrac{1}{2}\,Q_{\text{1d}}(r_i)
          \;+\; 2 \cdot \bar\phi_{i+1/2}}
        {\Delta r_i \cdot \Sigma_t(r_i) \;+\; 2},
   \qquad
   \bar\phi_{i-1/2} \;=\; 2 \cdot \bar\phi_i - \bar\phi_{i+1/2}

(sequential in cells from :math:`i = nx - 1` inward to
:math:`i = 0`, vectorised across groups).  The
``bc_outer_value`` at :math:`\bar\phi_{nx+1/2}` is the
outer-face angular flux at :math:`\mu = -1`, realised through
the BC operator on the persistent outflow buffer
``bc_outer``.

Equivalence on the converged eigenmode
--------------------------------------

Foundation test
:func:`tests.sn.sweep.core.test_sweep_vs_apply_consistency`
pins the source-vs-:math:`\psi` equivalence directly: for any
flat-:math:`\psi` field ``ψ_const`` with ``bc_outer_value =
ψ_const`` (reflective) and ``Q_1d = Σ_t · Σw · ψ_const`` (the
within-group source built by SI from
``φ_0 = Σw · ψ_const``), the two helpers return
**bit-identical seeds** (up to FP non-associativity).
Apply-path:
``CarlsonInwardSweep``
``(psi_level=ψ_const·ones, ctx)`` produces ``Q̄ = 0.5 · Σ_t · Σw
· ψ_const``; sweep-path:
:func:`carlson_inward_sweep_from_source`
``(Q_bar=0.5·Q_1d, ...)`` produces the same ``Q̄`` — the
recurrence is identical, the bit-equal result is the
**single-invariant property** the test pins.

The architectural choice: one helper, two consumers
---------------------------------------------------

Phase F's structural choice is **factor the helper, delegate
from the strategy** — the Cardinal Rule 2 (architecture)
imperative.  The pre-Phase-F implementation had Eqs.
:eq:`hebert-3-434`–:eq:`hebert-3-435` open-coded inside
``CarlsonInwardSweep.__call__``.  Naive options for the
backport:

* **Option 1 (REJECTED) — duplicate the recurrence loop in
  the sweep path.**  Two copies of the inward DD recurrence,
  one driven by ``Q̄ = 0.5 · Σ_t · φ_0`` (apply path), one by
  ``Q̄ = 0.5 · Q_1d`` (sweep path).  Equivalent at the
  algorithmic level but a Cardinal-Rule-2 architecture
  violation: a future bug fix to one copy would need to be
  audited against the sister — exactly the failure mode that
  produced the Phase F bug in the first place.
* **Option 2 (REJECTED) — invoke**
  ``CarlsonInwardSweep``
  **directly from the sweep path with a synthesized**
  ``psi_level``
  **array.**  The strategy's ``__call__(psi_level,
  context)`` Protocol signature takes ``(ng, M, nx)`` — the
  sweep would have to allocate a flat-:math:`\psi` proxy of
  the right shape just to feed it through the Legendre fold
  that would extract ``φ_0`` from the proxy.  Mathematically
  equivalent but wasteful and obscures intent.
* **Option 3 (SHIPPED) — factor**
  :func:`carlson_inward_sweep_from_source`
  **as a free function that takes** ``Q̄``
  **directly, and have the strategy delegate.**

``CarlsonInwardSweep.__call__`` now reads (in essence):

.. code-block:: python

   def __call__(self, psi_level, context):
       # ψ -> φ_0 -> Q̄ fold (apply-path-specific)
       phi_0 = np.einsum("gmi,m->gi", psi_level, context.weights)
       Q_bar = 0.5 * context.sigma_t * phi_0
       # Delegate to the source-driven recurrence
       return carlson_inward_sweep_from_source(
           Q_bar=Q_bar,
           sigma_t=context.sigma_t,
           dr=context.dr,
           bc_outer_value=context.bc_outer_value,
       )

The sweep path consumes the helper directly with ``Q_bar = 0.5
· Q_1d.T``.  **Single source of truth, two structurally
equivalent invocation points.**  A future bug fix to the
recurrence (e.g., an :math:`L \ge 1` anisotropic extension)
lands in **one** place; both consumers inherit it
automatically.

Why the cylindrical site needed the fix too
--------------------------------------------

Cylindrical Gate 1.1 passes empirically pre-Phase-F (for the
**level-symmetric** quadrature exercised here the first-swept
ordinate's seed weight is the **dead** first-ordinate weight
:math:`c_{\rm in}[m_0]=(1-\tau)/\tau=0` at raw :math:`\tau=1`, which
annihilates the wrong zero seed at source — **not** :math:`\alpha`-dome
telescoping "absorbing" it, a level-symmetric-only reading false for a
product quadrature; see the :ref:`sn-phase-d-gate-1-1-empirical`
discussion and its #280 Phase 2.5b correction for the
sphere-vs-cylinder asymmetry).  Phase F nonetheless fixes
both sites for two reasons:

#. **Cardinal Rule 2 (architecture)**: structural alignment of
   the canonical math at both sites prevents a future
   refactor from introducing an asymmetric bug that only the
   sphere catches.  The sweep-path helper is the same code
   regardless of geometry; consuming it consistently from
   both geometries is the architecturally clean choice.
#. **Defense in depth against future stress probes**: on any
   cylinder rule where the first-ordinate seed weight is **live**
   (a **product** quadrature already is — :math:`c_{\rm in}[m_0]\ne 0`,
   #280 Phase 2.5b), the wrong zero seed enters the fixed point.
   Fixing both sites now is cheap insurance.

The cylindrical fix sits inside the per-:math:`\mu`-level
loop (lines 678–714 of :file:`orpheus/sn/sweep.py`).  The
helper is invoked **once per level** with the level-specific
``bc_outer_value`` extracted from the persistent outflow
buffer at the most-inward ordinate of the level.  The
linearity of the helper in ``Q_bar`` and ``bc_outer_value``
ensures the per-level invocations remain commutative with
the outer-loop level iteration.

Phase F empirical evidence (post-fix)
-------------------------------------

The post-Phase-F state recovers the canonical SN behaviour on
the smoking-gun case:

.. list-table:: ``sphere_2g_3reg`` n=40 — pre/post Phase F
   :header-rows: 1
   :widths: 36 22 22 20

   * - Diagnostic
     - Pre-Phase-F (SI)
     - Post-Phase-F (SI)
     - Krylov (reference)
   * - :math:`\text{sf}[0]/\text{sf}[1]`
       (pole ratio, target ~1)
     - **0.522**
     - **0.778**
     - 1.029
   * - :math:`\text{sf}[N{-}1]/\text{sf}[N{-}2]`
       (outer ratio, target ~1)
     - 0.887
     - **0.997**
     - 0.974
   * - :math:`\text{cv}(\psi@i=0)`
       (Pomraning isotropy, target ~0)
     - 0.520
     - **0.404**
     - 0.445
   * - :math:`\max/\min(\psi@i=0)`
       (target ~1)
     - **6.4×**
     - **1.16×**
     - 1.18×
   * - :math:`k_{\text{eff}}`
     - 1.38069560
     - 1.38069560
     - 1.38464040

The pole ratio jumps from a structural plateau at 0.473–0.522
(divergent under refinement) up to a stable ~0.778 plateau
that holds at n=320 — the **structural divergence is
gone**.  ``sf[0]/sf[1] = 0.778`` is not yet ``1`` because the
SI fixed point still differs from the Krylov fixed point by
the residual O(h) WDD spatial-closure asymmetry (see
:ref:`sn-phase-f-residual-o-h-open` below), but the
**diverging-vs-refinement** signature that made the Phase E
flux-shape sentinel xfail-strict is closed.

.. note:: **Retraction (2026-06-13, Issue #196).**

   The table below logs a residual SI :math:`\neq` Krylov
   :math:`\mathcal{O}(h)` gap (pole ratio 0.778 vs Krylov 1.029;
   :math:`\Delta k` 0.286 % at n=40 halving per mesh doubling) and
   reads it as a benign discretisation artefact of "two methods now
   solving the same equation".  **That interpretation is wrong.**  The
   methods did NOT yet solve the same equation at Phase F: the
   *shared* closure seeds were still O(1)-wrong on non-flat fields
   (ERR-058).  After ERR-058 (#195) fixed the seeds, the
   :math:`\mathcal{O}(h)` gap **collapsed to the iteration floor** —
   SI :math:`\equiv` Krylov to :math:`|\Delta k|\approx
   1.9\mathrm{e}{-11}` and L∞ flux-shape :math:`\approx
   2.4\mathrm{e}{-10}` on ``sphere_2g_3reg`` n=40 (from a bug-era
   3.9e-3 / ~30 %); the pole ratio reaches 1 to that floor.  The
   measured numbers stay below as bug-era evidence; the
   production-decision record and post-fix evidence are
   :ref:`sn-issue-196-eigenvalue-equivalence`.

Mesh-refinement convergence (SI vs Krylov, post-Phase-F):

.. list-table:: Post-Phase-F SI-vs-Krylov convergence on ``sphere_2g_3reg`` (bug-era — gap closed by ERR-058/#196)
   :header-rows: 1
   :widths: 10 22 18 22 18 14

   * - :math:`n`
     - :math:`k_{\text{eff}}` (SI)
     - :math:`\text{sf}[0]/\text{sf}[1]` (SI)
     - :math:`k_{\text{eff}}` (Kr)
     - :math:`\text{sf}[0]/\text{sf}[1]` (Kr)
     - :math:`\Delta k`
   * - 40
     - 1.38069560
     - 0.7776
     - 1.38464040
     - 1.0288
     - 0.286 %
   * - 80
     - 1.38075258
     - 0.7771
     - 1.38261730
     - 1.0125
     - 0.135 %
   * - 160
     - 1.38078077
     - 0.7771
     - 1.38167934
     - 1.0018
     - 0.065 %

*(Bug-era reading, retracted — see the note above.)* The
:math:`k_{\text{eff}}` gap between SI and Krylov drops
by a factor of 2 per mesh doubling — apparent clean
:math:`\mathcal{O}(h)` convergence to a shared limit.
Pre-Phase-F the SI sat on the wrong structural fixed point
(0.473–0.522 ratio asymptote diverging from 1) while Krylov
converged to ~1 — the two methods **solved different
equations**, and refinement made it worse for SI.  Phase F
removed the *divergent* signature but, as ERR-058 later
showed, left a *shared* O(1)-on-non-flat seed defect: the two
paths still did not solve the same equation, and the residual
:math:`\mathcal{O}(h)` gap above is the slow trace of that
shared defect, not a discretisation artefact.  ERR-058 (#195)
fixed the seeds and the gap collapsed to the iteration floor
(:ref:`sn-issue-196-eigenvalue-equivalence`); there is no
residual O(h) gap in production.

Files touched by Phase F
------------------------

**Modified production code**

* :mod:`orpheus.sn.sweep.psi_half_angle_seed` — NEW free
  function :func:`carlson_inward_sweep_from_source` (lines
  358–419 of the module); ``CarlsonInwardSweep.__call__``
  refactored to delegate after folding ``psi_level → Q̄``;
  ``__all__`` extended.
* ``orpheus.sn.loss_representation`` (the dissolved ``sweep.py``) —
  ``_sweep_1d_spherical`` line ≈ 472–530: replaces the
  legacy ``psi_angle = np.zeros((nx, ng))`` with the Phase F
  Carlson seed call (uses ``bc_outer_obj.apply(bc_outer)`` to
  derive ``bc_outer_value`` at the most-inward ordinate, mirror
  of the apply-path's Phase D logic);
  ``_sweep_1d_cylindrical`` lines ≈ 678–714: per-level
  Carlson seed inside the :math:`\mu`-level loop, replaces
  the inline level-zero init.

**Tests added**

* :func:`tests.sn.sweep.core.test_phase_c_gates.test_sweep_curvilinear_per_ordinate_flat_flux_residual`
  — **Gate 1.6**, the dual of Gate 1.1 for the SI/sweep path.
  Parametrised over geometry (sphere × cylinder) and
  :math:`\Sigma_t \in \{0.5, 1.5\}`.  Pins
  apply-path-vs-sweep-path bit-identity on the helper output
  AND the flat-:math:`\psi` algebraic identity at :math:`\Sigma
  w = 2` (Hébert convention).  Carries
  ``@pytest.mark.verifies("dd-curvilinear-scalar")`` and
  ``@pytest.mark.catches("ERR-026")`` — see
  :ref:`sn-phase-f-test-wiring` for the proposed extension to
  the Phase F equation labels.
* :file:`tests/sn/sweep/core/test_sweep_vs_apply_consistency.py` —
  NEW file, **57 foundation tests** pinning:

  #. Apply-path vs sweep-path Carlson seed bit-equivalence on
     matching ``Q̄`` (the load-bearing structural invariant).
  #. Linearity of
     :func:`carlson_inward_sweep_from_source` in ``Q_bar`` and
     ``bc_outer_value`` independently (Protocol-shape contract
     preservation).
  #. SI-vs-Krylov :math:`k_{\text{eff}}` agreement on
     homogeneous reflective spheres (the degenerate case
     where the Phase F fix is invariant — same eigenvalue
     pre- and post-fix).

**Updated tests**

* :func:`tests.sn.verification.analytical.test_phase_c_crosscheck.test_phase_e_trajectory_resolvent_flux_shape_crosscheck`
  — *(Phase-F action, since superseded.)* Phase F updated the
  ``xfail-strict`` reason string from *"UNRESOLVED structural
  discrepancy with hypothesised pole issue"* to *"Phase F closed
  gross divergence; residual O(h) drift awaits further work"*, on
  the expectation a future tightening would self-enforce removal.
  **The xfail was removed by ERR-058 (#195):** the canary now runs
  as a plain L1 test (the structurally-independent Variant-α anchor;
  see :ref:`sn-issue-196-eigenvalue-equivalence`).

**Snapshot regeneration**

* 6 curvilinear regression snapshots regenerated under the
  Phase F fix:

  * ``tests/sn/regression/snapshots/sphere_2g_homogeneous_dd_n20.npz``
  * ``tests/sn/regression/snapshots/sphere_2g_3reg_dd_n40.npz``
  * ``tests/sn/regression/snapshots/sphere_2g_p1_aniso_dd_n20.npz``
  * ``tests/sn/regression/snapshots/cyl_1g_homogeneous_LS4_dd_n20.npz``
  * ``tests/sn/regression/snapshots/cyl_1g_homogeneous_product_dd_n20.npz``
  * ``tests/sn/regression/snapshots/cyl_2g_3reg_LS4_dd_n40.npz``

  Bit-identity break is principled per the
  ``vv-principles`` *"Bit-identity vs principled-equivalence"*
  framework: the new seed is the canonical Hébert value
  (replaces the diagnosed wrong zero); the
  structurally-independent verification reference is
  Variant α (composite-GL trajectory-resolvent, accessed via
  Gate 4.2); the drift is algorithmic (intended) and
  well-defined.  All 5 Gate 4.2 snapshots still PASS at the
  Phase E tightened tolerances (sphere
  :math:`r_{\text{tol}} = 2 \times 10^{-2}`, cylinder
  :math:`3 \times 10^{-2}`).

.. _sn-phase-f-residual-o-h-open:

ERR-026 manifestation #7 — CLOSED by ERR-058 (#195), verified + pinned by #196
------------------------------------------------------------------------------

.. admonition:: Status — manifestation #7 is CLOSED (Issue #196, 2026-06-13)
   :class: important

   **This was the LAST open manifestation of ERR-026; closing it
   formally retires the curvilinear-SN wrong-fixed-point family.**  The
   "residual :math:`\mathcal{O}(h)` SI-vs-Krylov gap" reading below —
   including the SI :math:`\neq` Krylov tables above (pole ratio 0.778
   vs Krylov 1.029; :math:`\Delta k` converging :math:`\mathcal{O}(h)`)
   and Options (a)/(b)/(c) — was the *two-distinct-systems* picture and
   is **bug-era history**.  The gap was NOT a discretisation artefact;
   it was the shared closure-seed defect (ERR-058), manifest
   differently on the two then-distinct paths.

   * **ERR-048** (Phase G Step 2, 2026-05-13) closed only the **L0
     flat-field** twin-agreement: it patched the SI sweep to MATCH the
     apply-matvec conventions on the homogeneous streaming-equilibrium
     gauntlet (pole-face WDD IC mirror + Carlson seed normalisation).
     The **L1 heterogeneous eigenvalue** :math:`\mathcal{O}(h)`
     asymmetry that manifestation #7 names **PERSISTED** — which is
     exactly why #196 stayed OPEN — because the shared closure seeds
     were still *exact-on-flat / O(1)-wrong-on-non-flat* (the ERR-058
     defect).
   * **ERR-058** (Issue #195, 2026-06-12) was the TERMINAL fix: it
     replaced the shared closure seeds with correct ones — the
     coupled-pole spatial seed :math:`\psi(0,+\mu)=\psi(0,-\mu)` and the
     ``AngularEdgeExtrapolation``
     half-angle seed — so BOTH inner solvers operate on the SAME correct
     discrete operator.
   * **#196** (2026-06-13) VERIFIED the eigenvalue-path equivalence and
     added the permanent regression gate.  See
     :ref:`sn-issue-196-eigenvalue-equivalence` for the measured
     evidence (sphere :math:`|\Delta k|=4.68\mathrm{e}{-12}`, cylinder
     :math:`1.91\mathrm{e}{-11}` on the bug-era snapshot cases) and the
     gate description.

   Option (c) (keep SI, accept an O(h) gap) is moot — there is no gap;
   Option (b) (flip to Krylov) is the opposite of what landed (SI is
   restored as the faster default).  The full production-decision
   record is :ref:`sn-issue-196-eigenvalue-equivalence`.

The bug-era reading (preserved as history) ran: Phase F closed the
**structural** pole defect (the divergent ratio at the pole cell on
heterogeneous MR) and the **outer-cell** defect (sf[-1]/sf[-2]
essentially reaches 1); what was thought to remain was a milder
**convergence-rate** gap between SI and Krylov on heterogeneous MR
snapshots (at n=40 per-cell shape differing by ~5 %, apparently
converging :math:`\mathcal{O}(h)` toward zero under refinement),
logged in ``error_catalog.rst`` as **ERR-026 manifestation #7**:

   *"SI-vs-Krylov per-cell agreement (residual O(h) WDD
   asymmetry) — OPEN, new follow-up after Phase F."*

That row now reads **CLOSED by ERR-058 (#195), verified + pinned by
#196**.  The Phase E flux-shape sentinel
:func:`tests.sn.verification.analytical.test_phase_c_crosscheck.test_phase_e_trajectory_resolvent_flux_shape_crosscheck`
**no longer xfails** — it runs as a plain L1 test (the
structurally-independent Variant-α anchor; see
:ref:`sn-issue-196-eigenvalue-equivalence`).  The two viable
closures that were tracked as Phase F-extensions are recorded here
**only as bug-era history** — neither was taken, because both
presupposed the shared fixed point was correct and only the
arithmetic differed, which the terminal diagnosis refuted:

* **Option (a) — Sweep WDD-closure refinement** *(bug-era,
  not taken).*  Investigate the per-cell WDD recurrence
  :math:`\psi_{n+1/2} = (\psi_n - (1-\tau)\psi_{n-1/2})/\tau`
  in ``_sweep_1d_spherical`` to identify the residual
  numerical asymmetry vs the apply matvec's symmetric closure
  :math:`\psi_{n\pm 1/2} = \tau \psi_{\text{next}} +
  (1-\tau) \psi_{\text{this}}`.  This presumed the seed was
  correct and only the spatial closure differed — false: the
  seed itself was the defect.
* **Option (b) — Flip curvilinear ``inner_solver`` default to
  Krylov** *(bug-era, not taken — in fact reverted).*
  :func:`solve_sn` for spherical / cylindrical would route
  through the Krylov inner (which carried the Phase D Carlson
  seed and produced the cleanly-converging fixed point).  This
  was the Phase-D-era flip; ERR-058 made the SI sweep correct, so
  the default was **reverted to ``source_iteration``** (SI is
  :math:`\sim 10^2\times` faster and now equivalent).

Phase F shipped **option (c)** at the time (keep SI default, achieve
structural alignment of the seed math, accept a residual O(h) gap)
on the reasoning that the methods "now solve the same equation".
The terminal diagnosis (ERR-058) showed they did **not** yet solve
the same equation — the *shared* fixed point was itself wrong on
non-flat fields.  Once the seeds were fixed there is no residual gap
to accept: SI and Krylov agree to the iteration floor at the
eigenvalue level (see :ref:`sn-issue-196-eigenvalue-equivalence`),
and bit-identically at the fixed-source level.

The anti-pattern Phase F surfaced
---------------------------------

**Twin-path fix incompleteness** is a Mode-3
(missing-factor / wrong-term-initialization) anti-pattern.
The Phase D fix was scoped to the apply-matvec path because
Gate 1.1 runs through the apply-matvec; the SI/sweep path's
zero seed was untouched.  The bug survived Phase D's entire
regression suite untouched because:

* Phase D's Gate 1.1 MMS xfail-strict marker is on the
  apply-matvec path; the SI/sweep path didn't run that probe.
* The 6 curvilinear regression snapshots were SI-generated
  under the wrong seed; the snapshots **encoded the bug
  bit-identically** and "passed" by tautology.
* Homogeneous degenerate cases (1G, flat-flux reflective) gave
  k = νΣ_f / Σ_a independent of the flux shape, masking the
  structural divergence on the eigenvalue side.
* The heterogeneous-MR case was marked ``xfail`` for **flux
  shape** (Phase E), not for **eigenvalue**, so the
  shape-sentinel signal was deliberately not enforced.

The lesson, **proposed for addition to**
``vv-principles/SKILL.md`` *§ Anti-patterns* (per the Phase F
closeout memo §"Lessons (proposed for skill catalogue)"):

   *Whenever a fix is applied to one of two structurally-
   mirrored production paths (apply-matvec vs SI/sweep,
   prepass vs postpass, etc.), MUST audit the OTHER path for
   the same defect.  Mode 3 wrong-term-initialization
   defects often appear in pairs; fixing one path without
   auditing its sister is a Cardinal Rule 2 (architecture)
   violation that ERR-026 instantiated twice.*

.. _sn-phase-f-test-wiring:

Test wiring proposal — Phase F equation labels
----------------------------------------------

Phase F declares three new equation labels:
:eq:`phase-f-q-1d-decomposition`,
:eq:`phase-f-source-eq-sigt-phi0`,
:eq:`phase-f-q-bar-twin-forms`, and
:eq:`phase-f-carlson-seed-source-driven`.  The
:eq:`phase-f-carlson-seed-source-driven` label is the
canonical Hébert (3.434)–(3.435) recurrence in
source-driven form — semantically the **same recurrence**
as :eq:`hebert-3-434` and :eq:`hebert-3-435` but with the
sweep-path source substitution made explicit.

The new Gate 1.6 test
:func:`tests.sn.sweep.core.test_phase_c_gates.test_sweep_curvilinear_per_ordinate_flat_flux_residual`
already carries
``@pytest.mark.verifies("dd-curvilinear-scalar")`` and
``@pytest.mark.catches("ERR-026")``.  Per the project's
V&V harness wiring, the test SHOULD additionally declare:

* ``@pytest.mark.verifies("phase-f-carlson-seed-source-driven")``
  — pins the source-driven recurrence on the bit-identity
  helper-vs-strategy probe.
* ``@pytest.mark.verifies("phase-f-q-bar-twin-forms")``
  — pins the apply-vs-sweep source-equivalence identity (the
  load-bearing structural invariant of Phase F).

The other two labels
(:eq:`phase-f-q-1d-decomposition` and
:eq:`phase-f-source-eq-sigt-phi0`) document the
*decomposition* of the SI source and the *fixed-point
identity* :math:`\Sigma_t \phi_0 = Q_{\text{1d}}`; both are
verified transitively by the existing SI convergence
infrastructure (the SI inner-tolerance is the gate that
enforces the fixed-point identity to machine precision).
The proposed wiring is tracked as a follow-up to the V&V
audit harness (see Issue #194 for the sister case of
``hebert-3-43X`` labels — same pattern, same fix).

Pointers
--------

* **Phase F plan**:
  ``.claude/plans/issue_168_phase_f_curvilinear_boundary_eigenvector.md``
  — context, hypothesis, three-step structure, sub-agent
  dispatch chain.
* **Step 2 numerics memo**:
  ``.claude/agent-memory/numerics-investigator/phase_f_step2_mesh_refinement.md``
  — mesh-refinement convergence study, SN-vs-Variant-α
  outlier identification, Step 2 branch-3 decision.
* **Step 3 diagnostic memo**:
  ``.claude/agent-memory/numerics-investigator/phase_f_step3_diagnostic.md``
  — fix-site identification (the smoking gun), SI-vs-Krylov
  isolation, Option-A-vs-B implementation analysis.
* **Phase F closeout memo**:
  ``.claude/agent-memory/method-implementer/issue_168_phase_f_closeout.md``
  — what shipped, the empirical evidence tables, files
  touched, residual-open items.
* **ERR-026 catalogue narrative**:
  ``docs/theory/verification/error_catalog.rst`` (§ ERR-026,
  *"What Wave H Phase F added"*) — manifestation table update
  #6 CLOSED, #7 (new) OPEN.
* **Sister section on the BC apply call sequence**:
  :ref:`bc-three-bc-applies-per-sweep-iteration` in
  :doc:`/theory/foundations/boundary_conditions` — extends the Phase D
  two-BC-applies-per-matvec narrative to cover the SI sweep's
  Phase F invocation.


.. _sn-err-058-closure-seed-closeout:

ERR-058 — the curvilinear closure-seed fix (Issue #195 CLOSED)
==============================================================

.. admonition:: Status banner
   :class: important

   **Issue #195 CLOSED 2026-06-12.**  ERR-058 closes the curvilinear
   *wrong-fixed-point* family — the open loop the Phase A–F narrative
   tracked under the name "ERR-026 PARTIAL CLOSURE".  Two
   independent closure SEEDS in the curvilinear within-group operator
   were wrong on every non-flat field; both are now replaced.  In
   production:

   * The **half-angle thread seed** is
     ``AngularEdgeExtrapolation``
     (the new
     :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
     ``psi_half_seed`` default).  It replaces
     ``CarlsonInwardSweep``,
     whose proxy source :math:`\bar Q = \Sigma_t\phi_0/\!\sum w` was the
     dominant defect.
   * The **spatial pole-face seed** of the outward (:math:`\mu>0`)
     sweep is the *mirror inward sweep's pole-face outflow* — the
     Carlson coupled-pole continuity :math:`\psi(0,+\mu)=\psi(0,-\mu)`
     — replacing the historical innermost-cell-centre read
     :math:`\psi(\Delta r/2)`.
   * The curvilinear inner default returned from ``"krylov"`` to
     ``"source_iteration"`` (both
     :func:`~orpheus.sn.solver.solve_sn_fixed_source` and the
     eigenvalue :func:`~orpheus.sn.solver.solve_sn`): post-unification
     the sweep and matvec are ONE discrete system, so SI
     :math:`\equiv` Krylov **bit-identical for fixed-source** and to
     the **iteration floor for eigenvalue** (the eigenvalue solve wraps
     the inner in power iteration — see
     :ref:`sn-issue-196-bit-identical-vs-floor`).  SI is
     :math:`\sim 10^2\times` faster than GMRES (no restart, ERR-053).

   ``CarlsonInwardSweep`` is **retained** (not deleted) as the
   registered host of the canonical Hébert §3.9.4 recurrence
   (:func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`),
   reachable only by explicit opt-in.  Its class docstring carries a
   ``.. warning::`` block recording the proxy-source caveat by design,
   so a future session cannot re-activate it as a default unaware of
   the falsification.

   .. note:: **Retraction (2026-07-04, Issue #282 route (a)).**  The
      ``CarlsonInwardSweep`` *strategy class* was NOT ultimately
      retained — route (a) deleted the whole ``PsiHalfAngleSeed``
      family.  What survives is the pure Hébert recurrence
      :func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`
      (a free function, now the SOLVE engine driven by the **true** q½
      source rather than the falsified proxy, no opt-in), plus the
      inlined
      :meth:`~orpheus.sn.angular.closure.MorelMontryAngularSweep.edge_extrapolated_seed`
      for non-carrying cylinder levels — a class no ``SNMesh``-admitted
      cylinder has since Q5.6.3, leaving that inline unreachable
      through the mesh.  See
      :ref:`sn-direct-seed-strategy-zoo`.

   The **anisotropic** curvilinear MMS gates improved :math:`\sim 50\times`
   and are now limited by a *fixed-quadrature angular floor* of the
   half-angle thread interpolation — a test-design retune tracked at
   `Issue #229 <https://github.com/deOliveira-R/ORPHEUS/issues/229>`_,
   **not** a residual instance of this wrong-fixed-point class.

Motivation preserved — what the Phase A–F loop was chasing
-------------------------------------------------------------

The Phase A–F sections — A, D, and F above; B and C with the
production machinery in :doc:`curvilinear_one_group` — are preserved
verbatim as the *investigation history*.  Their reasoning was sound at the time and is
pedagogically load-bearing — a future reader asking "why did anyone
try a Carlson inward sweep, an apply-vs-sweep twin audit, a
Krylov-default flip?" must find the answer there.  This subsection only
flips the tenses on the *terminal* claims those sections reached:

* Phase D **was expected to** close ERR-026 once the apply-matvec
  half-angle seed was made non-zero; it closed the per-ordinate
  *flat-flux identity* but left the assembled operator wrong on
  non-flat profiles (the Carlson proxy source).  The "PARTIAL CLOSURE /
  pre-asymptotic transient" framing it shipped **was** the best
  available reading of the evidence then; it is now superseded.
* Phase F **was expected to** close the SI-vs-Krylov gap by backporting
  the same Carlson seed to the sweep.  It did make the two paths share
  the *seed strategy*, but both still drove it with the wrong proxy
  source, so the residual "manifestation #7 O(h) gap" it logged was a
  symptom of the shared defect, not a discretisation artefact.  After
  the Depth-B/Wave-T matvec unification, the sweep and matvec became
  ONE discrete system; the gap vanished by construction once the seed
  was fixed.

The premise the *issue itself* carried — a benign "pre-asymptotic
transient" that finer meshes would clear — was **empirically refuted**:
on ``main`` the isotropic curvilinear MMS error PLATEAUS
mesh-independently (sphere :math:`\sim 0.0413`, cylinder
:math:`\sim 0.0494`, :math:`n_x` 20 :math:`\to` 640, orders
:math:`\to 0`), with SI :math:`\equiv` Krylov bit-identical.  No
refinement helps a plateau.

The two manifestations — one class, both flat-field-exact
-------------------------------------------------------------

ERR-058 is a **closure-seed inconsistency**: two discrete seeds inside
the curvilinear within-group operator were constructed so that they are
*exact* on spatially/angularly flat fields and *O(1)-wrong* on every
other field.  Because a discrete closure seed is part of the operator,
each seed had to be verified per-ordinate on a NON-flat field — and was
not.

.. _sn-err-058-manifestation-a:

Manifestation (a) — the spatial pole-face seed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The outward (:math:`\mu>0`) radial sweep needs an inflow value at the
pole face :math:`r=0`.  The historical matvec
(now the fused
:meth:`~orpheus.sn.loss_representation._OneDimScanWalk._apply_walk`) and
the sweep both read the innermost CELL-CENTRE flux :math:`\psi(\Delta r/2)`
as if it were the pole-FACE value — a half-cell offset.  On a flat
radial profile :math:`\psi(\Delta r/2)=\psi(0)`, so the read is exact;
on the manufactured :math:`A(r)=\sin(\pi r/R)` ansatz it is
:math:`\mathcal{O}(h)`-wrong.  The DD face chain propagates that seed
error as an *undamped odd–even alternation*, and the area-weighted
streaming amplifies it by :math:`\sim A/V \sim 1/r` near the pole.

**The fix — Carlson coupled-pole continuity.**  At :math:`r=0` the
outward characteristic is the *continuation* of the inward one: a
neutron travelling inward along :math:`-\mu` that reaches the centre
emerges travelling outward along :math:`+\mu`, so

.. math::
   :label: sn-err-058-coupled-pole-continuity

   \psi(0,\,+\mu) \;=\; \psi(0,\,-\mu).

.. (vv-status rationale) Representational identity: the r=0
   pole-continuity boundary condition coupling the mirror ordinates.
   Not a solver claim (no eigenvalue / flux value). The verifiable
   content is the per-ordinate operator-admission gate
   (test_curvilinear_operator_admits_mms, catches ERR-058) + the
   strategy-owned seed-adjoint bit-identity (test_g_adjoint_reciprocity).
.. vv-status: sn-err-058-coupled-pole-continuity documented

The :math:`-1` (inward) sweep is therefore run FIRST.  Its pole-face
outflow, read at the *mirror* ordinate (the x-mirror pairing
``_ensure_pole_mirror`` derives from
:meth:`~orpheus.numerics.quadrature.Quadrature.ordinate_permutation`),
seeds the :math:`+1` (outward) sweep.
This is **data** — propagated from the outer boundary, lower-triangular
in cell-visit order — not a self-reference.  It is the
"inward-determines-outward" pole condition deferred at Phase C
(`Issue #192 <https://github.com/deOliveira-R/ORPHEUS/issues/192>`_),
now landed.  The seed is exact on flat :math:`\psi` (so every flat-flux
gate is untouched), lower-triangular (so the operator stays
forward-substitutable), and the matvec and sweep capture/consume it
identically (so the pair stays ONE discrete system).

The **adjoint** routes the :math:`+1` seed cotangent into the
:math:`-1` reversal's initial outflow cotangent at the mirrored
ordinates (see
:meth:`~orpheus.sn.loss_representation._OneDimScanWalk.loss_action_transpose`,
pinned by the dense-probe transpose oracle
``derivations/diagnostics/diag_p42_adjoint_oracle.py``).

.. _sn-coupled-pole-mu-level-invariant:

The μ-level-preservation invariant the mirror seed relies on
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The coupled-pole continuity :eq:`sn-err-058-coupled-pole-continuity`
seeds the outward (:math:`+\mu`) pole face from the inward (:math:`-\mu`)
sweep's pole outflow read **at the mirror ordinate** — concretely,
``pole_face_seed = outflow_at_inner.T[self._ensure_pole_mirror()]`` in
the fused matvec
(:meth:`~orpheus.sn.loss_representation._OneDimScanWalk._apply_walk` and
its adjoint partner) and ``psi_in = pole_outflow[mirror[global_n]]`` in
the SI sweep twin (:meth:`~orpheus.sn.loss_representation._OneDimScanWalk.sweep`),
with the pairing derived ONCE per mesh from the :math:`\sigma_x` mirror
motion via
:meth:`~orpheus.numerics.quadrature.Quadrature.ordinate_permutation`.
That single index — ``mirror[n]`` — is what makes the
seed correct, and it carries a load-bearing assumption that is invisible
in the code but essential to the physics.

**The invariant.**  For the mirror seed to realise
:math:`\psi(0,+\mu_r)=\psi(0,-\mu_r)`, the partner
``mirror[n]`` MUST be the *intra-level sign-flip partner*
of ordinate :math:`n`: the ordinate in the **same** :math:`\mu`-level
(same axial cosine :math:`\mu_z` — the level index) with the radial
cosine :math:`\mu_x` negated and :math:`\mu_y,\mu_z` held.

.. math::
   :label: sn-coupled-pole-mu-level-invariant-eq

   m \;=\; \pi_{\sigma_x}(n)
   \;\Longrightarrow\;
   \mu_x[m] = -\,\mu_x[n],\quad
   \mu_y[m] = \mu_y[n],\quad
   \mu_z[m] = \mu_z[n],

where :math:`\pi_{\sigma_x}` is the x-mirror's derived ordinate
permutation (:eq:`quadrature-ordinate-permutation`).

.. (vv-status rationale) Structural / representational identity: the
   defining property the x-mirror partner MUST satisfy for the
   coupled-pole seed to be physically correct (intra-level μ_x
   sign-flip). Not a solver claim (no eigenvalue / flux value). The
   verifiable content is the foundation gate
   test_x_reflection_is_intra_level_signflip_partner (asserts the three
   equalities over gauss_legendre/level_symmetric/product cubatures) +
   its involution sibling; documented-only.
.. vv-status: sn-coupled-pole-mu-level-invariant-eq documented

**Why the physics demands it.**  The pole continuity is a statement at a
*fixed axial direction*: a neutron travelling inward along :math:`-\mu_x`
at axial cosine :math:`\mu_z` reaches the centre and emerges travelling
outward along :math:`+\mu_x` at the **same** :math:`\mu_z`.  The axial
component does not turn at the pole — only the radial one reverses.  So
the reflected partner must stay in the same :math:`\mu`-level; a
cross-level partner would couple two different axial directions and seed
the outward sweep with a value from the *wrong* characteristic.

**Why it holds by construction today.**  Two facts conspire.  First,
the seed derives its pairing (``_ensure_pole_mirror``, once per mesh)
from the :math:`\sigma_x` mirror motion through
:meth:`~orpheus.numerics.quadrature.Quadrature.ordinate_permutation` —
every image matched to a node, bijection, equal weights (the
certification Q5.0.1 introduced; the bare nearest-neighbour
``_find_reflections`` it replaced was ERR-074's site, and the
precomputed ``reflection_index`` table that carried these answers until
G6.3 §7d read the same certificate): the mirror's action negates
**only** :math:`\mu_x` — :math:`\mu_y,\mu_z` are passed through
unchanged.  Second, the cylinder/sphere level is grouped on the
**axial** cosine: the level factories key ``level_indices`` on
:math:`|\mu_z|` (sphere / level-symmetric — ``rules_sphere.py``) or hold
:math:`\mu_z=\mu_{\rm GL}` fixed per level (product — ``rules_product.py``),
never on :math:`\mu_x`.  Because the x-mirror holds :math:`\mu_z` and the
level is indexed by :math:`\mu_z`, the x-mirror provably maps an ordinate
to a partner in its own level.  The two facts are *independent* code
sites, so the invariant is an emergent property of their agreement — not
something either site enforces alone.

.. warning::

   **This is a silent-corruption surface — a Mode-7 blind spot at the
   operator-internals level.**  If the derived pairing ever
   returned a **cross-level** partner — a future cubature, or a
   refactor of
   :meth:`~orpheus.numerics.quadrature.Quadrature.ordinate_permutation`'s
   match machinery that no longer holds :math:`\mu_z` — then
   ``pole_outflow[mirror[n]]`` would read a *different axial direction's*
   pole value, and the break would be **completely silent under the
   existing solver suite**.  The reason is the same blindness that hid
   ERR-058 itself: on a spatially/angularly **flat** :math:`\psi` field
   the mirror partner's pole value equals the ordinate's own value, so
   the seed is exact regardless of *which* ordinate it reads.  Every
   flat-flux gate, every streaming-equilibrium L0, every reflective
   :math:`k_\infty` check would stay green while the operator quietly
   coupled the wrong characteristics on any non-flat field (``vv-principles``
   Mode 7 — the ansatz-simplification blindness — operating on the
   operator's own internals, exactly the ERR-058 class).  A scalar /
   particle-balance residual cannot see it either, because the
   :math:`\alpha`-dome telescoping (above) sums away per-ordinate seed
   errors.

**Why it now has its own foundation gate.**  Because the solver tests are
structurally blind to a cross-level regression, the invariant is pinned
*directly* — not through any flux or eigenvalue, but as a property of the
derived mirror pairing itself — by the foundation test
:func:`tests.sn.sweep.curvilinear.test_coupled_pole_mu_level_invariant.test_x_reflection_is_intra_level_signflip_partner`.
It asserts all three equalities of
:eq:`sn-coupled-pole-mu-level-invariant-eq` (intra-level membership,
:math:`\mu_x` sign-flip, :math:`\mu_y,\mu_z` held) over the
``gauss_legendre`` / ``level_symmetric`` / ``product`` cubatures the
curvilinear sweep actually uses; the sibling
:func:`~tests.sn.sweep.curvilinear.test_coupled_pole_mu_level_invariant.test_x_reflection_is_an_involution`
pins ``mirror ∘ mirror = id`` (the partner relation is symmetric — a
necessary corollary of the sign-flip).  Both are
``@pytest.mark.foundation``; the first carries
``@pytest.mark.verifies("sn-err-058-coupled-pole-continuity")``, tying
the table-level invariant to the continuity equation it underwrites.
This gate is the regression tripwire that turns the silent-corruption
surface into a loud one: a cross-level reflection table fails it
immediately, *before* any solver ever runs.

.. note::

   **Re-scope of Issue #193.**  This invariant is what
   `Issue #193 <https://github.com/deOliveira-R/ORPHEUS/issues/193>`_
   now pins.  The issue *originally* targeted a different "level-locality"
   concern — that the cylindrical matvec's
   ``bc_outer.apply``-once-then-per-level-extract pattern stayed correct.
   That concern **dissolved**: Wave O O.4a.2 removed the ``bc_outer.apply``
   keystone from the matvec entirely (the reflective coupling :math:`B`
   moved *outside* the bare sweep as a first-class sibling — see the
   boundary-condition extraction record at :ref:`bc-extraction`), and the
   surviving SI-sweep seed reads the **raw** inflow trace with no
   ``apply`` at all, so the
   apply/restrict commutativity the original test would have guarded is
   now vacuous.  The genuinely load-bearing :math:`\mu`-level invariant
   *moved* to the coupled-pole seed mirror documented here, and that is
   what #193 was re-scoped to gate.

.. _sn-err-058-manifestation-b:

Manifestation (b) — the angular half-angle thread seed (the dominant defect)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Morel–Montry **weighted** angular recurrence
(:cite:`MorelMontry1984`; implemented form :cite:`BaileyMorelChang2010`
Eqs. (42)/(43) — *not* Hébert, whose Eqs. 3.437/3.439 are the plain
:math:`\tau \equiv \tfrac12` diamond; see
:ref:`sn-tau-source-of-record`)
threads the half-angle face fluxes
:math:`\psi_{m\pm 1/2,i}` across a :math:`\mu`-level and needs a starting
seed :math:`\psi_{1/2,i}` at the level's most-inward angular edge.  The
Phase D / Phase F ``CarlsonInwardSweep`` solved the canonical
*sweep-side* starting-direction ODE (Hébert Eqs. 3.432–3.435) for that
seed, but drove it with the **proxy source**

.. math::
   :label: sn-err-058-proxy-source

   \bar Q_i \;=\; \frac{\Sigma_{t,i}\,\phi_{0,i}}{\sum_n w_n},
   \qquad \phi_{0,i} = \sum_n w_n\,\psi_{n,i},

.. (vv-status rationale) Literature-transcribed definition of the
   falsified proxy source (the CarlsonInwardSweep half-angle seed).
   Recorded as the diagnosed defect, not a solver claim. Its falsity
   is what the per-ordinate operator-admission gate (catches ERR-058)
   detects; documented-only.
.. vv-status: sn-err-058-proxy-source documented

which equals the true within-group source ONLY at the flat-flux
equilibrium :math:`\Sigma_t\phi_0 = \bar Q`.  On any non-equilibrium
field — every MMS reference, every vacuum or heterogeneous problem —
the seed solves the *wrong* starting-direction ODE.  The measured
consequence on the isotropic MMS input
(:math:`\psi_n = A(r)/W`, scalar value :math:`0.5`): the seed returns
:math:`\bar\phi = 0.5777` where the correct angle-flat value is
:math:`0.5000`, and the per-ordinate redistribution residual reaches
:math:`\pm 55` at the pole, :math:`\pm 13` in the bulk, against a
continuous streaming of :math:`\pm 0.31`.  **This was the dominant
defect.**

**The fix —**
``AngularEdgeExtrapolation``.
For the *operator* (matvec) to be consistent, the seed must approximate
the *input field's* own value at the level edge :math:`\mu_{\rm start}`
— a pure angular-extrapolation problem with NO dependence on
:math:`\Sigma_t`, the source, or the boundary trace.  The new strategy
extrapolates linearly in :math:`\mu` through the level's two most-inward
distinct-:math:`\mu` ordinates :math:`(m_0, m_1)`:

.. math::
   :label: sn-err-058-edge-extrapolation

   \psi_{1/2,i} \;=\; (1-t)\,\psi_{m_0,i} + t\,\psi_{m_1,i},
   \qquad
   t \;=\; \frac{\mu_{\rm start} - \mu_{m_0}}{\mu_{m_1} - \mu_{m_0}}.

.. (vv-status rationale) Representational identity: the
   operator-consistent half-angle thread seed (AngularEdgeExtrapolation,
   the new psi_half_seed default) as a fixed linear map. Not a solver
   claim. The verifiable content is the per-ordinate operator-admission
   gate (catches ERR-058), the isotropic MMS L1 ladders, and the
   strategy-owned seed-adjoint bit-identity; documented-only.
.. vv-status: sn-err-058-edge-extrapolation documented

The starting-direction edge :math:`\mu_{\rm start}` (sphere
:math:`-1`; cylinder :math:`-\sqrt{1-\xi_p^2}`, the level's most-inward
azimuthal edge) is single-sourced from the SAME
:math:`\alpha`/:math:`\tau` construction site as the
:math:`\alpha`-dome, and read by every consumer from that one owner:
:attr:`~orpheus.sn.angular.redistribution.AngularRedistribution.mu_start_per_level`.
A forgotten cylinder site cannot silently fall back to the sphere value.

.. note::

   That guarantee was **strengthened** on 2026-08-26, and the mechanism
   named in earlier revisions of this paragraph is gone.  It used to read
   "threaded to the strategy via the REQUIRED ``CarlsonSweepContext.mu_start``
   field — no default".  Two things happened.  ``CarlsonSweepContext``
   was retired (its work is
   :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`'s);
   and the un-weld gave :math:`\mu_{\rm start}` **one owner** on the
   angular factor, after which the thread through ``StreamingTerms`` and
   ``StreamingCoefficientCache`` was found to be dead — `[M]` its terminal had
   zero readers — and retired.

   The ERR-058 property is unchanged and now rests on something stronger
   than a convention: not "a required field with no default", but a single
   non-optional owner that every consumer reads.  There is no second place
   for a sphere value to be defaulted from.

**Exactness ladder.**  The extrapolation is

* **exact on angle-flat fields**, because the barycentric weights sum
  to one: :math:`(1-t)+t=1`.  Every per-ordinate flat-flux identity
  gate is therefore untouched.
* **exact on linear-in-:math:`\mu` fields**: write the level's input as
  :math:`\psi_{m,i}=a_i+b_i\,\mu_m`.  Then

  .. math::

     \psi_{1/2,i}
       &= (1-t)(a_i + b_i\mu_{m_0}) + t(a_i + b_i\mu_{m_1}) \\
       &= a_i + b_i\bigl[(1-t)\mu_{m_0} + t\mu_{m_1}\bigr]
        = a_i + b_i\,\mu_{\rm start},

  the last bracket collapsing to :math:`\mu_{\rm start}` by the
  definition of :math:`t` in :eq:`sn-err-058-edge-extrapolation`.  The
  M-M recurrence is itself a Möbius/affine map in :math:`\mu`; seeded
  with :math:`\psi_{1/2}=a+b\,\mu_{1/2}` it threads the ENTIRE
  half-angle grid exactly as :math:`\psi_{m+1/2}=a+b\,\mu_{m+1/2}` (for
  *unclamped* :math:`\tau`).  Hence the P1-class anisotropic MMS
  references — whose ansatz is exactly :math:`(A(r)+B(r)\mu)/W` — are
  *admitted* by the operator.
* **O(\Delta\mu^2)-consistent** on general smooth angular profiles —
  the same order as the angular discretisation itself.
* **linear in the input**, so the operator-algebra operations
  (:meth:`apply`, :meth:`apply_transpose`, dense probing) are
  preserved.  The strategy OWNS its adjoint
  (``PsiHalfAngleSeedBase.seed_adjoint``),
  a fixed linear scatter of the seed cotangent onto the two stencil
  ordinates, so a strategy swap on
  :class:`MorelMontryAngularSweep` swaps both the forward and reverse
  maps at once.

.. note:: **The clamp that used to spoil this is gone (Q5.6.4,
   2026-08-11).**

   The cylinder's :math:`\tau`-absorber
   (:math:`\tau \to \max(0.5,\min(1.0,\tau))`) broke the *exact*
   linear-in-:math:`\mu` threading wherever it was active, contributing a
   constant to the residual anisotropic angular floor below — never a
   wrong-fixed-point defect.  It is **retired in both arms** (sphere at
   W1, 2026-06-13; cylinder at Q5.6.4), so the parenthetical
   "*(for unclamped* :math:`\tau`)" above is now unconditional.

   ⚠ It was **never** Bailey–Morel–Chang's: an earlier revision of this
   note attributed the clamp to them, and they prescribe no limiter at
   all — their admissible range is :math:`[0, 1]` and their own
   :math:`S_2` example gives :math:`\tau_1 \approx 0.4226 < \tfrac12`.
   See :ref:`sn-tau-absorber-retirement` in
   :doc:`/theory/foundations/structured_geometry`.

Why every gate stayed green — the blindness analysis
-------------------------------------------------------

Both seeds hid behind a regime in which they are exact, and the V&V
suite sat *entirely inside* that regime.  This is
``vv-principles`` Mode 7 (MMS / ansatz simplification bias) operating
not on a manufactured solution but on the *operator's own internals*.

.. list-table:: Which fields each closure seed is exact on (the blind regime)
   :header-rows: 1
   :widths: 30 34 36

   * - Closure seed
     - Exact on
     - Gate that sat in the blind regime
   * - Spatial pole-face (a)
     - flat radial :math:`\psi`
       (:math:`\psi(\Delta r/2)=\psi(0)`)
     - streaming-equilibrium L0; reflective-equilibrium k\ :sub:`∞`
   * - Angular thread (b)
     - flat-flux equilibrium
       (:math:`\Sigma_t\phi_0 = \bar Q`)
     - per-ordinate flat-flux identity (Gate 1.1); homogeneous
       reflective

**The :math:`\alpha`-dome telescoping made scalar checks blind to
(b).**  The M-M redistribution coefficients form a dome that telescopes
under the angular weight sum: :math:`\sum_n w_n\,(\alpha_{n+1/2} -
\alpha_{n-1/2}) = 0` REGARDLESS of the half-angle thread values.  Any
weight-summed (scalar-flux / particle-balance) residual therefore cannot
see a wrong half-angle thread — ``vv-principles`` anti-pattern #8
("NEVER accept particle balance as L0 evidence; require per-ordinate
residual") instantiated *inside a diagnostic*.  During the #195
investigation this telescoping made the scalar residual go
:math:`\mathcal{O}(h^2)` after fixing only (a), while the per-ordinate
residual was still :math:`\mathcal{O}(10)` — which mis-supported a
"near-singular operator / two-solutions gauge mode" hypothesis until a
dense SVD showed :math:`\sigma_{\min}\approx 0.9` (never near-singular)
and the *per-ordinate* check named the real defect.

**Historical compensation explains the Phase-D-era O(h²) reading.**  At
Phase D time the apply path measured :math:`\mathcal{O}(h^2)` under
Krylov (the premise this issue inherited), because its closure internals
compensated differently from the sweep.  The Depth-B/Wave-T matvec
rebuild changed the redistribution assembly and surfaced the latent seed
inconsistency; the SWEEP, by contrast, had ALWAYS plateaued (#195's own
SI data :math:`[0.083, 0.095, 0.098]`).  The wrong fixed point was the
sweep's all along — the same class as #98's original 35 %-at-:math:`r=0`
finding.

The three refuted intermediate hypotheses
--------------------------------------------

Recording the dead paths so a future session does not re-run them
(Sphinx is the brain):

#. **"Pre-asymptotic transient"** (the issue's own premise).  Refuted by
   the :math:`n_x` 20 :math:`\to` 640 plateau — orders :math:`\to 0`, no
   refinement helps.
#. **A pure :math:`r=0`-regularity extrapolation spatial seed**
   (:math:`1.5\,\psi_0 - 0.5\,\psi_1`).  Implemented; it drove the
   *scalar* residual to :math:`\mathcal{O}(h^2)` but the solution still
   plateaued, because the dominant defect was the *angular* seed (b),
   invisible to the scalar residual by the telescoping above.  Superseded
   by the coupled-pole seed :eq:`sn-err-058-coupled-pole-continuity` for
   (a) — which is *data* rather than a one-sided stencil — once (b) was
   diagnosed.
#. **A "near-null gauge mode" theory** (apparent two-solutions paradox).
   Falsified by a dense SVD: :math:`\sigma_{\min}\approx 0.9`, the
   operator is well-conditioned.  The paradox was an artefact of the
   scalar-blind diagnostic, not a property of the operator.

Production closure decision — post-fix evidence
-------------------------------------------------

Post-fix (measured 2026-06-12), the isotropic curvilinear MMS solution
error collapses into a clean second-order ladder, with SI and Krylov
bit-identical:

.. list-table:: Post-fix isotropic curvilinear MMS L2 ladders (SI ≡ Krylov)
   :header-rows: 1
   :widths: 16 16 16 16 16 16

   * - :math:`n_x`
     - 20
     - 40
     - 80
     - 160
     - 320
   * - sphere :math:`\|\phi_h-A\|_{L^2}`
     - 1.49e-2
     - 3.73e-3
     - 9.28e-4
     - 2.31e-4
     - 5.74e-5
   * - sphere order
     -
     - 2.00
     - 2.01
     - 2.01
     - 2.01
   * - cylinder :math:`\|\phi_h-A\|_{L^2}`
     - 2.16e-3
     - 5.39e-4
     - 1.35e-4
     - 3.37e-5
     -
   * - cylinder order
     -
     - 2.00
     - 2.00
     - 2.00
     -

The magnitude band :math:`10^{-8} < {\rm err} < 10^{-3}` is satisfied
(sphere :math:`n_x \ge 80`, cylinder :math:`n_x \ge 40`).  SI converges
:math:`\sim 10^2\times` faster than GMRES here (sphere :math:`n_x=160`:
:math:`\sim 0.11\,\mathrm{s}` SI vs :math:`\sim 69\,\mathrm{s}` Krylov),
which is why the curvilinear default returned to source iteration.

The decisive *structural* gate is the **per-ordinate, volume-weighted**
operator-admission residual of :math:`\psi_{\rm ref}` (the scalar
residual is blind, per the telescoping above).  The four-term operator
below is exact for this fixture: every MMS mixture is built with
:math:`\Sigma_{2n} \equiv 0`
(:mod:`orpheus.derivations.continuous.mms.sn` mints
``Sig2 = csr_matrix(zeros((ng, ng)))``), so the shipped fifth member
:math:`N_{2n}` of :eq:`sn-within-group-with-n2n` contributes exactly
nothing to the residuals tabulated here:

.. list-table:: Per-ordinate volume-weighted residual of ψ_ref under (L+C−S−B), post-fix
   :header-rows: 1
   :widths: 25 25 25 25

   * - Geometry
     - :math:`n_c=40`
     - :math:`n_c=80`
     - measured order
   * - sphere
     - 1.97e-3
     - 9.7e-4
     - :math:`\approx 1.5` (pole-adjacent bounded band under
       the :math:`r^2\,dr` weight)
   * - cylinder
     - 5.50e-5
     - 1.37e-5
     - :math:`\approx 2.0` (pointwise :math:`\mathcal{O}(h^2)`
       everywhere)

The sphere's sub-quadratic residual order is benign: the
pole-adjacent cells legitimately carry a bounded non-decaying
*pointwise* residual where the closure truncation meets the
:math:`\Delta A/V \sim 1/h` geometry factor on cells whose volume
vanishes as :math:`r^2\,dr`; the solution-error ladder above proves it
harmless.  **Bug-era** values for this gate were :math:`\mathcal{O}(10^{-1})`-class
(per-ordinate pointwise up to :math:`\pm 55` at the pole) — three-plus
orders of magnitude above the post-fix bounds, which is the margin the
ERR-058 catcher asserts.

The quadrature/truncation floor is the radial DD closure order itself;
the post-fix sphere/cylinder ladders sit at the DD design order
(2.00–2.01), so "have you tried finer quadrature?" is pre-empted — the
solution-error *is* second-order, and the only residual non-quadratic
behaviour is the volume-weighted pole band, which the solution error
does not inherit.

.. _sn-err-058-aniso-floor:

The anisotropic angular floor — deferred to Issue #229
--------------------------------------------------------

.. note:: **Resolved (2026-06-13, W1–W5).**

   This deferral is now fully treated at
   :ref:`sn-curvilinear-aniso-norm-reconciliation`.  The W1–W5
   root-cause program found the single "floor" sketched below was
   **three distinct errors** (a sphere pole-cell spatial closure, a
   sphere angular :math:`\tau`-clamp floor, and a cylinder angular
   floor), separated by a norm difference (volume-weighted L2 vs
   pointwise / L∞).  In particular, the "Open research paths" item
   below — "Unclamped-:math:`\tau` threading on a linear-in-:math:`\mu`
   shell" — was **executed** (W1): the sphere clamp was removed (it was
   mis-cited and 100 % spurious; see
   :ref:`sn-tau-clamp-vindication`), which cleaned the coarse rate but,
   surprisingly, *raised* the S16 fine floor (the prior lower floor was
   a fortuitous cancellation, not a gain).  The **cylinder** half
   followed at Q5.6.4 (2026-08-11), with the same accuracy signature and
   a deeper cause — a wrong angular cell partition the absorber was
   compensating; see the same research-path item below.  The numbers
   preserved below
   are correct historical evidence; the comprehensive treatment and the
   per-error production decisions are at the reconciliation section.

The **anisotropic** curvilinear MMS (ansatz :math:`(A(r)+B(r)\mu)/W`)
dropped :math:`\sim 50\times` under the fix and is now limited by a
*fixed-quadrature angular floor*, NOT by a residual wrong-fixed-point
defect.  The mechanism: the aniso MMS imposes :math:`\psi_n` per
ordinate, so there is no angular error *at the imposed ordinates* — but
the M-M redistribution consumes half-angle THREAD values
:math:`\psi_{m\pm 1/2}` that the recurrence *interpolates*.  On an
angle-varying ansatz the thread's interpolation error is an
angular-quadrature-resolution effect: under spatial refinement at fixed
quadrature the solution converges to an angular floor, and the
pure-spatial rate + magnitude assertions cannot both hold once the
spatial error drops below it.  The floor *scales with quadrature order*
in both geometries — the structural signature confirming the
angular-thread attribution:

.. list-table:: Anisotropic angular floor vs quadrature order (post-ERR-058, SI inner)
   :header-rows: 1
   :widths: 22 24 54

   * - Case
     - Quadrature
     - Floor behaviour
   * - sphere aniso
     - S16 (shipped)
     - :math:`n_x` 10→160: ``[5.9e-2, 1.5e-2, 4.0e-3, 1.15e-3, 7.3e-4]``;
       floor :math:`\approx 7\mathrm{e}{-4}`
   * - sphere aniso
     - S32
     - err@80 = 9.5e-4, err@160 = 2.9e-4 (floor drops :math:`\sim 2.5\times`)
   * - cylinder aniso
     - :math:`n_\mu{=}4` (shipped)
     - :math:`n_x` 40→80: ``1.91e-2 → 1.90e-2`` (hard floor 1.9e-2)
   * - cylinder aniso
     - :math:`n_\mu{=}8`
     - :math:`n_x` 40→80: ``7.50e-3 → 7.39e-3``
       (floor drops :math:`\sim 2.6\times` per :math:`n_\mu` doubling)

The :math:`\tau`-clamp (above) contributed a constant to this floor by
breaking the exact linear-in-:math:`\mu` threading where active — `[M]`
quantified at Q5.6.4 and **retired** in both arms (see the note above).
The quadrature-aware retune (raise the case quadrature, or split the
claim into a pre-floor spatial-O(h²) segment + a separate
angular-convergence assertion) landed as W3 under `Issue #229
<https://github.com/deOliveira-R/ORPHEUS/issues/229>`_, **now CLOSED**
(2026-06-13) — that issue is the measurement record for this floor, not
an open work item.

Infrastructure retained
-------------------------

Per the aggressive-retirement *exception* (a correct primitive that
would be needed if the obstruction is ever bypassed is kept as an
oracle), ERR-058 deletes no correct machinery:

.. list-table:: Curvilinear closure-seed primitives — status after ERR-058
   :header-rows: 1
   :widths: 38 18 44

   * - Primitive
     - Status
     - Why kept
   * - ``AngularEdgeExtrapolation``
     - **production**
     - The new ``psi_half_seed`` default; operator-consistent on
       non-flat fields.
   * - ``CarlsonInwardSweep``
       + :func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`
     - retained, opt-in
     - Correct *source-driven* Hébert §3.9.4 recurrence; would seed a
       future TRUE-source sweep-side closure.  Proxy-source caveat
       pinned in its docstring ``warning`` block.
   * - ``ZeroSeed``
     - retained, ablation
     - Reproduces the Phase B behaviour for A/B regression-safety
       comparison.
   * - Coupled-pole spatial seed in
       :meth:`~orpheus.sn.loss_representation._OneDimScanWalk._apply_walk`
       / :meth:`~orpheus.sn.loss_representation._OneDimScanWalk.sweep`
     - **production**
     - The :math:`\psi(0,+\mu)=\psi(0,-\mu)` continuity; matvec + sweep
       share it (one discrete system).

.. note:: **Retraction (2026-07-04, Issue #282 route (a)).**  The three
   ``PsiHalfAngleSeed`` strategy rows above (``AngularEdgeExtrapolation``
   / ``CarlsonInwardSweep`` / ``ZeroSeed``) are superseded: route (a)
   **deleted** the whole strategy family — including the
   ``AngularEdgeExtrapolation`` "production default", which was itself
   the #282 walk-order back edge.  What is genuinely retained is the
   free-function Hébert engine
   :func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`
   (now the SOLVE driver, on the **true** q½ source) and the inlined
   :meth:`~orpheus.sn.angular.closure.MorelMontryAngularSweep.edge_extrapolated_seed`
   (non-carrying cylinder levels — unconstructible through ``SNMesh``
   since the Q5.6.3 admission).  The coupled-pole spatial seed row is
   unaffected.  See :ref:`sn-direct-seed-strategy-zoo`.

Open research paths
---------------------

Two paths could lift the anisotropic angular floor without changing the
isotropic O(h²) result:

#. **TRUE-source-driven sweep-side seed** — **LANDED as Issue #282 route
   (a) (2026-07-04).**  This path predicted the resolution exactly:
   replace the ``AngularEdgeExtrapolation`` *input-field* extrapolation
   with the canonical Hébert recurrence
   :func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`
   driven by the genuine within-group source
   :math:`\bar Q_i = \sum_\ell \tfrac{2\ell+1}{2}Q_\ell(r_i)(-1)^\ell`
   (the full Legendre fold, not the :math:`\Sigma_t\phi_0` proxy), making
   the *starting-direction transport* exact rather than the *input-field
   value* exact.  Route (a) shipped precisely this — and the "likely
   diagnostic probe" proposed here (holding spatial mesh fixed and
   sweeping quadrature order) is exactly the **angular-order N-sweep**
   that certified the re-pose principled.  The one refinement over the
   prediction: the seed also had to become **first-class typed state**
   (not just a better strategy) to kill the walk-order back edge that
   made the *solve* non-direct.  See
   :ref:`sn-direct-seed-solve`.
#. **Unclamped-:math:`\tau` threading on a linear-in-:math:`\mu` shell**
   — **LANDED, and the prediction was half right** (sphere at W1
   2026-06-13; cylinder at Q5.6.4 2026-08-11).  The path as written read:
   *"quantify the clamp's contribution to the floor and, where the cell
   is well-resolved (*\ :math:`\tau_{\rm raw}\in[0.5,1.0]`\ *), thread
   unclamped to recover the exact P1 admission.  Likely probe: the floor
   table with the clamp disabled on resolved cells."*

   The probe was run exactly as proposed, and it **refuted the expected
   outcome**: disabling the clamp on the then-current cylinder partition
   made the anisotropic floor :math:`1.8`--:math:`3.4\times` **worse**,
   not better.  Two things were wrong with the framing, and finding them
   is the actual result:

   * :math:`[0.5, 1.0]` **is not a "well-resolved" criterion.**  The
     admissible range of :math:`\tau` is :math:`[0, 1]` (predicate P3);
     :math:`[\tfrac12, 1]` was the absorber's own box, with no source
     behind it.
   * **The clamp was compensating a wrong angular cell partition**, not
     merely truncating a resolved value.  The cylinder's edges were taken
     at the *chord* midpoint while :math:`\alpha` used the real
     half-angle — a permanent :math:`\approx 17.5\,\%` disagreement in
     :math:`\omega`-width.  Taking the partition in :math:`\omega`
     removes the disagreement and leaves nothing to clamp.

   ⟹ the exact P1 admission is recovered, and the honest cost is stated
   rather than hidden: the principled weight is
   :math:`\sim 1.8`--:math:`2\times` *worse* in L2 at
   :math:`n_\varphi \ge 16` on this MMS, because the L2 norm measures
   truncation order — the thing :math:`\tau \equiv \tfrac12` optimises and
   the thing that is blind to the diffusion limit :math:`\tau` exists to
   fix.  Full account: :ref:`sn-tau-absorber-retirement` in
   :doc:`/theory/foundations/structured_geometry`.

Session trail (V&V audit trail)
---------------------------------

* **ERR-058 catalogue narrative**:
  ``docs/theory/verification/error_catalog.rst`` (§ ERR-058) — the
  authoritative two-manifestation mechanism + post-fix evidence.
* **Re-scope record**: `Issue #195
  <https://github.com/deOliveira-R/ORPHEUS/issues/195>`_ comments
  (2026-06-12) — the premise refutation and the decisive probe-3
  residual evidence.
* **Diagnostics**: ``derivations/diagnostics/diag_195_probe{1,2,3}_*.py``
  (the plateau / error-profile / operator-admission probes), promoted to
  the gate ``tests/sn/verification/mms/test_curvilinear_operator_admits_mms.py``.
* **Investigator memo**:
  ``.claude/agent-memory/numerics-investigator/issue_195_root_cause_2_pole_closure.md``.

Verification chain
-------------------

The ERR-058 fix is pinned by, in order of structural decisiveness:

#. :func:`tests.sn.verification.mms.test_curvilinear_operator_admits_mms.test_operator_admits_isotropic_mms_per_ordinate`
   (``@pytest.mark.l1`` + ``catches("ERR-058")``) — the fast
   per-ordinate volume-weighted operator-admission gate (the structurally
   decisive check, immune to the telescoping blindness).
#. :func:`tests.sn.verification.mms.test_mms_curvilinear.test_sn_spherical_mms_converges_second_order`
   and
   :func:`tests.sn.verification.mms.test_mms_curvilinear.test_sn_cylindrical_mms_converges_second_order`
   (``catches("ERR-058")``) — the end-to-end L1 ladders whose
   ``xfail`` markers came off with this fix; they ``verifies`` the
   :eq:`sn-mms-spherical-psi` / :eq:`sn-mms-spherical-qext` /
   :eq:`sn-mms-cylindrical-psi` / :eq:`sn-mms-cylindrical-qext` labels.
#. The flat-flux and streaming-equilibrium gates pin the flat-field
   exactness BOTH fixes preserve (so they did not regress).
#. :func:`tests.sn.operators.test_g_adjoint_reciprocity` — pins the
   strategy-owned seed adjoints.

.. note::

   **vv-status (eq-labels added by this section).**  The labels
   :eq:`sn-err-058-coupled-pole-continuity`,
   :eq:`sn-err-058-proxy-source`, and
   :eq:`sn-err-058-edge-extrapolation` are *structural / representational*
   identities (the pole-continuity boundary condition; the falsified
   proxy-source definition; the operator-consistent edge-extrapolation
   map).  They are NOT solver claims (no eigenvalue / flux-value claim).
   Per the vv-status discipline they are ``documented`` — the
   verifiable content is the per-ordinate operator-admission gate
   (``catches("ERR-058")``) plus the strategy-owned adjoint
   bit-identity, named in the verification chain above.

.. _sn-issue-196-eigenvalue-equivalence:

Issue #196 — eigenvalue-path SI≡Krylov verification and the permanent gate
--------------------------------------------------------------------------

.. admonition:: Status — manifestation #7 verified and pinned (Issue #196 CLOSED, 2026-06-13)
   :class: important

   ERR-058 (#195, above) replaced the wrong shared closure seeds; **#196
   is the verification and regression-gate step** that confirms the
   replacement closed ERR-026 manifestation #7 at the *eigenvalue*
   level and locks the closure against re-introduction.  This was the
   LAST open manifestation of ERR-026 — with #196 closed, the
   curvilinear-SN wrong-fixed-point family is **formally retired**.

The two-layer history (why the L0 close did not suffice)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Manifestation #7 names a specific defect: a residual :math:`\mathcal{O}(h)`
SI-vs-Krylov WDD spatial-closure asymmetry on curvilinear SN.  Pre-fix,
the source-iteration inner (drives the curvilinear sweep) and the Krylov
inner (drives the apply-matvec) produced eigenvalues differing at
:math:`\mathcal{O}(h)`: **0.286 %** on ``sphere_2g_3reg`` at n=40, **~30
%** per-cell on eigenvector shape, the gap halving under mesh refinement.

Two closures touched this defect, and the honest history distinguishes
them:

* **ERR-048** (Phase G Step 2, 2026-05-13) closed only the **L0
  flat-field** twin-agreement.  It patched the SI sweep
  (``_sweep_1d_spherical`` / ``_sweep_1d_cylindrical``) to MATCH the
  apply-matvec conventions on the homogeneous streaming-equilibrium
  gauntlet (pole-face WDD IC mirror + Carlson seed :math:`\bar Q`
  normalisation).  The **L1 heterogeneous eigenvalue**
  :math:`\mathcal{O}(h)` asymmetry that manifestation #7 names
  **PERSISTED** — which is exactly why #196 stayed OPEN — because the
  shared closure seeds were still *exact-on-flat /
  O(1)-wrong-on-non-flat* (the ERR-058 defect).
* **ERR-058** (Issue #195, 2026-06-12) was the TERMINAL fix.  It
  replaced the shared closure seeds with correct ones (the coupled-pole
  spatial seed :math:`\psi(0,+\mu)=\psi(0,-\mu)` and the
  ``AngularEdgeExtrapolation``
  half-angle seed), making BOTH inner solvers operate on the SAME
  correct discrete operator.

.. _sn-issue-196-bit-identical-vs-floor:

Bit-identical (fixed-source) vs floor-equivalent (eigenvalue)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The post-fix SI :math:`\equiv` Krylov agreement is **not the same kind
of agreement** on the two solver entry points, and the distinction is
load-bearing:

* **Fixed-source** (:func:`~orpheus.sn.solver.solve_sn_fixed_source` on
  the curvilinear MMS ladders): SI :math:`\equiv` Krylov is
  **BIT-IDENTICAL**.  Post-unification the sweep and the matvec are ONE
  discrete operator on one quadrature; the within-group inner
  (``A.solve`` vs Krylov-on-``apply``) realises the *same* :math:`A^{-1}`
  arithmetic, so the two paths return the same bits.
* **Eigenvalue** (:func:`~orpheus.sn.solver.solve_sn` with
  ``inner_solver="source_iteration"`` vs ``"krylov"``): SI
  :math:`\equiv` Krylov to the **ITERATION FLOOR** (~:math:`1.9\mathrm{e}{-11}`
  in :math:`k_{\text{eff}}`, ~:math:`2.4\mathrm{e}{-10}` in flux shape),
  **NOT bit-identical**.  The eigenvalue solve wraps the inner in power
  iteration; SI and Krylov are *different iteration schemes* that
  converge to the **same correct fixed point** only to ~``inner_tol``.
  Same physics, not the same arithmetic.

Confusing the two would mis-state the verification claim.  The earlier
close-out's "SI :math:`\equiv` Krylov bit-identical on the curvilinear
MMS ladders" is correct **for the fixed-source ladders specifically**;
the eigenvalue verification below is *floor-equivalence*, which is the
right and sufficient claim for an eigenvalue solve.

Measured eigenvalue-path equivalence (Issue #196)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All values measured 2026-06-12 under tight iteration tolerances for
BOTH inner solvers (``keff_tol=1e-12``, ``flux_tol=1e-10``,
``inner_tol=1e-10``).  The eigenvalue snapshot cases are the exact
acceptance cases of #196:

.. list-table:: SI≡Krylov eigenvalue equivalence — the bug-era heterogeneous snapshot cases
   :header-rows: 1
   :widths: 30 22 22 26

   * - Case
     - :math:`|\Delta k|` (post-fix)
     - max :math:`|\Delta\varphi|_{L\infty}` (post-fix)
     - Bug-era (pre-ERR-058)
   * - ``sphere_2g_3reg_dd_n40``
     - :math:`4.68\mathrm{e}{-12}`
     - :math:`5.88\mathrm{e}{-11}`
     - :math:`|\Delta k|=3.9\mathrm{e}{-3}` (0.286 %), shape ~30 %
   * - ``cyl_2g_3reg_LS4_dd_n40``
     - :math:`1.91\mathrm{e}{-11}`
     - :math:`4.32\mathrm{e}{-11}`
     - same :math:`\mathcal{O}(h)` family, gap halving under refinement

The homogeneous (k_inf-degenerate, flat-flux) curvilinear snapshots
agree at the rounding floor — as expected, since on a flat eigenmode the
redistribution terms null and SI/Krylov differ only by accumulated
round-off:

.. list-table:: SI≡Krylov eigenvalue equivalence — homogeneous (flat-flux) snapshots
   :header-rows: 1
   :widths: 38 26 26

   * - Case
     - :math:`|\Delta k|`
     - relative :math:`\varphi` diff
   * - ``sphere_2g_homogeneous_dd_n20``
     - :math:`6.92\mathrm{e}{-13}`
     - :math:`2.15\mathrm{e}{-10}`
   * - ``cyl_1g_homogeneous_LS4_dd_n20``
     - :math:`2.22\mathrm{e}{-16}`
     - :math:`2.27\mathrm{e}{-11}`
   * - ``cyl_1g_homogeneous_product_dd_n20``
     - :math:`6.66\mathrm{e}{-16}`
     - :math:`1.10\mathrm{e}{-10}`

.. note::

   The homogeneous cases agree to the floor but, on their own, supply
   **no** evidence for the curvilinear closure — a flat eigenmode is
   degenerate (``flat = flat``; 1-group :math:`k=\nu\Sigma_f/\Sigma_a`
   is flux-shape independent, vv-principles anti-patterns #3/#4).  The
   load-bearing evidence is the **heterogeneous 2-group** cases above,
   where the flux is genuinely non-flat and the angular-redistribution
   terms are exercised.

The permanent regression gate (Issue #196)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The manifestation-#7 catcher is
:func:`tests.sn.eigenvalue.test_keff_curvilinear.test_si_krylov_eigenvalue_equivalence_sphere`
and
:func:`tests.sn.eigenvalue.test_keff_curvilinear.test_si_krylov_eigenvalue_equivalence_cylinder`,
each carrying ``@pytest.mark.catches("ERR-026")``.  Configuration:
heterogeneous 2-group fuel|moderator (region A inner, region B outer,
n=10+10), solved twice under ``inner_solver="source_iteration"`` and
``inner_solver="krylov"`` at the tight tolerances above.  The gate
asserts:

.. list-table:: Manifestation-#7 gate thresholds vs measured vs bug-era
   :header-rows: 1
   :widths: 26 24 24 26

   * - Quantity
     - Asserted bound
     - Measured (post-fix)
     - Bug-era (would trip)
   * - sphere :math:`|\Delta k|`
     - :math:`< 1\mathrm{e}{-7}`
     - :math:`1.9\mathrm{e}{-11}`
     - :math:`3.9\mathrm{e}{-3}`
   * - sphere per-group :math:`|\Delta\varphi|_{L\infty}`
     - :math:`< 1\mathrm{e}{-6}`
     - :math:`2.4\mathrm{e}{-10}`
     - ~30 %
   * - cylinder :math:`|\Delta k|`
     - :math:`< 1\mathrm{e}{-7}`
     - :math:`1.1\mathrm{e}{-11}`
     - same family
   * - cylinder per-group :math:`|\Delta\varphi|_{L\infty}`
     - :math:`< 1\mathrm{e}{-6}`
     - :math:`2.6\mathrm{e}{-11}`
     - ~30 %

A **non-flat-flux guard** (group-0 radial ``max/min > 1.2``) precedes
the equivalence assertion so the test cannot pass vacuously on a
degenerate flat mode — sphere radial ``max/min`` = 3.34, cylinder =
1.67, both well above the guard.  The bug-era values (3.9e-3 / ~30 %)
exceed the asserted bounds by **4–5 orders of magnitude**, so the gate
would have tripped loudly on the pre-fix code.

.. note::

   **Runtime-mode discipline (vv-principles anti-pattern #8).**  The
   canonical ORPHEUS invocation is ``python -O``, under which bare
   ``assert`` statements are stripped to no-ops.  These gates assert via
   bare ``assert`` inside the *collected test module*, which pytest
   rewrites at collection time so the asserts fire under ``-O``.  This
   was confirmed empirically: a negative control with a
   :math:`1\mathrm{e}{-15}` tolerance failed as required under ``-O``.

Structural-independence — why SI≡Krylov alone is not the whole proof
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SI :math:`\equiv` Krylov is **twin-path agreement** — necessary but not
sufficient (vv-principles L11: two implementations agreeing is
cross-implementation evidence, not correctness evidence).  Both inner
solvers could in principle converge to the same *wrong* fixed point.
The independent ground that makes the closure a *correctness* claim, not
merely a *consistency* claim, comes from two structurally-independent
legs:

* The **k_inf homogeneous legs** — on a uniform reflective infinite
  medium :math:`k_\infty=\nu\Sigma_f/\Sigma_a` is an analytical
  (closed-form) eigenvalue the SN snapshots must reproduce.
* The **Variant-α Green's-function cross-check**
  (:func:`tests.sn.verification.analytical.test_phase_c_crosscheck.test_phase_e_trajectory_resolvent_flux_shape_crosscheck`),
  now a plain L1 test (xfail removed), which compares the SN flux-shape
  snapshot against the composite-GL trajectory-resolvent reference
  within 8 % (sphere) / 12 % (cylinder).  This reference is a
  semi-analytical pillar structurally independent of the SN sweep, so
  agreement pins the *converged-to value*, not just twin-path
  consistency.

Production-decision record — curvilinear default reverted to SI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The curvilinear ``inner_solver`` default on
:class:`~orpheus.sn.solver.SNSolver`
is now ``"source_iteration"``, **reverted from the Phase-D Krylov
flip**.  The Phase-D flip existed ONLY because the sweep's fixed point
was wrong; ERR-058 made it correct, so SI is restored as the default —
it is :math:`\sim 10^2\times` faster than GMRES (no restart) and now
equivalent (bit-identical fixed-source / floor-equivalent eigenvalue).

Crucially, **neither of the old Phase-F closures was taken**:

* *Option (a) — make SI bit-identical to Krylov by refining the WDD
  closure* presupposed the seed was correct and only the spatial
  arithmetic differed.
* *Option (b) — flip the default to Krylov* presupposed the Krylov fixed
  point was the correct one and SI's was a discretisation artefact.

Both presupposed the *shared* fixed point was correct and only the
arithmetic differed.  The terminal diagnosis (ERR-058) showed the shared
fixed point **itself** was wrong on non-flat fields; once the seeds were
fixed, both inner solvers reach the same *correct* fixed point and the
"choose between them" framing dissolves — SI is restored on speed alone.
