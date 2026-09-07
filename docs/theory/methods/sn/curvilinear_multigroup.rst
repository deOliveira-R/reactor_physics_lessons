.. _sn-curvilinear-multigroup:

Curvilinear multigroup: the group axis rides the walk
=====================================================

This chapter closes Part B's broadening progression: **energy** joins
the curvilinear machinery of :doc:`curvilinear_one_group`, exactly the
axis :doc:`slab_multigroup` added to :doc:`slab_one_group`.  And its
headline is a deliberate anticlimax: **there is nothing to derive.**
The group axis couples the transport equations *only through their
sources* — that was the structural fact of :doc:`slab_multigroup` —
and everything curvature added in the previous chapter is streaming:
the redistribution cascade, the geometry factor, the :term:`starting-direction <starting direction>`
state are all *in-flight* physics.  A neutron flying through a curved
coordinate system changes its local direction coordinate; it does not
change its energy.  Group transfer happens at collision events, and
collision events live in :math:`S` and :math:`F` — which curvature
never touched.

That "nothing" is this chapter's content, because it is a *claim about
the architecture*, not an absence of one: the curvilinear walk is a
geometry :math:`\times` :term:`quadrature` object, and energy is a data axis
that broadcasts along it.  The claim is checkable — arrays either
carry a group axis or cannot — and this chapter states exactly which
do, where the axis enters, and which tests pin the composition.  The
chain of the book repeats one last time, with every step a
cross-reference:

1. **the invariant** — sinks = sources per (cell × ordinate × group),
   with the curvilinear streaming of :eq:`balance-general` in the
   streaming slot of :eq:`multigroup`;
2. **the operators** — :math:`S` and :math:`F` gain group structure
   exactly as :doc:`slab_multigroup` derived, unchanged by curvature;
   :math:`L+C` stays group-diagonal, now *including* the
   redistribution cascade;
3. **the eigenvalue posing** — :math:`(L+C-S-N_{2n}-B)\,\psi =
   \tfrac{1}{k}\,F\,\psi` (:eq:`sn-within-group-with-n2n`), verbatim;
4. **the strategy encodings** — the same monolithic within-group
   iteration; no new loop, no new operator, no new closure.

.. admonition:: Key Facts
   :class: tip

   * The multigroup curvilinear equation **is** :eq:`multigroup` with
     its streaming term read as the curvilinear conservative form
     :eq:`balance-general` (redistribution cascade included).
     Redistribution is *streaming* — in-flight, group-diagonal — so
     **all group coupling stays in the sources**, exactly as in the
     slab.
   * The walk **structure** — the direction-keyed cell order
     (:class:`~orpheus.transport.spatial.scheme.CellVisit`), the
     :math:`\alpha`-dome recursion :eq:`alpha-recursion`, the
     Morel--Montry weights :eq:`mm-weights` and their derived
     constants :math:`c_{\rm in}/c_{\rm out}`, the geometry factor
     :math:`\Delta A_i/w_n` — is computed from geometry and quadrature
     alone and **carries no group axis**.  The realized :term:`sweep`
     **data** — the WDD denominator and attenuation (through
     :math:`\Sigt{g} V_i`) and the starting-direction state
     :math:`\psi_{1/2,g}` — is per-group, **group-diagonal**: parallel
     lanes broadcast along the same walk, never mixed by it.
   * **Vocabulary trap**: in this book :math:`\tau` and
     :math:`c_{\rm in}/c_{\rm out}` are the Morel--Montry *angular*
     closure weights — functions of :math:`(\mu, w)` only, shape
     ``(N,)``, no group axis.  They are **not** :term:`optical thicknesses <optical thickness>`.
     The per-group "optical" quantity is the WDD denominator /
     attenuation of the cross-section stratum.
   * **No group loop exists on the iterative path.**  The within-group
     system is monolithic over all groups — the *full* multigroup
     :math:`S` (every :math:`g' \to g` transfer) sits inside
     :math:`A = L+C-S-N_{2n}-B` (:eq:`sn-within-group-with-n2n`),
     together with the :math:`(n,2n)` gain :math:`N_{2n}`, which is a
     group transfer at a collision event exactly as :math:`S` is; and
     fission enters externally as
     :math:`q = F\psi/k`.  Source iteration lags the entire
     :math:`S + N_{2n}`; Krylov iterates the full state
     (:math:`n_{\rm dof} = N \cdot n_g \cdot n_x`).  The sweep
     :math:`(L+C)^{-1}` *factors* per group — a mathematical fact
     about group-diagonality — realized as one vectorised sweep with
     the group axis broadcast.
   * The verification chain: closed-form :math:`\kinf` recovery on
     {slab, sphere, cylinder} × {1, 2, 4 groups} × {SI, Krylov}; the
     heterogeneous-2G SI :math:`\equiv` Krylov equivalence gate
     (:ref:`#196 <sn-issue-196-eigenvalue-equivalence>`); the MR↔MG
     trajectory-resolvent flux-shape cross-checks
     (:ref:`sn-curvilinear-trajectory-resolvent-crosscheck-section`).
     The curvilinear MMS family is **1-group only** — an honest gap;
     the multigroup chain rides eigenvalue and analytical gates.
   * The group count *does* interact with iteration **behavior** (not
     with the walk): the inner-tolerance amplification
     :math:`\rho/(1-\rho)` into :math:`\kinf`.  (The sphere-4G
     unpreconditioned-GMRES budget was the second such interaction until
     2026-08-10, when it was retired as healed — see the Gotcha admonition
     below; every cell of the grid is now gated.)


The posing: energy on the curvilinear invariant
===============================================

There is no new equation to write, and writing none is the point.
:eq:`multigroup` was already stated with a coordinate-agnostic
streaming slot — "the streaming operator depends on the coordinate
system" — and :doc:`curvilinear_one_group` derived what that slot
contains for the sphere and the cylinder: the conservative radial
derivative *plus* the angular redistribution cascade of
:eq:`balance-general`, closed by :eq:`wdd-closure` with the
Morel--Montry weights :eq:`mm-weights`.  The multigroup curvilinear
balance is the composition of the two chapters, term for term.

The one structural question worth asking is whether curvature moved
anything *across* the group-diagonal line that organized
:doc:`slab_multigroup`: streaming and collision group-diagonal on the
left, all group coupling in the sources on the right.  It did not,
and the reason is physical.  The redistribution term is the angular
derivative that appears when straight-line flight is described in
curved coordinates — a neutron at fixed energy, colliding with
nothing, drifts through the local direction coordinate :math:`\mu` as
:math:`r` changes.  It is *streaming*, bookkept per group exactly
like :math:`\mu\,\partial_x`: the cascade coefficients
:math:`\alpha_{n\pm 1/2}` multiply :math:`\psi_{g,n\pm 1/2}` at the
*same* :math:`g`.  Energy changes at collision events only —
scattering transfer (:math:`S`) and fission emission (:math:`F`) —
the :math:`(n,2n)` gain :math:`N_{2n}` is a group transfer at a
collision event on exactly the same footing, and the
boundary-reflection gain :math:`B` preserves energy just as
it did in the slab.  So the honest within-group operator keeps its
shape,

.. math::

   (L + C - S - N_{2n} - B)\,\psi \;=\; q
   \qquad\text{and}\qquad
   (L + C - S - N_{2n} - B)\,\psi \;=\; \tfrac{1}{k}\,F\,\psi ,

with :math:`L` now carrying the curvilinear streaming (cascade
included), :math:`C` the collision diagonal, and :math:`S`,
:math:`N_{2n}`, :math:`B`,
:math:`F` **unchanged** — the same operators :doc:`slab_multigroup`
derived, consumed by composition
(:ref:`sn-scattering-fission-operators`).

.. note::

   **Once, for this whole chapter:** since #426 step 2 (2026-09-04)
   :math:`S` and :math:`N_{2n}` are two *instances of one binding* —
   the same
   :class:`~orpheus.transport.operators.transfer.TransferOperator`
   with the same faces, arms and transposes — differing only in the
   yield :math:`y` inside :math:`\Lambda_c` (:math:`y = 1` for
   :math:`S`, :math:`y = \nu_{2n} = 2` for :math:`N_{2n}`) and in the
   channel's own Legendre stack.  So every statement this chapter
   makes about :math:`S` — group structure, anisotropy, the
   collision-event argument, the lagging in source iteration — holds
   for :math:`N_{2n}` verbatim with :math:`y = 2`, and the two terms
   are named separately below only where the member list itself is the
   point.

:math:`L+C` is
group-diagonal; its inverse — the sequential sweep of
:doc:`curvilinear_one_group` — acts group by group.  No curvilinear
term crosses the line.


What carries a group axis — and what does not
=============================================

The claim "the group axis rides the walk untouched" deserves a
precise spelling, because it is *three* claims of different strength,
and the code separates them cleanly.

Group-independent structure
---------------------------

These objects have **no group axis at all** — their arrays are shaped
by quadrature and mesh, and no cross section enters their
construction:

* **The walk itself.**  ``SNMesh.dag_walk`` yields the per-(:term:`ordinate`,
  cell) visit sequence — the direction-keyed topological order the
  sweep follows — as
  :class:`~orpheus.transport.spatial.scheme.CellVisit` packets
  carrying the cell index, the
  :class:`~orpheus.transport.spatial.scheme.StreamingTerms`, and the
  sweep-resolved downstream face area — three fields, all Tier-1.
  Per-group data is deliberately *not* in the packet: the cell update
  reads ``total_xs[:, cell_idx]`` and ``source[:, cell_idx]`` at the
  call site.  (⛔ The packet also carried the closure floats ``tau`` /
  ``c_in`` / ``c_out`` until 2026-08-28.  P4.9a retired all three: the
  stamp made the *mesh* a second home for closure-owned data, and the
  angular contributions now reach the scheme already multiplied out,
  assembled by the caller from the closure's own per-ordinate arrays.
  Being ordinate-keyed floats they were Tier-1 too, so this does not
  disturb the stratification argument — it removes three fields from
  the stratum, not a stratum boundary.)
* **The** :math:`\alpha` **-dome.**  The recursion
  :eq:`alpha-recursion` on
  :class:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator`
  consumes the quadrature :math:`(\mu_n, w_n)` and nothing else — which
  is why it moved OFF that class in the 2026-08-26 un-weld and onto
  :class:`~orpheus.sn.angular.redistribution.AngularRedistribution`, the
  angular half of the redistribution operator's tensor factorization.
  Per level the dome is ``(M_p + 1,)`` (sphere: one level, ``(N+1,)``);
  the geometric factor is ``delta_A``, and the product
  :math:`\Delta A_i / w_n` is built by each consumer from those two
  factors rather than held in a store that owns neither. ⛔ This last
  clause read *"formed by each consumer rather than cached"* until
  2026-08-29; two of the three formers do cache their own copy (the
  angular closure at construction, the scan cache at build), which
  changes nothing about the stratification argument but is not the same
  sentence — see :ref:`sn-geometry-factor`.
* **The Morel--Montry closure.**  :math:`\tau` is an angular-scheme
  property — a function of :math:`(\mu, w)` only, per the Issue #236
  Step C ruling — owned by
  :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
  and stamped per ordinate on each visit; the derived constants
  :math:`c_{\rm in} = (1-\tau)/\tau\,\alpha_{\rm out} + \alpha_{\rm in}`
  and :math:`c_{\rm out} = \alpha_{\rm out}/\tau` are ``(N,)``.
* **The geometry stratum of the sweep cache.**
  :class:`~orpheus.sn.sweep.cache.StreamingCoefficientCache` documents the
  boundary explicitly: *no* ``ng`` *axis — no cross-section
  dependence*.  It is built once per (mesh, quadrature, angular closure)
  and survives every cross-section rebind.  Since P4.9b the "once" is
  enforced rather than hoped for: the table is resolved **lazily**, on
  first need, through the strategy layer's hub-keyed intern
  :func:`~orpheus.sn.loss_representation.geometry_cache_for`, and a
  ``foundation`` gate pins the build count at exactly one per hub across
  a whole solve (:ref:`sn-p49b-operator-poses-with-closures`).

.. note:: **The τ/c vocabulary trap.**  A reader arriving from the
   general transport literature will want to read :math:`\tau` as an
   optical thickness :math:`\Sigma_t \Delta r / |\mu|` — a per-group
   quantity.  In this book it is not: :math:`\tau` is the
   Morel--Montry *angular* closure weight of :eq:`mm-weights`,
   group-blind by construction.  The per-group attenuation-like
   quantities live in the next tier, under different names.  (The
   #236 τ-campaign of :doc:`curvilinear_one_group` is the derivation
   that earned :math:`\tau` this ownership.)

   ⚠ The trap has **three** horns, not two, and the full nomenclature
   table — the closure weight (this one), the optical depth
   :math:`\Sigma_t s` of ``peierls_nystrom`` / MoC / the ``transport``
   spatial schemes, and the critical half-thickness in mean free paths of
   ``fn_method`` — is in
   :mod:`orpheus.derivations.discrete.sn.angular_differencing`, alongside
   the matching two-sense split of :math:`\beta`.  Note also that
   ``tau_inv`` in :mod:`orpheus.sn.sweep.cache` is :math:`1/\tau` of
   *this* sense, sitting a few files from code using the optical-depth
   sense.

Group-diagonal data riding the walk
-----------------------------------

These arrays **do** carry a group axis — but a *diagonal* one: group
:math:`g`'s lane never reads group :math:`g'`'s.  The axis enters
through exactly one algebraic door,
:func:`~orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`,
where the WDD cell update's denominator picks up the collision term:

.. math::
   :label: sn-curvilinear-mg-cell-denominator

   \text{denom}_{g,n}
   \;=\; 2\,|\mu_n|\,A_{{\rm down},n}
   \;+\; \text{(angular closure term)}_n
   \;+\; \Sigt{g}\,V_i .

.. (vv-status rationale) Structural / representational identity: the WDD
   cell-update denominator, showing Σ_t,g·V_i as the SOLE group-diagonal door.
   Not a solver claim; the denominator assembly is pinned by the
   ``@pytest.mark.foundation`` gates
   ``tests/sn/sweep/core/test_cell_balance_for_streaming.py`` and
   ``test_cache.py::test_cache_populator_matches_cell_balance_for_streaming``
   (the cache's populated denominator vs a direct call to
   ``cell_balance_for_streaming`` at rtol=1e-14; renamed from
   ``…_matches_cell_balance_terms`` on 2026-08-28 when the scalar twin it
   used to compare against was retired) — foundation software-invariant
   tests carry no ``verifies(...)`` by design.
.. vv-status: sn-curvilinear-mg-cell-denominator documented

:math:`\Sigt{g} V_i` is the **sole group dependence** of the cell
update — everything else in the row is Tier-1 structure.  Downstream
of that door:

* the **cross-section stratum** of the sweep cache (``CollisionCache``:
  inverse denominators, attenuation factors, their cumulative
  products) is shaped ``(N, ng, nx)`` and is rebuilt on every
  cross-section rebind while the geometry stratum survives — the
  cache's two-strata split *is* the code's own statement of this
  tier boundary;
* the **starting-direction state** :math:`\psi_{1/2,g}` is per-group
  data produced by group-blind machinery:
  :func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`
  takes ``(ng, nx)`` source and cross-section fields and marches the
  :math:`\mu = -1` characteristic *sequentially in cells, vectorised
  across groups* — one march, :math:`n_g` lanes.  The System B slot
  of the route-(a) composite
  (:ref:`sn-direct-seed-solve`) carries
  ``(ng, nx)`` views per leg for the same reason.

Each group's sweep is therefore an *independent triangular solve*:
the group axis rides the walk as parallel lanes, and the realization
is one vectorised pass with NumPy broadcasting rather than a Python
loop over :math:`g` — the same "trailing group axis" statement
:doc:`slab_multigroup` made for the slab, unchanged.

Group coupling: S and F, unchanged
----------------------------------

All :math:`g' \to g` mixing lives in the two operators
:doc:`slab_multigroup` derived, and neither has a curvilinear term:

* **Scattering** is the conjugation :math:`S = R \circ \Lambda \circ M`
  — project to moments, transfer on the group axis, reconstruct.  The
  group-asymmetric factor is *exactly one*:
  :class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`,
  the per-:math:`\ell` cross-section matmul on the energy axis.  The
  projection and reconstruction faces are quadrature objects —
  group-blind.
* **The** :math:`(n,2n)` **secondary emission** is a *separate*
  within-group gain :math:`N_{2n}` on the same energy slot, likewise
  with no curvilinear term (:ref:`n2n-reactions`; its lift and
  transpose at :ref:`sn-n2n-adjoint`).  It rode inside :math:`S`'s
  conjugation as an :math:`\ell = 0` summand,
  :math:`S = R\circ(\Lambda + N_{2n})\circ M`, until the CS4c step-3
  extraction on 2026-08-30.
* **Fission** is the rank-1-in-energy dyad :math:`F = \chi \otimes
  \nu\Sigma_f` — a contraction over groups followed by a broadcast
  across the emission spectrum — and it never enters the swept
  operator: it is applied at the *outer* level as
  :math:`q = F\psi/k` (:ref:`sn-mg-eigenvalue-posing`).

Curvature changed *where neutrons stream*, not *how they change
group*; the operators are consumed by composition, not re-derived.


The monolithic multigroup iteration
===================================

The textbook narrative for multigroup S\ :sub:`N` is group-by-group:
sweep group :math:`g` with its scattering source built from the
latest available fluxes, march down the group axis Gauss--Seidel
style, iterate on upscatter.  **ORPHEUS deliberately does not do
this**, in any geometry — and it is worth stating here because the
curvilinear chapters are where a reader might expect the walk to
force per-group orchestration.  It doesn't; the design is monolithic:

* :func:`~orpheus.sn.coupled_system.build_within_group_system`
  constructs **one** system over the full :math:`(N, n_g, n_x)`
  state: :math:`A = L+C - S - N_{2n} - B` with the *full* multigroup
  :math:`S` — every :math:`g' \to g` transfer, anisotropy included —
  and the :math:`(n,2n)` gain beside it, inside the operator.
  "Within-group" in this codebase means
  *fission-external* (fission enters as :math:`q_{\rm ext} =
  F\psi/k`), **not** per-group.
* **Source iteration** lags the whole :math:`S` as one Jacobi-style
  gain: :math:`\psi \leftarrow (L+C)^{-1}(S\,\psi + q_{\rm ext})`,
  all groups advanced together per iteration.
* **Krylov** builds its subspace on the full state — the restart
  dimension is sized from :math:`n_{\rm dof} = N \cdot n_g \cdot n_x`
  (the ERR-053 family lesson: a composite's Krylov dimensions come
  from ``to_flat``, never from a sub-block).
* The only Gauss--Seidel schedule anywhere in the solver is over
  sweep **octants** (multi-D Cartesian boundary coupling,
  :doc:`cartesian_multid`) — never over groups.

The two spellings of "per group" must be kept apart.  That
:math:`(L+C)^{-1}` *factors* into independent per-group triangular
solves is a **mathematical fact** about group-diagonality — and the
dense assembly arm exhibits it structurally: the S\ :sub:`N` code's
dense-assembly loop over :math:`g` materializes one sparse block
per group *over the same walk graph* (same visit order, same
:math:`\alpha`-cascade; only the :math:`\Sigt{g}` diagonal differs).
That the *iteration* is monolithic is a **design choice** about where
:math:`S` is lagged — chosen so the group axis is data for the
operator algebra rather than an orchestration axis for the solver.
The inner-loop machinery of :doc:`slab_one_group` §7 — one resolvent,
SI or Krylov, sweep as kernel or preconditioner
(:ref:`choosing-inverse-realisation`) — serves the multigroup
curvilinear problem with no new code path.

.. admonition:: Gotcha — the group count changes iteration behavior,
   not the walk
   :class: warning

   One real multigroup :math:`\times` curvilinear interaction remains, and
   it lives in the *iteration*, not the discretization.  A second was
   retired as healed on 2026-08-10; it is kept below because the *reason*
   it survived eleven weeks past its own cure is the transferable lesson.

   * **Inner-tolerance amplification**: the SI stopping residual
     propagates into :math:`\kinf` amplified by
     :math:`\rho/(1-\rho)` with :math:`\rho` the scattering-iteration
     dominance ratio — documented at the head of the same test file.
     Multigroup data moves :math:`\rho`; the walk is untouched.
   * **Sphere-4G Krylov budget** (`#200
     <https://github.com/deOliveira-R/ORPHEUS/issues/200>`_) — **no longer
     a live limitation, retired 2026-08-10.** From 2026-05-19 the 4-group
     homogeneous sphere was the xfail'd cell of the coordinates × groups ×
     drivers grid in
     :file:`tests/sn/verification/analytical/test_kinf_homogeneous.py`, on
     the grounds that it exceeded the unpreconditioned GMRES iteration
     budget.  The cell now passes and is gated like every other, agreeing
     with the closed-form reference at :math:`\mathrm{rel} = 3.6\times
     10^{-15}`.  The cure was **not** the preconditioner: #200 is still
     open and ``_within_group_krylov`` still ships an explicit identity.
     What healed it is the GMRES ``restart``-sizing lineage — ERR-053
     (2026-05-28) removing the ``restart=min(50, full_size)`` clamp, then
     #282 / #280 route (a) (2026-07-04) sizing ``restart`` from the full
     augmented ravel.  The stated budget was never even the live knob on
     this path: ``max_inner`` becomes scipy's ``maxiter``, which counts
     restart CYCLES, and one cycle spans the whole Krylov space when
     ``restart == n_dof``.  See that test module's "History" section; the
     lesson the eleven-week delay bought is recorded under issue #340 —
     an imperative ``pytest.xfail()`` can never report ``XPASS``, so the
     exclusion could not retire itself.


Verification
============

The composition claim — group physics from Part A, walk from Part B,
coupled only through :math:`S` and :math:`F` — is pinned at four
levels.  Per the ``vv-principles`` degeneracy rules, every case below
is :math:`\ge 2` groups (a 1-group eigenvalue cannot see group
coupling at all), and the flux-shape and equivalence gates are
heterogeneous by design.

* **Closed-form eigenvalue, full matrix.**
  :file:`tests/sn/verification/analytical/test_kinf_homogeneous.py`
  recovers the analytical
  :math:`\kinf = \lambda_{\max}(\mathbf{\Sigma}_a^{-1}\chi\,\nu\Sigma_f^\top)`
  (:ref:`mg-eigenvalue-problem`) on {slab, sphere, cylinder} ×
  {1, 2, 4 groups} × {SI, Krylov} to ``rtol=1e-10``, and pins the
  multigroup **eigenvector spectrum** on all three coordinate
  systems.  The base chapter's Gate 4.1
  (:ref:`sn-curvilinear-homogeneous-kinf-recovery-section`) is the
  2-group sphere member of this family — with its honest-scope note:
  a homogeneous :math:`\kinf` is flux-shape *independent*, necessary
  but never sufficient.
* **Flux shape, multigroup.**  The trajectory-resolvent
  cross-check table
  (:ref:`sn-curvilinear-trajectory-resolvent-crosscheck-section`)
  carries the multigroup rows: the 2-group three-region sphere
  against the multiregion Green's-function reference (MR↔MG
  reduction ``rtol=1e-9``) and the 2-group three-region cylinder
  (MR↔MG, :math:`K=3`).  This is the structurally-independent
  flux-*shape* evidence the eigenvalue rows cannot supply.
* **Path equivalence, heterogeneous 2G.**  The #196 permanent gate
  (:ref:`sn-issue-196-eigenvalue-equivalence`,
  :file:`tests/sn/eigenvalue/test_keff_curvilinear.py`) asserts
  SI :math:`\equiv` Krylov on 2-group fuel|moderator sphere *and*
  cylinder — :math:`|\Delta k| < 10^{-7}` (observed floor
  :math:`\sim 10^{-11}`), per-group eigenvector shape agreement, and
  a non-flatness guard so the redistribution terms are genuinely
  exercised, not nulled by a homogeneous/1G degenerate.
* **Multi-region integration.**  The
  ``TestCylinderMultiGroupMultiRegion`` and
  ``TestMultiGroupMultiRegionSpherical`` families (same file) pin
  2-group heterogeneous :math:`\keff`, 4-group scattering
  convergence, eigenvector non-flatness, and particle balance on
  both geometries, plus resolution / spatial-convergence coverage
  (2-group different-resolutions on the cylinder; 1-group spatial
  convergence on the sphere).

.. admonition:: Honest scope — no multigroup curvilinear MMS
   :class: important

   The curvilinear MMS suites
   (:file:`tests/sn/verification/mms/test_mms_curvilinear.py` and
   the anisotropic family) are **1-group**.  Multigroup curvilinear
   correctness rides the eigenvalue, analytical, and equivalence
   chains above — there is no manufactured-solution gate on the
   composed group :math:`\times` curvature system.  By the Mode-7
   discipline this is a *declared* gap, not a hidden one: the terms
   the MMS family does exercise (cascade, closure, pole) are
   group-blind Tier-1 structure, and the group coupling the MMS
   family nulls is pinned independently by the 2G flux-shape and
   equivalence gates.


What broadens next
==================

Nothing — this rung completes the book's broadening progression.
Part A posed the problem and broadened it in energy and space; Part B
re-posed streaming for curved coordinates and, in this chapter,
confirmed the energy axis rides that machinery as data.  What remains
on the curvilinear axis is not a broadening:

* the **refinement record** — how the curvilinear closure seeds were
  found wrong twice and fixed, and the campaign that ended in the
  route-(a) architecture — is :doc:`curvilinear_numerics`;
* **multi-dimensional curvilinear** (2-D :math:`r`--:math:`z`) has no
  ORPHEUS realization; the book documents machinery that exists, so
  the chapter for it is written when the code is.
