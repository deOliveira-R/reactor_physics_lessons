.. _sn-adjoint:

Adjoint transport: the dual operators
=====================================

This chapter is the S\ :sub:`N` book's adjoint rung — the dual
operators that the perturbation-theory, detector-sensitivity, and
adjoint-weighted-homogenisation chains consume.  The adjoint chain has
three layers, and **all three are now landed**:

* The **walk adjoint** — the loss composite's transpose
  :math:`(L+C)^{\mathsf T} = L^{\mathsf T} + C` — is machinery of the
  loss *representations* and lives with them: the orientation axis,
  its swap law, and the deferral ledger
  (:ref:`loss-rep-orientation-two-frames`), realised as the Wave-O
  analytic reverse-direction matvec
  (`#280 <https://github.com/deOliveira-R/ORPHEUS/issues/280>`_ /
  `#310 <https://github.com/deOliveira-R/ORPHEUS/issues/310>`_).  This
  chapter points, it does not re-derive.
* The **scattering adjoint** :math:`S^{\mathsf T}`
  (:ref:`sn-scattering-adjoint`, the #276 P3 record): free by frame
  conjugation, with no per-geometry derivation to verify (closes
  `#118 <https://github.com/deOliveira-R/ORPHEUS/issues/118>`_).
* The **daggered posing and the adjoint flux** :math:`\psi^*`
  solving :math:`A_{\rm loss}^{\dagger}\,\psi^* =
  \tfrac1k\,F^{\dagger}\,\psi^*` (with
  :math:`A_{\rm loss} = L+C-S-N_{2n}-B`,
  :eq:`sn-within-group-with-n2n`) landed at **#276 A4/A5**
  (:ref:`sn-adjoint-daggered-posing`): the whole eigenproblem is posed
  by DAGGER-ing the forward operator triple through
  :func:`~orpheus.numerics.iteration.KEigenvalue`, and the importance
  map :math:`\varphi^*` rides a role-typed
  :class:`~orpheus.sn.solution.AdjointSolution` carrier
  (:ref:`sn-adjoint-carrier`).  The :math:`\varphi^*` consumers —
  adjoint-weighted homogenisation (frame-machinery P6,
  `#51 <https://github.com/deOliveira-R/ORPHEUS/issues/51>`_ /
  `#281 <https://github.com/deOliveira-R/ORPHEUS/issues/281>`_) and
  perturbation theory / response estimation — are now **unblocked**
  and grow the chapter as they land (:ref:`sn-adjoint-consumers`).

The **spine of the chapter is the route decision**
(:ref:`sn-adjoint-route`): ORPHEUS poses the adjoint by transposing
the *discrete* forward operator, **not** by discretising the
*continuous* adjoint.  Every property below — the exact
:math:`k^{\dagger} = k` identity, the reciprocity that holds at finite
:math:`N` and :math:`h`, the absence of any adjoint-specific loop or
sweep — is a consequence of that one choice.

.. admonition:: Key Facts
   :class: tip

   * **The route (the spine).**  The adjoint is the exact **discrete
     transpose** of the forward operator triple —
     ``KEigenvalue((L+C).H, (S+N2N+B).H, F.H)`` (the daggered resolvent,
     gain, and fission; the loss :math:`A_{\rm loss}^{\dagger} =
     (L{+}C).\mathtt{H} - (S{+}N_{2n}{+}B).\mathtt{H}` is formed inside,
     the gain being the FOLD of the builder's ``explicit_gains`` rather
     than a hand-written member list) fed to
     the UNCHANGED
     :func:`~orpheus.numerics.eigenvalue.power_iteration`.  There is
     **no** discretise-then-adjoint step, so duality holds EXACTLY at
     finite :math:`N` and :math:`h` and :math:`k^{\dagger} = k` is an
     exact algebraic identity, not a converged agreement
     (:ref:`sn-adjoint-route`).  The textbook :math:`\mu`-reversal
     (continuous route) survives ONLY as a slab oracle, never in
     production.
   * **Three transposes, one landmine** (:ref:`sn-adjoint-three-transposes`).
     (1) the **Euclidean matrix transpose** :math:`A^{\mathsf T}` (the
     scattering group-transpose :math:`S^{\mathsf T}`, #118, and the
     reverse-scan walk transpose, #280); (2) the **Hilbert / G-metric
     adjoint** :math:`A^{\dagger} = A.\mathtt{H} = G^{-1}A^{\mathsf T}G`
     (:ref:`operator-adjoint`); (3) the **continuous adjoint operator**
     (whose spatial signature is :math:`\mu`-reversal).  Conflating
     them is the #1 way to a plausible-but-wrong adjoint.
   * **The operator algebra IS the implementation.**  No
     adjoint-specific solver code exists anywhere: ``.H`` is the exact
     discrete Hilbert adjoint of every leaf, and the swap law
     :math:`(A^{\dagger})^{-1} = (A^{-1})^{\dagger}`
     (:eq:`loss-rep-adjoint-inverse-swap`) makes :math:`(L+C).\mathtt{H}`
     invertible for free — the daggered inner solve rides the
     reverse-scan transpose sweep behind ``A.H.inverse()``.
   * **The G-metric is a free parameter that no eigenvalue gate can
     see.**  :math:`A.\mathtt{H} = G'^{-1}A^{\mathsf T}G'` is
     metric-similar to :math:`A^{\mathsf T}` for ANY invertible
     :math:`G'`, so :math:`k^{\dagger}` is EXACTLY invariant even under
     a wrong metric.  A metric bug is **k-blind but vector-visible**:
     the coupled-sphere defining-law residual row is its sole catcher
     (:ref:`sn-adjoint-verification`; the ERR-067 family).
   * **The importance carrier is a TYPE, not a flag.**
     :class:`~orpheus.sn.solution.AdjointSolution` is a sibling of
     :class:`~orpheus.sn.solution.Solution` under a role-agnostic base;
     the forward-physics operations (``homogenize`` / ``condense`` /
     ``reaction_rate_density``) are **structurally absent** on it,
     because there is no reaction rate to preserve on an importance map
     (:ref:`sn-adjoint-carrier`).  ``.importance`` is the domain-named
     alias for :math:`\varphi^*`.
   * **The scattering adjoint** :math:`S^{\mathsf T}` is assembled from
     **leaf transposes** of the frame conjugation,
     :math:`(R \circ \Lambda_{\ell\ge0} \circ M)^{\mathsf T}
     = M^{\mathsf T} \circ \Lambda^{\mathsf T} \circ
     R^{\mathsf T}` — no per-geometry derivation (#276 P3, closes
     #118), reciprocity-pinned against the structurally *independent*
     forward fast-path.  The forward-adjoint **asymmetry is
     principled**: the forward source keeps the scalar fast-path for
     SI-sweep performance; the adjoint — not the hot path — rides the
     validated frame form, which is what makes the reciprocity gate a
     genuine cross-check rather than a tautology.
   * **The** :math:`(n,2n)` **channel is a SEPARATE operator with its
     own transpose** since CS4c step 3 (:ref:`sn-n2n-adjoint`).  It was
     a passenger of :math:`S` until 2026-08-30 — inside the
     :math:`\Lambda + N_{2n}` sum of the frame conjugation on the
     adjoint side — and left because its bundling (with :math:`S` for
     anisotropy, with :math:`F` for production accounting) is
     **context-dependent and must not be decided at the operator
     level**.
   * ⭐ **Both collision gains are one binding at one order since #426
     step 2 (2026-09-04), so the** :math:`(n,2n)` **transpose is a
     product reversal, not a special case.**  :math:`S` and
     :math:`N_{2n}` are the two roles of
     :class:`~orpheus.transport.operators.transfer.TransferOperator`,
     differing in the **yield** :math:`y` carried inside :math:`\Lambda`
     and in nothing else:

     .. math::

        T_c \;=\; \tfrac1W\,R\,\Lambda_c\,M ,
        \qquad
        \Lambda_c \;=\; y_c \sum_{\ell=0}^{L}
                        \mathbf{P}_\ell \otimes \Sigma_{c,\ell} ,
        \qquad
        y_S = 1,\; y_{2n} = 2 ,

     so :math:`N_{2n}^{\mathsf T} = \tfrac1W\,M^{\mathsf T}
     \Lambda_{2n}^{\mathsf T} R^{\mathsf T}` by the SAME factor
     reversal as :math:`S`'s (:eq:`sn-n2n-adjoint-source`), over
     :math:`L{+}1` blocks rather than one.  The :math:`\ell = 0`
     half is still the isotropic lift run backwards —
     :math:`\tfrac{w_m}{W}\,K^{\mathsf T}\sum_n\chi_n`
     (:eq:`sn-n2n-isotropic-lift`) — note the :math:`w_m`, which an
     equal-weight fixture is structurally blind to.
   * ⛔ **Until 2026-09-04 the** :math:`(n,2n)` **lift and its transpose
     were single-**:math:`\ell`\ **, and that was a MODEL imposed at the
     operator tier — a defect, catalogued as ERR-082**
     (:ref:`the L0 error catalogue
     <theory-verification-error-catalog>`).  The kernel held
     ``Sig2[0]`` alone and the
     binding minted its frame at order 0, while the shipped GENDF files
     store seven Legendre moments for MT=16 — the same order as elastic
     — and (since #426 step 1) ``Mixture.Sig2`` carried all of them.
     ``[M]`` restoring them moves :math:`k` by :math:`-413.55` in
     :math:`\Delta k\cdot10^{5}` on the Be-reflected fast slab
     (:math:`-346.01` in :math:`\Delta\rho\cdot10^{5}`;
     :ref:`the measured block <sn-n2n-p0-truncation-measured>` carries
     all three conventions, three fixtures and the controls).  The
     shipped library now reads that value with no probe —
     ``tests/sn/verification/analytical/test_be_reflected_n2n_anisotropy.py``.
     The history, and what is still :math:`P_0` **by physics** and must
     not be "fixed", are at
     :ref:`the truncation record <sn-n2n-p0-truncation>`
     (`#426 <https://github.com/deOliveira-R/ORPHEUS/issues/426>`_).

The continuous adjoint problem and importance
=============================================

Before the discrete machinery, the physics.  The adjoint transport
equation, its importance interpretation, and the reciprocity duality
are the *targets* the discrete construction must reproduce — and the
route decision (:ref:`sn-adjoint-route`) is precisely a claim about how
faithfully it reproduces them.

The adjoint transport equation
------------------------------

The forward within-group transport equation streams a particle along
:math:`\hat\Omega`, removes it at rate :math:`\Sigma_t`, and gains it
back through in-scatter and fission.  The **continuous adjoint** is the
formal transpose of that operator under the phase-space inner product
:math:`\langle a,b\rangle = \int_{\mathcal D}\!\mathrm dV
\int_{4\pi}\!\mathrm d\Omega\,\sum_g a\,b`:

.. math::
   :label: sn-adjoint-continuous

   -\,\hat\Omega\cdot\nabla\psi^*_g
   + \Sigma_{t,g}\,\psi^*_g
   \;=\; \sum_{g'} \Sigma_{s,\,g\to g'}\,\psi^*_{g'}
   \;+\; \frac1k\,\nu\Sigma_{f,g}\sum_{g'}\chi_{g'}\,\psi^*_{g'} .

.. (vv-status rationale) Literature-transcribed definitional identity: the
   continuous adjoint transport equation (Bell & Glasstone §6, Lewis & Miller
   §6).  It states the CONTINUOUS target the discrete construction reproduces;
   it is NOT a per-term solver claim about ORPHEUS code.  Its verifiable
   discrete counterpart is the daggered eigenproblem :eq:`sn-adjoint-eigenproblem`
   and the reciprocity duality :eq:`sn-adjoint-duality`, which the daggered
   posing satisfies EXACTLY (P1.3/P1.2 certification rows).
.. vv-status: sn-adjoint-continuous documented

This is the **eigenvalue** (criticality) adjoint, with the fission gain
scaled by :math:`1/k` and :math:`k^{\dagger} = k` (below).  The
**fixed-source (importance)** adjoint drops the fission term and adds a
prescribed adjoint source — the detector response —
:math:`-\hat\Omega\cdot\nabla\psi^*_g + \Sigma_{t,g}\psi^*_g -
\sum_{g'}\Sigma_{s,\,g\to g'}\psi^*_{g'} = q^*_g = \Sigma_{d,g}`.  Three
sign/role changes distinguish either form from the forward equation, and
each is the continuous face of a discrete transpose the code must
realise:

* **The streaming term flips sign**,
  :math:`\hat\Omega\cdot\nabla \to -\hat\Omega\cdot\nabla` (equivalently
  :math:`\mu\to-\mu`): the adjoint particle propagates *against* the
  physical flow.  This is the :math:`\mu`\ **-reversal**.
* **The scattering kernel transposes** its energy transfer,
  :math:`\Sigma_{s,\,g'\to g}\to\Sigma_{s,\,g\to g'}`: the adjoint
  particle downscatters where the forward particle upscattered.  With
  the ORPHEUS convention ``SigS[g_from, g_to]``, the forward in-scatter
  source is :math:`\Sigma_s^{\mathsf T}\varphi` and the adjoint source
  is :math:`\Sigma_s\varphi^*` — the transpose is dropped.
* **The fission term swaps emission and production**,
  :math:`\chi_g\,\nu\Sigma_{f,g'} \to \nu\Sigma_{f,g}\,\chi_{g'}`: the
  adjoint particle is "born" with the production spectrum
  :math:`\nu\Sigma_f` and "weighted" by the emission spectrum
  :math:`\chi`.  This :math:`\chi\leftrightarrow\nu\Sigma_f` swap is the
  canonical adjoint-fission trap.

Importance — the interpretation of the adjoint flux
---------------------------------------------------

The adjoint flux is not a flux.  :math:`\psi^*(\vec r,\hat\Omega,g)` is
the **importance**: the expected contribution to a detector response
:math:`\langle\Sigma_d,\psi\rangle` from a single particle introduced
at the phase-space point :math:`(\vec r,\hat\Omega,g)`
(:cite:`BellGlasstone1970` §6; :cite:`LewisMiller1984` §6;
:cite:`Lux1991` for the Monte-Carlo importance).  A neutron born deep
in the fuel with an inward-pointing direction is *more important* to
the chain reaction than one born at the vacuum boundary heading out —
importance is direction- and energy-resolved, and on a finite system
it is genuinely :math:`\mu`-asymmetric, which is exactly the content
the :math:`\mu`-reversal carries.

For the eigenvalue problem the adjoint source vanishes
(:math:`q^* = 0`) and the importance is the fundamental **adjoint
eigenmode** :math:`\varphi^*` — the reactor's "worth function": the
first-order sensitivity of :math:`k` to a perturbation introduced at
each phase-space point.  This is why perturbation theory and
generalised perturbation theory (:ref:`sn-adjoint-consumers`) are the
adjoint flux's native consumers.

Reciprocity and the fundamental duality
----------------------------------------

The single identity from which every adjoint application descends: for
a forward fixed-source solve :math:`A_{\rm loss}\,\psi = q` and an
adjoint solve :math:`A_{\rm loss}^{\dagger}\,\psi^* = \Sigma_d` driven
by a detector response :math:`\Sigma_d`,

.. math::
   :label: sn-adjoint-duality

   \langle \Sigma_d,\,\psi\rangle
   \;=\;
   \langle \psi^*,\, q\rangle ,

.. The label sn-adjoint-duality is a verifies-target (a solver claim:
.. the discrete reciprocity the daggered posing satisfies EXACTLY).
.. It carries NO vv-status sentinel by the wired⟹no-sentinel convention —
.. the pinning L1 gate (test_sn_adjoint_entries.TestSolveSnAdjointFixedSource.
.. test_duality_cross_group_source_detector) carries the @verifies marker
.. (wired at the A6 marker commit).

where the discrete pairing is the solution G-inner-product
:math:`\langle\cdot,\cdot\rangle_G` (the discrete realisation of the
phase-space integral): the detector reading computed from the forward
flux equals the source
weighted by the importance.  The proof is one line —
:math:`\langle\psi^*, q\rangle = \langle\psi^*, A_{\rm loss}\psi\rangle
= \langle A_{\rm loss}^{\dagger}\psi^*, \psi\rangle = \langle\Sigma_d,
\psi\rangle` — using only the defining adjoint relation
:math:`\langle\psi^*, A\psi\rangle = \langle A^{\dagger}\psi^*,
\psi\rangle`.  It is exact for the *continuous* operators, and — this
is the whole point of the route decision — it is exact for the
*discrete* operators too, at finite :math:`N` and :math:`h`, because
:math:`A_{\rm loss}^{\dagger}` is built as the exact transpose of the
discrete :math:`A_{\rm loss}` (:ref:`sn-adjoint-route`).  The
verification rows (P1.2, :ref:`sn-adjoint-verification`) exercise the
un-hideable case — source and detector in *different* energy groups
*and* regions — so the reciprocity forces the downscatter chain
through :math:`S^{\mathsf T}` and a dropped transpose reddens O(1).

The adjoint eigenproblem and the exact eigenvalue identity
----------------------------------------------------------

The forward criticality problem is :math:`A_{\rm loss}\,\psi =
\tfrac1k\,F\,\psi` with :math:`M = A_{\rm loss}^{-1}F` and
:math:`k = \lambda_{\max}(M)`.  The adjoint criticality problem is its
transpose,

.. math::

   A_{\rm loss}^{\dagger}\,\psi^* \;=\; \frac{1}{k^{\dagger}}\,
   F^{\dagger}\,\psi^* ,
   \qquad
   M^{\dagger} = \bigl(A_{\rm loss}^{\dagger}\bigr)^{-1}F^{\dagger},
   \qquad
   k^{\dagger} = \lambda_{\max}(M^{\dagger}) .

**The eigenvalue is unchanged:** :math:`k^{\dagger} = k`, an *exact
algebraic identity*.  With the Hilbert adjoint
:math:`A^{\dagger} = G^{-1}A^{\mathsf T}G` (and likewise
:math:`F^{\dagger} = G^{-1}F^{\mathsf T}G`),

.. math::

   M^{\dagger}
   = \bigl(G^{-1}A^{\mathsf T}G\bigr)^{-1}\!\bigl(G^{-1}F^{\mathsf T}G\bigr)
   = G^{-1}\,A^{-\mathsf T}F^{\mathsf T}\,G
   = G^{-1}\,\bigl(F A^{-1}\bigr)^{\mathsf T}\,G ,

so :math:`M^{\dagger}` is similar to :math:`(FA^{-1})^{\mathsf T}`,
whose spectrum equals that of :math:`FA^{-1}`, which equals that of
:math:`A^{-1}F = M` (the shared non-zero spectrum of :math:`XY` and
:math:`YX`).  Hence :math:`\operatorname{eig}(M^{\dagger}) =
\operatorname{eig}(M)` and :math:`k^{\dagger} = k` *for any invertible
metric* :math:`G`.

The **eigenVECTOR is not** :math:`\varphi`, however.  :math:`\varphi^*`
is the dominant *left* eigenvector of :math:`M`, equivalently the
dominant right eigenvector of :math:`(A^{\mathsf T})^{-1}F^{\mathsf T}`.
That the value is shared but the vector is not is the load-bearing
Mode-12 fact of this whole chapter: **a** :math:`k^{\dagger} = k`
**gate carries zero information about** :math:`\varphi^*`, so every
eigenvalue check is *designed-green* on the entire adjoint mutation
class.  The vector must be pinned separately, and by a reference of the
right structural form — the :math:`(A^{\mathsf T})^{-1}F^{\mathsf T}`
spectrum, **not** :math:`\operatorname{eig}(M^{\mathsf T})`
(:ref:`sn-adjoint-verification` records the factor-order trap in full).

.. _sn-adjoint-route:

The route decision — the exact discrete transpose
=================================================

This is the spine of the chapter.  Every property above — the exact
:math:`k^{\dagger} = k`, the reciprocity that holds at finite mesh, the
absence of any adjoint-specific solver code — is a consequence of a
single design choice about *when* to transpose.

Two routes to a discrete adjoint
--------------------------------

There are two ways to obtain a discrete adjoint solver, and they do
**not** commute:

#. **Discretise-then-adjoint** (the textbook continuous route).  Take
   the continuous adjoint equation :eq:`sn-adjoint-continuous` —
   :math:`\mu`-reversal, kernel transpose, :math:`\chi\leftrightarrow
   \nu\Sigma_f` swap, all written in continuous form — and discretise
   it from scratch with its own sweep, its own upwinding, its own
   angular closure.
#. **Adjoint-of-the-discrete** (the ORPHEUS route).  Take the *already
   discretised* forward operator :math:`A_{\rm loss}` and form its
   exact matrix transpose, wrapped in the phase-space metric:
   :math:`A_{\rm loss}^{\dagger} = G^{-1}A_{\rm loss}^{\mathsf T}G`.

The two agree only in the limit :math:`h\to 0`, :math:`N\to\infty`.  At
finite resolution they differ by a discretisation error, and route (1)
carries a subtle, expensive failure: the discretised-continuous-adjoint
operator is **not** the transpose of the discrete forward operator, so
the discrete reciprocity :eq:`sn-adjoint-duality` holds only to
:math:`\mathcal O(h^p)`, and :math:`k^{\dagger} \ne k` except in the
limit.  A perturbation-theory or GPT chain built on route (1) inherits
that inconsistency as a spurious first-order error term — the adjoint
is "correct" only asymptotically, exactly where it is least useful for
a sensitivity estimate on a real, coarse mesh.

Why ORPHEUS transposes the discrete operator
--------------------------------------------

ORPHEUS takes route (2): it poses the adjoint eigenproblem by
**DAGGER-ing the forward operator triple**, feeding
:func:`~orpheus.numerics.iteration.KEigenvalue` the daggered triple
``((L+C).H, (S+N2N+B).H, F.H)`` — the daggered RESOLVENT, gain, and
fission; the loss dagger is formed inside the posing — and running
the **unchanged**
:func:`~orpheus.numerics.eigenvalue.power_iteration`
(:func:`~orpheus.sn.solver.solve_sn_adjoint`).  The consequences are
exactly the properties route (1) cannot deliver at finite resolution:

* **Reciprocity holds EXACTLY at finite** :math:`N` **and** :math:`h`.
  :math:`A_{\rm loss}^{\dagger}` *is* the discrete transpose, so
  :math:`\langle\Sigma_d,\psi\rangle = \langle\psi^*,q\rangle` is an
  algebraic identity of the discrete system, not an
  :math:`\mathcal O(h^p)` approximation.
* :math:`k^{\dagger} = k` **is an exact algebraic identity**
  (derived above), not a converged agreement of two independent
  discretisations.
* **There is no adjoint-specific loop or sweep code anywhere.**  The
  operator algebra IS the implementation: ``.H`` is the exact discrete
  Hilbert adjoint of every leaf, composed through
  :meth:`OperatorSum.apply_transpose
  <orpheus.numerics.operator.OperatorSum>` /
  :meth:`OperatorProduct.apply_transpose
  <orpheus.numerics.operator.OperatorProduct.apply_transpose>`; the
  **swap law** :math:`(A^{\dagger})^{-1} = (A^{-1})^{\dagger}`
  (:eq:`loss-rep-adjoint-inverse-swap`) makes :math:`(L+C).\mathtt{H}`
  invertible by routing to :math:`(L+C).\mathtt{inverse()}.\mathtt{H}`,
  so the daggered inner solve rides the reverse-scan transpose sweep
  (:ref:`loss-rep-orientation-two-frames`) behind ``(L+C).H.inverse()``
  with no new machinery.

This is Cardinal Rule 2 (architecture) paying a physics dividend: the
adjoint is not a parallel implementation to keep in sync with the
forward one — it is the *same* operators, transposed.  A bug in the
forward streaming operator and its adjoint cannot silently diverge,
because there is only one operator.

The continuous route, kept only as a slab oracle
------------------------------------------------

Route (1) is not discarded — it is kept as a **verification oracle**.
On the 1-D slab, where the geometry is simple enough that the
:math:`\mu`-reversed continuous adjoint discretises cleanly, the
continuous route provides a *structurally independent* reference for
the discrete-transpose adjoint (the fuller-view-oracle exception of the
project's retirement discipline).  It is never a production path: a
user never solves the adjoint by flipping :math:`\mu` and re-sweeping.
The distinction matters because :math:`\mu`-reversal *looks* like the
adjoint and is the single most common way to build a
plausible-but-wrong one — which is the subject of the next section.

.. _sn-adjoint-three-transposes:

The three transposes — the recurring landmine
=============================================

The word "transpose" names three different objects in the adjoint
chain, and conflating any two of them produces a plausible-but-wrong
adjoint that passes every eigenvalue gate.  This is the **#1 landmine**
of the whole carve.  The symptom table (:ref:`sn-symptom-table`) routes
"the adjoint reciprocity gate reds" straight here.

(1) The Euclidean matrix transpose
----------------------------------

:math:`A^{\mathsf T}` — the plain linear-algebra transpose of the
discrete operator's matrix, carrying **no metric**.  It appears in two
places in ORPHEUS, both realised without ever forming the matrix:

* the **scattering group-transpose** :math:`S^{\mathsf T}`
  (:ref:`sn-scattering-adjoint`), deliberately Euclidean — the frame
  conjugation transposes leaf-by-leaf, closes #118;
* the **reverse-scan walk transpose** :math:`(L+C)^{\mathsf T}` — the
  loss representation's orientation axis
  (:ref:`loss-rep-orientation-two-frames`): reversed DAG order + face
  in↔out swap over the *same* per-ordinate cell graph.

The walk transpose is the object the loss-representation warning
contrasts against :math:`\mu`-reversal: reversing the *within-octant
cell order* is what gives :math:`(L+C)^{\mathsf T}`, **not** reflecting
the angular quadrature.  Both :math:`S^{\mathsf T}` and
:math:`(L+C)^{\mathsf T}` are category (1).

(2) The Hilbert (G-metric) adjoint
----------------------------------

:math:`A^{\dagger} = A.\mathtt{H} = G^{-1}A^{\mathsf T}G` — the
metric-weighted adjoint, the *physical* one, and the object the
daggered posing actually uses.  It rides *on top of* the Euclidean
transpose: the function space owns the metric :math:`G` and the ``.H``
wrapper (:class:`~orpheus.numerics.operator.LinearOperator`,
:ref:`operator-adjoint`) applies :math:`G` before and :math:`G^{-1}`
after the leaf's ``apply_transpose``.  The metric is the phase-space
measure (:eq:`g-adjoint-block-metric`):

.. math::

   G \;=\; \operatorname{diag}\bigl(G_{\rm bulk},\,G_{\rm trace},\,
   G_{\rm sd}\bigr),
   \qquad
   G_{\rm bulk} = V_{\rm cell}\,w_n,
   \quad
   G_{\rm trace} = |\Omega\cdot\hat n_f|\,w_n,
   \quad
   G_{\rm sd} = V_{\rm cell}
   \qquad(\text{sd = the starting-direction ray — the spelling the}
   \text{tests and the error catalog use}),

the bulk volume·weight block :math:`V_{\rm cell}\,w_n`, the
partial-current trace block :math:`|\Omega\cdot\hat n_f|\,w_n` (with a
pseudo-inverse on the singular grazing-ordinate trace), and — on a
carrying (sphere) mesh — the System-B ray block :math:`G_{\rm sd} =
V_{\rm cell}`.  The sweep itself carries **no metric code**: the metric
enters only at the space boundary, so the same ``.H`` wrapper serves a
flat spherical-harmonic metric and a composite ``FullField`` metric
alike.

(3) The continuous adjoint operator
-----------------------------------

The adjoint of the *continuous* transport operator
(:eq:`sn-adjoint-continuous`), whose discrete signature is
:math:`\mu`\ **-reversal** (reflecting the angular quadrature) plus the
continuous kernel transpose.  This is what the discretise-then-adjoint
route (1) would build.  ORPHEUS does **not** use it in production — it
survives only as the slab oracle.  The loss-representation warning's
"NOT :math:`\mu`-reversal and NOT the continuous transport adjoint" is
precisely the statement that the walk transpose (category 1) must not
be confused with category (3).

The taxonomy reconciles the two framings a reader will meet elsewhere:
the loss-representation carve's "three transposes" are {the walk's
Euclidean transpose, :math:`\mu`-reversal (which it identifies with
the continuous adjoint — one object in its framing), and the Hilbert
G-adjoint riding on top}, and the thin pre-A6 Key Facts named
the trio *Euclidean / Hilbert / walk-orientation* — the walk-orientation
transpose being simply the streaming realisation of category (1).  All
three framings are the same taxonomy: **Euclidean** (bare
:math:`A^{\mathsf T}`, including both :math:`S^{\mathsf T}` and the
walk), **Hilbert** (:math:`G^{-1}A^{\mathsf T}G`, riding on top), and
**continuous** (whose signature is :math:`\mu`-reversal).

The G-metric free-parameter lesson
----------------------------------

The most dangerous property of the Hilbert adjoint, and the one every
future adjoint session must internalise: **the metric** :math:`G` **is a
free parameter that no eigenvalue gate can ever see.**  Because
:math:`A.\mathtt{H} = G'^{-1}A^{\mathsf T}G'` is *metric-similar* to
:math:`A^{\mathsf T}` for **any** invertible :math:`G'`, the daggered
spectrum — and therefore :math:`k^{\dagger}` — is EXACTLY invariant
even under a **wrong** metric.  A metric bug is **k-blind but
vector-visible**.

This is not a hypothetical.  The ghost :math:`G_{\rm sd} \equiv 0`
defect (ERR-067) put the System-B seed rows in :math:`\ker G` and made
``A.H`` a *wrong* adjoint for any non-zero seed, while every eigenvalue
gate stayed green.  The sole catcher is the coupled-sphere
defining-law residual row (:ref:`sn-adjoint-verification`): dropping
:math:`G_{\rm sd} = V_{\rm cell} \to 1` leaves
:math:`|k^{\dagger}_{\rm mut} - k_{\rm fwd}| = 2.6\times10^{-11}`
(EXACTLY k-blind) while the vector residual reds O(1) at
:math:`2.35`.  **NEVER** certify a metric by an eigenvalue — pin the
adjoint *vector* against a metric built independently from raw mesh
data.

.. _sn-scattering-adjoint:

The scattering adjoint, free from the harmonic frame
====================================================

The loss composite's analytic adjoint is hard — sign-flipping the upwind
direction, transposing the M–M closure, re-deriving the per-level azimuthal
redistribution, each an AI-failure-mode trap.  It is carried as the
**orientation axis** of the loss-representation machinery
(:ref:`loss-rep-orientation-two-frames`): the Wave-O analytic
reverse-direction matvec, its swap law, and the deferral ledger — landed
after a dense-transpose interim.  The **scattering**
operator :math:`S` is the counterexample: campaign **#276 P3** (commit
``15185e5``, closes
`#118 <https://github.com/deOliveira-R/ORPHEUS/issues/118>`_) made
:math:`S^{T}` fall out **for free**, because :math:`S` is already written as
a harmonic-frame conjugation.

The modernised in-scatter source is ONE frame-conjugated operator
(:attr:`~orpheus.transport.operators.transfer.TransferOperator.full_transfer_kernel`):

.. math::
   :label: sn-scattering-adjoint-kernel

   \mathrm{full\_transfer\_kernel}
   \;=\; R \circ \Lambda_{\ell\ge 0} \circ M ,

.. (vv-status rationale) Representational identity: the frame-conjugation
   definition of the full ℓ≥0 in-scatter kernel (analysis M, moment-space
   Legendre transfer Λ, reconstruction R).  Its verifiable content —
   the frame form reproduces the independent scalar fast-path forward source —
   is the ``@pytest.mark.foundation`` gate
   ``tests/sn/operators/test_scattering_adjoint.py::TestFullScatterKernel::test_reproduces_forward_scattering_source``
   (rtol 1e-12); the gate is unwired, so the label stays ``documented``
   with the gate named here (wiring backlog: #309).
.. vv-status: sn-scattering-adjoint-kernel documented

where :math:`M` / :math:`R` are the angular frame's analysis /
reconstruction faces and :math:`\Lambda_{\ell\ge 0}` is the
per-:math:`\ell` moment-space group transfer
(:class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`),
carrying **both** the :math:`\ell = 0` in-scatter and the
:math:`\ell\ge1` redistribution so one analysis and one reconstruction
serve the whole scattering source.

⚠ **Two names in this equation moved on 2026-09-04 (#426 step 2), and a
memo may still carry either.**  The attribute is now
:attr:`~orpheus.transport.operators.transfer.TransferOperator.full_transfer_kernel`
(it was ``full_scatter_kernel``) and the middle factor's class is
:class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`
(it was ``LegendreMomentScattering``).  Both renames have the same
cause and it is not cosmetic: this composite is no longer
:math:`S`'s — it is the **transfer family's**, built identically for
:math:`S` and for :math:`N_{2n}`, which differ only in the yield
:math:`y_c` inside :math:`\Lambda_c` (:eq:`sn-n2n-transfer-binding`).
Nothing in the equations of this section changed; what changed is how
many operators they describe.

.. note::

   ⚠ **Since 2026-09-02** :math:`\Lambda` **and** :math:`\Lambda^{\mathsf T}`
   **read the LAYOUT off the angular head rather than assuming the
   rectangular one** (#429 / ERR-080). Both verbs take the operator's
   own head — its domain — and ask it for the degree-:math:`\ell` block
   and for how many axes precede the group axis; the group contraction
   is then selected by the head's RANK:
   ``"mfc...,fg->mgc..."`` for the real harmonics (rank 2, an
   :math:`m` axis in front of the group axis) and
   ``"fc...,fg->gc..."`` for the FLAT Legendre head a 1-D rule binds
   (rank 1). The harmonic specs are the former inline ones verbatim, so
   that path is bit-identical by construction; an unshipped rank is
   refused by name. ⛔ Before that, the loop spelled the :math:`m` axis
   into its ``einsum`` and its slicing, so a flat head would have
   contracted the **group** axis as if it were :math:`m` — silently.
   See :ref:`spaces-moment-head`.

Its transpose is therefore the product transpose

.. math::
   :label: sn-scattering-adjoint-kernel-transpose

   \mathrm{full\_transfer\_kernel}^{T}
   \;=\; M^{T} \circ \Lambda^{T} \circ R^{T},

.. (vv-status rationale) Structural / representational identity: the product
   transpose assembled from the leaf transposes (no per-geometry derivation).
   Its verifiable content is the Euclidean reciprocity ⟨kernel ψ, c⟩ =
   ⟨ψ, kernelᵀ c⟩, pinned by the ``@pytest.mark.foundation`` gate
   ``tests/sn/operators/test_scattering_adjoint.py::TestFullScatterKernel::test_full_kernel_euclidean_reciprocity``
   (scalar + LD trailing spectator) — foundation gates carry no
   ``verifies(...)`` by design.
.. vv-status: sn-scattering-adjoint-kernel-transpose documented

which :meth:`OperatorProduct.apply_transpose
<orpheus.numerics.operator.OperatorProduct.apply_transpose>` assembles from
the leaf transposes — the frame's :math:`M^{T}` / :math:`R^{T}` faces (landed
in the Frame/Basis carve) and the per-:math:`\ell` group transpose
:math:`\Lambda^{T}` — with **no per-geometry
derivation to verify** (the trap the streaming adjoint above could not
avoid).  :math:`\Lambda^{T}` is the ONLY group-asymmetric factor, which
is why the whole product transpose is one expression.

.. note::

   **What these two equations said until 2026-08-30, and why it
   changed.**  Both were written with a third factor:
   :math:`\mathrm{full\_scatter\_kernel} = R\circ(\Lambda_{\ell\ge 0} +
   N_{2n})\circ M`, the :math:`(n,2n)` multiplication channel summed
   with :math:`\Lambda` **inside** the conjugation so that one analysis
   and one reconstruction covered P0 + :math:`\ell\ge1` +
   :math:`(n,2n)` together.  That was a faithful description of the
   tree, and it was structurally a **bundling decision taken at the
   operator level**.  CS4c step 3 (design record §14.1) reversed it:
   :math:`(n,2n)` is now the first-class
   :class:`~orpheus.transport.operators.n2n.N2NOperator` and carries
   its own transpose (:ref:`sn-n2n-adjoint`), so the composite became
   scattering *only*.  The moment-space channel object survived that
   step — the :math:`\ell = 0` transfer ``N2NMomentOperator`` was still
   built and still tested; what it no longer did was ride inside
   :math:`S`'s composite.  Its remaining production-facing role was as
   the *algebra of record* for the fast path, pinning ``N2N.apply(ψ)``
   against ``(1/W)·frame.conjugate(N2NMomentOperator).apply(ψ)`` — the
   very conjugation these equations used to spell.

   ⭐ **And #426 step 2 (2026-09-04) closed the loop the other way.**
   Not by putting :math:`N_{2n}` back inside :math:`S`'s composite —
   the CS4c ruling stands, and the two terms remain two objects in two
   slots — but by making them two INSTANCES of this composite.  Each
   builds its own :math:`R\,\Lambda_c\,M` over its own channel's
   stack; ``N2NMomentOperator`` retired into
   :class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`,
   which is the :math:`\Lambda` of these very equations.  So the object
   the note above kept as an *oracle for a one-block special case* is
   now the general factor, and the gate that consumed it exercises
   :math:`L+1` blocks
   (``tests/sn/operators/test_n2n_operator.py::TestTheBindingAtTheSolveOrder::test_apply_equals_the_frame_conjugation_at_the_solve_order``).

The :term:`per-ordinate <ordinate>` adjoint scattering source is then

.. math::
   :label: sn-scattering-adjoint-source

   S^{T}\chi \;=\; \tfrac{1}{W}\,\mathrm{full\_transfer\_kernel}^{T}\,\chi ,

.. (vv-status rationale) Definitional identity: the per-ordinate adjoint
   scattering source (the producer-side 1/W transposing as the scalar it is).
   Its verifiable content is the LOAD-BEARING per-group Euclidean reciprocity
   ⟨Sψ,χ⟩=⟨ψ,Sᵀχ⟩ — the frame-form Sᵀ cross-checked against the structurally
   INDEPENDENT scalar fast-path S — plus the S.apply_transpose == (1/W)·kernelᵀ
   wiring gate, both ``@pytest.mark.foundation`` in
   ``tests/sn/operators/test_scattering_adjoint.py::TestFullScatterKernel``;
   both gates are unwired, so the label stays ``documented`` with the
   gates named here (wiring backlog: #309).
.. vv-status: sn-scattering-adjoint-source documented

the producer-side :math:`1/W` transposing as the scalar it is
(:math:`(A/W)^{T} = A^{T}/W`).
:class:`~orpheus.transport.operators.scattering.ScatteringOperator` now
reports ``is_adjointable = True`` (it has a working ``apply_transpose``),
and the old "no ``apply_transpose``" class-docstring confession is
retired.

**Forward fast-path, adjoint frame-path — and why the asymmetry is
principled.**  The production FORWARD source keeps the scalar fast-path
(:attr:`~orpheus.transport.operators.scattering.ScatteringOperator.isotropic_energy`
for P0 — since the CS4c §14.1 extraction the :math:`(n,2n)` term is the
first-class :class:`~orpheus.transport.operators.n2n.N2NOperator`, whose
lift and transpose ride the same producer-side combine — and the
per-:math:`\ell` redistribution body behind
:meth:`~orpheus.transport.operators.transfer.TransferOperator.apply`,
spelled ``build_aniso_source`` until #448)
for SI-sweep performance; the **adjoint** — not the hot path — rides the
validated frame form instead.  The two are thus structurally *different*
representations of the same operator, which is exactly what makes the
verification a genuine cross-check rather than a tautology: the per-group
Euclidean reciprocity
:math:`\langle S\psi, \chi\rangle = \langle\psi, S^{T}\chi\rangle`
(``tests/sn/operators/test_scattering_adjoint.py``,
``TestFullScatterKernel::test_S_euclidean_reciprocity``) pins the frame-form
:math:`S^{T}` against the *independent* scalar fast-path :math:`S`, and the
forward equivalence
:math:`(1/W)\,\mathrm{full\_transfer\_kernel}.\mathrm{apply} \equiv
S.\mathrm{apply}` holds to :math:`\sim 10^{-12}`.

.. note::

   This :math:`S^{T}` is the **Euclidean** transpose (the plain
   group-and-angle matvec adjoint), NOT the metric Hilbert adjoint
   :math:`S^{\dagger} = G^{-1}S^{T}G` — that angular-Gram weighting is the
   :attr:`~orpheus.numerics.operator.LinearOperator.H` wrapper's job.  The
   campaign and commit name it "S†" colloquially; the precise object the
   operator computes is the transpose.

This is the discrete scattering adjoint the SN adjoint chain builds on: the
adjoint flux :math:`\psi^{*}` solving :math:`(L+C-S-N_{2n})^{T}\psi^{*} =
q^{*}`,
adjoint-weighted homogenisation, perturbation theory, and detector
sensitivity all need :math:`S^{T}`.  Its companion forward step (campaign
**#276 P2**, commit ``dcea43a``) routes the SN forward *isotropic* source
through the same model-shared
:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicScattering`
(:math:`\Sigma_{s0}`) and
:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicN2N`
(:math:`\nu_{2n}\Sigma_{2n}`) operators (0-ULP bit-identical), so the
:math:`K_\mathrm{iso}` energy operator — which also assembles the
infinite-medium loss matrix (:ref:`direct-eigensolve-assembly`) — is one
cross-model source.  These model-shared operators live in
:mod:`orpheus.transport.operators`.  Since CS4c step 3 the two are
composed **by the solver**, at the one within-group construction site
(:func:`~orpheus.sn.coupled_system.build_within_group_system`), as
:attr:`S.isotropic_energy
<orpheus.transport.operators.transfer.TransferOperator.isotropic_energy>`
``+`` ``N2N.isotropic_energy`` rather than being read off a bundling
accessor on :math:`S` — the subsection below is why.  Both halves are
now spelled the same way, because both are the SAME accessor on the
shared transfer core (#426 step 2, 2026-09-04); the :math:`(n,2n)` side
read ``N2N.energy`` until that date, when the channel still had a class
of its own to hang a differently-named accessor on.  Note what the
accessor is and is not: it is the :math:`\ell = 0` **energy** binding of
each gain, which is what the ray seed's emission needs (:math:`\ell = 0`
by physics, not by truncation), and it is unaffected by the operator's
Legendre order.

.. _sn-n2n-adjoint:

The (n,2n) adjoint — the lift is its own reversal
-------------------------------------------------

**Why the channel is no longer inside** :math:`S`.  Until 2026-08-30 the
:math:`(n,2n)` emission was an unnamed passenger of the scattering
operator: it rode :math:`S`'s isotropic accumulator on the forward fast
path and the :math:`\Lambda + N_{2n}` sum inside
:eq:`sn-scattering-adjoint-kernel` on the adjoint path.  The ruling that
extracted it (CS4c design record §14.1) is a statement about *where a
bundling decision may be taken*, and it is worth stating in full because
it generalises past this channel:

   :math:`(n,2n)` is **scattering-like** — a group-to-group transfer
   which in principle carries its own anisotropy — **and
   production-like** — it carries a multiplicity :math:`\nu_{2n}`.
   Which of the two it should be grouped with therefore depends on the
   question being asked: with :math:`S` when scattering anisotropy is
   the axis of interest, with :math:`F` when production accounting is.
   A **context-dependent bundling must not be decided at the operator
   level**, because an operator that hard-codes one grouping makes the
   other unspellable.

The ruling's *"in principle"* was a hedge when it was written, became a
**measurement** on 2026-09-03, and is **what ships** since 2026-09-04.
The evaluated data ORPHEUS itself ships carries seven Legendre moments
for this channel; #426 step 1 carried all seven through to
``Mixture.Sig2``; #426 step 2 made the binding read them at the solve's
order.  The ruling is *strengthened* by that — the anisotropy axis it
declined to foreclose is not merely real, it is the axis that carries a
few hundred :math:`10^{-5}` of :math:`k` on a Be-reflected fast system
(:ref:`the measured block <sn-n2n-p0-truncation-measured>`) — and the
quote stays verbatim because it is the record of what was argued on
2026-08-30.

⭐ And the ruling's own logic is what made the repair small.  Because
the bundling was NOT decided at the operator level, restoring the
anisotropy did not have to unpick a fused source: the two terms were
already two objects in two slots, so the change is that they became two
**instances of one binding** rather than two classes with parallel
arithmetic.  :math:`\Lambda` gained a yield, the frame gained an order,
and every equation below kept its shape.

So the within-group algebra spells the channel out,

.. math::
   :label: sn-within-group-with-n2n

   A \;=\; L + C - S - N_{2n} - B ,

.. (vv-status rationale) Structural identity: the within-group loss
   composite's member list after the CS4c §14.1 extraction — a
   composition-site fact, not a solver claim.  Its verifiable content is
   that the shipped builder composes exactly these members and that the
   composed pair reproduces the pre-extraction fused source: the
   ``@pytest.mark.foundation`` gates
   ``tests/sn/operators/test_n2n_operator.py`` (the lift, its transpose,
   the carrier arms) and
   ``tests/sn/operators/test_scattering_operator.py::TestAnisoMomentSourcePath``
   (``S.apply + N2N.apply`` against the frozen pre-extraction snapshots).
.. vv-status: sn-within-group-with-n2n documented

and any bundling is a solver-side
:class:`~orpheus.numerics.operator.OperatorSum` grouping — which is
exactly what the 1-D diffusion solver does, summing
:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicScattering`
with
:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicN2N`
into the one :math:`S` its :math:`A = L + C - S - B` needs.  The two
solvers now disagree about the grouping *in the composition*, where a
disagreement is legible, instead of agreeing inside an operator that had
chosen for both.

.. _sn-n2n-p0-truncation:

.. warning::

   **HISTORY (2026-04 – 2026-09-04): ORPHEUS modelled** :math:`(n,2n)`
   **emission as isotropic.  The reaction is not.**  Read this whole
   block in the past tense: since #426 step 2 (2026-09-04) the
   :math:`(n,2n)` binding reads the tape's Legendre stack at the solve's
   order, exactly as :math:`S` does, and every equation in this
   subsection is stated per-:math:`\ell` below.  The block is kept —
   rather than deleted — for three reasons: the **tape facts** it
   measures are unchanged and are the evidence that the model was wrong;
   the ``Sig2[0]`` reads it enumerates as :math:`\ell = 0` **by
   physics** are still there and must **not** be "fixed"; and a reader
   may still meet the model in an older checkout or a quoted memo.  The
   defect is catalogued as ERR-082 (:ref:`the L0 error catalogue
   <theory-verification-error-catalog>`).

   **Where the truncation lived, and how it moved TIERS before it
   died.**  It began as a data-layer loss.  #426 step 1 (2026-09-03)
   made the ingest lossless in :math:`\ell`: ``Isotope.sig2`` and
   :attr:`~orpheus.data.macro_xs.mixture.Mixture.Sig2` carry every
   Legendre order the tape stores, exactly as
   :attr:`~orpheus.data.macro_xs.mixture.Mixture.SigS` does
   (:ref:`the ingest stack note <n2n-legendre-stack-at-ingest>`).  That
   left **three** operator-tier sites, and #426 step 2 closed two of
   them and re-read the third:

   .. list-table:: The three sites of the operator-tier model, and their fate
      :header-rows: 1
      :widths: 34 40 26

      * - site (as of 2026-09-03)
        - what it did
        - fate at step 2
      * - ``N2NKernel.from_mixture``
          (``orpheus/transport/kernels.py``)
        - densified ``mixture.Sig2[0]`` and stored it as a single
          ``matrix`` — the kernel had no :math:`\ell` stack, where
          ``ScatteringKernel`` did
        - **retired.**  Both channels are now
          :class:`~orpheus.transport.kernels.TransferKernel` — one
          Legendre stack plus a yield; the constructor is
          :meth:`~orpheus.transport.kernels.TransferKernel.n2n`
      * - ``N2NOperator.from_solver_data``
          (``orpheus/transport/operators/n2n.py``)
        - minted the binding frame at
          ``HarmonicFrame.for_space(interior, 0)`` — order 0, where
          ``ScatteringOperator`` passed the solve's ``scattering_order``
        - **retired.**  The tier-2 mint is now the shared core's
          (:meth:`~orpheus.transport.operators.transfer.TransferOperator.from_solver_data`);
          it takes ``scattering_order`` and mints the SAME interned
          frame for both terms, the role supplying only which
          ``Mixture`` channel to read
      * - ``MaterialXSField._build_dense_caches``
          (``orpheus/transport/mesh/material_xs_field.py``)
        - cached ``mix.Sig2[0]`` dense for
          :meth:`~orpheus.transport.mesh.material_xs_field.MaterialXSField.n2n_matrix`,
          beside a scattering cache that keeps **every** order of
          ``mix.SigS``
        - **kept, and it is not a truncation.**
          :meth:`~orpheus.transport.mesh.material_xs_field.MaterialXSField.n2n_matrix`
          feeds the removal / fold predicate, which is a reaction rate:
          :math:`\ell = 0` by physics.  The gain no longer reads it

   ⛔ **This block asserted TWO superseded things, and the sentences are
   worth preserving because a reader may still meet either.**  (i) Until
   2026-09-03 it read, verbatim, that the truncation was "taken at
   ingest and unrecoverable downstream: the GENDF reader parses the
   whole Legendre stack of the MF=6/MT=16 section and then keeps
   ``sig2_data[(0, 0)]`` alone", so that "``Isotope.sig2`` and
   ``Mixture.Sig2`` are ONE matrix where ``Mixture.SigS`` is a list over
   :math:`\ell`".  Step 1 repealed both halves: that subscript no longer
   exists and any consumer can read ``Mixture.Sig2[1]``.  (ii) From
   2026-09-03 to 2026-09-04 it read that the model lived at "TWO LINES
   of the transport layer" and that restoring the moments **through**
   the operator was "step 2 of #426".  Step 2 landed on 2026-09-04
   (``1a3b78ec``); the two lines are gone.

   ⚠ **Every surviving** ``Sig2[0]`` **read is** :math:`\ell = 0` **BY
   PHYSICS and must not be "fixed" with the gain.**  This is the block
   that stays LIVE.  ``[M]`` 2026-09-04, re-censused on the post-carve
   tree by AST over ``orpheus/`` (subscript-``0`` reads of a
   ``Sig2``/``sig2`` attribute or name; docstrings and comments are not
   AST nodes and so cannot be counted): **7 sites, and all 7 are
   correct** —

   * a **reaction rate** is the :math:`P_0` row sum, and every higher
     Legendre moment integrates to zero over angle, so it contributes
     exactly nothing: ``mixture.py:169``
     (:attr:`~orpheus.data.macro_xs.mixture.Mixture.n2n_out_xs`),
     ``mixture.py:703`` (``compute_macro_xs``'s
     :eq:`sigT-computed`) and ``gendf.py:472`` (the isotope-level
     :math:`\Sigma_{\rm t}` accumulation) — ×3;
   * those families' emission sources are isotropic **by
     construction** — a property of the *method*, not of this channel,
     and each site carries an inline comment saying so:
     ``cp/solver.py:514``, ``moc/core.py:96``, ``mc/solver.py:378`` —
     ×3;
   * ``material_xs_field.py:682``, the dense cache behind
     :meth:`~orpheus.transport.mesh.material_xs_field.MaterialXSField.n2n_matrix`
     — ×1.  ⭐ **This one changed COLUMN without changing line.**  It was
     the third model site while the gain read it; since #426 step 2 the
     gain reads a
     :class:`~orpheus.transport.kernels.TransferKernel` instead, and the
     only consumers left are the removal term and the :math:`\sigma_r`
     fold predicate — both reaction rates.  The same expression is now
     right for a different reason.

   ⛔ **Two corrections to the census this block carried until
   2026-09-04**, both found by re-running it rather than by reading it.
   (i) It reported **9** sites — 2 model + 7 physics — and named the
   third model site (the frame's order) as one the predicate
   *structurally cannot see*, which was true and remains the right
   caution: a census keyed on a subscript cannot return a site that
   passes an ORDER.  (ii) It listed, among the seven, a
   ``gendf.py`` guard spelled ``if sig2[0].nnz > 0``.  ``[M]`` **no such
   guard exists**, at HEAD or at either parent of the two #426 steps —
   ``grep -rn nnz orpheus/`` returns only a docstring, a ``repr`` and
   the HDF5 schema.  The seventh physics site was, and is, the dense
   cache above.  The bullet's *count* was right and one of its
   *members* was not, which is what a re-run separates and a re-read
   does not.

   **What is truncated is not small.**  Measured over the 13
   NJOY-GROUPR GENDF files ORPHEUS ships
   (``orpheus/data/micro_xs/*.GXS``, 421 groups, T = 293.6 K, read with
   the project's own parser): MF=6/MT=16 stores **NL = 7** Legendre
   moments — the same order as elastic scattering, which stores 7 in
   13 of 13 files — on **10 of the 11** files that carry the section
   (NL = 1 for Na-23 alone; the section is absent for B-10 and H-1).
   ``NL`` tracks the *evaluation*, not a processing request: one file —
   and a GENDF file is the output of a single GROUPR run — carries three
   different values, ``NA023.GXS`` giving MT=2 → 7, MT=51 → 7 and
   MT=16 → 1, which no global Legendre-order request can produce.  On Be-9 all 8195 transfer entries are non-zero at **every**
   :math:`\ell = 1 \ldots 6`, with :math:`\lVert P_1 \rVert_\infty /
   \lVert P_0 \rVert_\infty = 0.690` and a mean emission cosine
   :math:`\bar\mu = \sigma_1/\sigma_0 = +0.278` summed over the 50
   incident groups where the channel is open.  Be-9 carries **no**
   inelastic MF=6 section at all, so ``elastic + 2·(n,2n)`` is its
   complete fast emission source and the share is exact rather than a
   subset: :math:`(n,2n)` supplies a **median 62 %** of the
   :math:`P_0` emission source and a **median 45 %** of the
   :math:`P_1` source — and it was the second that ORPHEUS dropped.
   ``[M]`` re-measured 2026-09-04 on the post-carve tree with
   ``load_isotope("BE009", 294)``: the numbers above are unchanged
   (``NL = 7``; ``nnz`` **8195 at every one of the seven** orders;
   :math:`\lVert P_1\rVert_\infty/\lVert P_0\rVert_\infty =
   0.68969`; :math:`\bar\mu = +0.27825` over 50 live incident groups;
   rank 50, best rank-1 relative error 0.5818).  They are properties of
   the TAPE and no ORPHEUS change can move them, which is exactly why
   they are the evidence that the model was wrong.  ⚠ The equal-``nnz``
   reading is an **isotope** property, not an ingest property: on U-235
   the same census reads ``6067 / 6067 / 5834 / 5334 / 3165 / 2773 /
   1887``, i.e. genuine exact zeros the evaluation carries.

   **It is not fission-like either, and that is the reason** :math:`F`
   **and** :math:`N_{2n}` **must not be collapsed.**  There is no
   :math:`\chi`-like emission spectrum to factor out: fission's
   MF=6/MT=18 carries a distinguished incident-energy-INDEPENDENT
   ``ig = 0`` record — which is exactly what makes the rank-1 dyad
   :math:`\chi \otimes \nu\Sigma_f` faithful — while MT=16 carries none,
   and Be-9's :math:`\ell = 0` matrix has numerical rank **50** (every
   live incident group) with a best rank-1 relative error of **58 %**.
   ⛔ The last sentence of this paragraph read, until 2026-09-04, *"the
   two operators' present code similarity is a coincidence of THIS
   truncation, not of the physics"* — and the prediction was exactly
   right, in a way the sentence could not have anticipated: #426 step 2
   dissolved the similarity, but **not** by collapsing :math:`N_{2n}`
   into :math:`F`.  It collapsed :math:`N_{2n}` into :math:`S`, where
   the shared structure is real — one non-separable Legendre transfer
   stack with a yield — and left :math:`F` a type of its own, because a
   rank-1 dyad under a :math:`1/k` morphism is a different object.  The
   split the tape argues for is :math:`\{S, N_{2n}\} \mid \{F\}`, and
   that is what ships.

   *Instrument control.*  Elastic :math:`\sigma_1/\sigma_0` at low
   energy reproduces the analytic s-wave :math:`\bar\mu = 2/(3A) =
   0.074615` (Be-9, ``AWR = 8.93478``) to six significant figures,
   which validates the extraction and pins the GENDF convention —
   stored moments carry no :math:`(2\ell+1)` factor, so
   :math:`\sigma_1/\sigma_0` IS the mean lab cosine.

   ⚠ **The 62 % / 45 % figures are cross-section shares, not an error
   in any flux or eigenvalue.**  That distinction was, until
   2026-09-03, the whole honest scope of this warning — no transport
   solve had been run.  One was; it is the next block, and it is a
   different and stronger kind of claim.  The full data-layer taxonomy
   is at `#426 <https://github.com/deOliveira-R/ORPHEUS/issues/426>`_.

.. _sn-n2n-p0-truncation-measured:

.. important::

   **What the** :math:`P_0` **model was worth, measured in** :math:`k`.
   ``[M]`` 2026-09-03 on ``main`` ``1e02f6b1`` — the PRE-carve tree.
   The :math:`\ell = 1\ldots6` MT=16 moments are read off the tape with
   the production parser, yield-stripped with the same per-row diagonal
   production applies at :math:`\ell = 0`, and injected as **scattering**
   moments with multiplicity 2 —
   :math:`\Sigma_{s,\ell} \mathrel{+}= 2\,\Sigma_{2n,\ell}` for
   :math:`\ell \ge 1` only, so the :math:`\ell = 0` channel stayed on
   :math:`N_{2n}` and nothing was double-counted.  This was a *probe* of
   what step 2 would change; the shipped path reproduces its first row
   with no probe at all — :ref:`the shipped ladder
   <sn-n2n-anisotropy-shipped-ladder>` below.

   .. list-table::
      :header-rows: 1
      :widths: 30 20 17 17 16

      * - fixture (Be-9 reflector \| core \| Be-9)
        - :math:`k` shipped
        - :math:`\Delta k\cdot10^{5}`
        - :math:`\frac{\Delta k}{k_0}\cdot10^{5}`
        - :math:`\Delta\rho\cdot10^{5}`
      * - fast, **3 cm** reflectors: U-235 metal 4 cm
          (:math:`N = 0.04894`); 12/16/12 cells
        - ``1.095322188``
        - **−413.55**
        - **−377.56**
        - **−346.01**
      * - fast, **10 cm** reflectors; same core; 40/16/40
        - ``1.526231521``
        - **−529.26**
        - **−346.78**
        - **−228.00**
      * - thermal, 10 cm reflectors: U-235 :math:`5\!\times\!10^{-4}`
          + H 0.0669 + O 0.0334, 30 cm; 40/60/40
        - ``1.745071904``
        - **−155.61**
        - **−89.17**
        - **−51.15**

   Configuration: 1-D slab, vacuum both sides, ``gauss_legendre(8)``,
   421 groups, ``keff_tol = 1e-9``, ``flux_tol = 1e-8``,
   ``inner_tol = 1e-10``, every arm ``fully_converged``; pure-isotope
   mixtures at 294 K; the scattering stacks held at :math:`P_2` in both
   arms, so the ladder isolates the :math:`(n,2n)` anisotropy alone.
   Rows are the :math:`\ell = 1` arm; :math:`\ell = 2\ldots6` add under
   :math:`2\cdot10^{-5}` in :math:`\Delta k` on every fixture — **the
   dipole carries essentially all of it**, which is what a
   forward-peaked emission should do.

   *Sign, and why it is the physics.*  Adding the true forward peak
   makes :math:`k` **fall**: the emitted pair leaves the reflector
   outward rather than isotropically, so less returns to the core.  The
   controls agree.  Flipping the sign of the injected moments moves
   :math:`k` the other way by a comparable magnitude
   (``+409.77`` / ``+520.83`` / ``+152.43`` in :math:`\Delta k\cdot10^5`
   — linear, as a first-order perturbation must be); zero-padding the
   shipped mixtures to :math:`L = 6` without adding moments moves
   :math:`k` by **0.00**; and restricting the injection to the
   *reflector alone* reproduces the effect to
   :math:`0.20\cdot10^{-5}` (``−412.05`` against the same arm's
   ``−412.25``, both at :math:`\ell \le 2`), so **99.95 % of it is the
   reflector's**.  That is where the channel does its work: the
   :math:`(n,2n)` neutrons that matter are the ones a reflector would
   otherwise send *back*, and Be-9's MT=16 is open over ``[M]`` **50**
   incident groups against U-235's **22**.  Flux changes are of the
   same order: relative :math:`L_2` of the normalised 421-group flux
   :math:`1.8`/:math:`3.4`/:math:`2.2 \times 10^{-3}`, concentrated in
   the reflector.

   ⚠ **Quote the convention with the number.**  ``pcm`` is
   :math:`10^{-5}` and says nothing about what was divided by what.
   The three columns differ by :math:`k_0`, and on this fixture set the
   choice is not cosmetic: **in** :math:`\Delta k` **the thick
   reflector looks worse than the thin one (−529 vs −414), and in**
   :math:`\Delta\rho` **it looks better (−228 vs −346)** — the
   comparison inverts.  Any sentence ranking these fixtures against one
   another is a statement about a convention as much as about the
   physics.

   *Provenance.*  Every derived column above was re-computed for this
   page from the recorded :math:`k` values and reproduces the source
   table exactly, in all three conventions, on all three fixtures.  The
   effect itself was independently reproduced by a second agent on its
   own instrument and own pipeline, to every published digit, with the
   two shared conventions closed against **physics** rather than
   against the probe: strict upper-triangularity of the energy-losing
   transfer matrix (8195 of 8195 entries) and the entrywise bound
   :math:`|\Sigma_\ell|/\Sigma_0 = |\langle P_\ell\rangle| \le 1`,
   where a stray :math:`(2\ell+1)` on :math:`\ell = 1` would read 2.9.

   ⛔ **What this does not license.**  Three fixtures, one geometry,
   one quadrature, one library.  It establishes that the :math:`P_0`
   model is a **defect and not a documentable approximation** on
   systems ORPHEUS is meant to solve — a few hundred :math:`10^{-5}` on
   a Be-reflected fast system, still 50–150 behind a water-moderated
   core — and it does **not** establish a magnitude for any other
   problem class.  A beryllium reflector is close to the worst case in
   this library.

.. _sn-n2n-anisotropy-shipped-ladder:

.. important::

   **The shipped ladder — what the LIBRARY now returns, with no probe.**
   ``[M]`` 2026-09-04 on ``1a3b78ec``, ONE fixture (the §0 fast/thin
   row above: Be 3 cm | U-235 metal 4 cm | Be 3 cm, 12/16/12 cells,
   ``gauss_legendre(8)``, 421 groups, ``keff_tol = 1e-9``,
   ``flux_tol = 1e-8``, ``inner_tol = 1e-10``, every arm
   ``fully_converged``).  The arms differ ONLY in how many
   :math:`(n,2n)` Legendre orders the mixture carries; the **elastic**
   stacks are at :math:`P_2` in every arm, which is what makes the
   ladder the :math:`(n,2n)` anisotropy alone.

   .. list-table::
      :header-rows: 1
      :widths: 26 30 15 15 14

      * - arm
        - :math:`k`
        - :math:`\Delta k\cdot10^{5}`
        - :math:`\frac{\Delta k}{k_0}\cdot10^{5}`
        - :math:`\Delta\rho\cdot10^{5}`
      * - :math:`(n,2n)` at :math:`P_0` (the retired model)
        - ``1.0953221881419453``
        - —
        - —
        - —
      * - :math:`(n,2n)` at :math:`\ell \le 1`
        - ``1.0911866898558749``
        - **−413.55**
        - **−377.56**
        - **−346.01**
      * - :math:`(n,2n)` at :math:`\ell \le 2`
        - ``1.0911996566537725``
        - **−412.25**
        - **−376.38**
        - **−344.92**
      * - :math:`(n,2n)` at :math:`\ell \le 6`
        - ``1.0911996566537725``
        - **−412.25**
        - **−376.38**
        - **−344.92**
      * - :math:`L = 0` solve (isotropic control)
        - ``1.1587120371368607``
        - **0.00**
        - **0.00**
        - **0.00**

   Three readings, and the third is the one that is easiest to
   mis-state:

   1. **The** :math:`P_0` **row is bit-identical to the pre-carve
      record**, digit for digit
      (``1.0953221881419453``), as is the :math:`L = 0` solve
      (``1.1587120371368607``).  The carve moved the :math:`(n,2n)`
      answer and nothing else.
   2. **The ladder has converged by** :math:`\ell = 1`: :math:`\ell\le2`
      differs from :math:`\ell\le1` by :math:`1.30\cdot10^{-5}` in
      :math:`\Delta k`, which is the dipole carrying essentially all of
      a forward-peaked emission's effect.
   3. ⚠ **The** :math:`\ell\le6` **row equals** :math:`\ell\le2` **to
      the BIT, and that is not convergence** — it is the solve's own
      order.  Every arm runs at ``scattering_order = 2``, so
      :math:`\Lambda` has three blocks and a :math:`\ell \ge 3` moment
      is never read.  A reader who quotes the :math:`\ell\le6` row as
      *"and higher orders add nothing"* has quoted the truncation, not
      the physics.  (What the :math:`\ell\le1 \to \ell\le2` step
      shows is real, because :math:`\ell = 2` IS inside the solve's
      order.)

   ⭐ **The number is a RECORD, and it is pinned as one.**  There is no
   structurally-independent eigenvalue reference for a 421-group
   heterogeneous reflected slab — the only physics-sourced claim in the
   gate is the **sign**: :math:`\bar\mu` of Be-9's MT=16 is positive on
   50 of 50 live groups, so the emitted pair continues outward, less
   returns to the fuel, and :math:`k` must FALL.  The gate is
   ``tests/sn/verification/analytical/test_be_reflected_n2n_anisotropy.py``
   (``@pytest.mark.l2``, deliberately **not** ``slow`` — 25 s inside a
   ≥90-minute gate), and it is the catcher for ERR-082.

   ⛔ **What this ladder is blind to, stated so nobody over-reads it.**
   Every arm holds the solve at ``scattering_order = 2``, and #426
   step 1 made the ingest carry the elastic :math:`P_3\ldots P_6` the
   tape stores.  ``[M]`` 2026-09-04, this page's own re-measurement on
   the post-carve tree (same fixture, same tolerances, all seven arms
   ``converged``), :math:`k` against the :math:`L = 2` row:

   .. list-table::
      :header-rows: 1
      :widths: 10 34 28 28

      * - :math:`L`
        - :math:`k`
        - :math:`\Delta k\cdot10^{5}`
        - :math:`\frac{\Delta k}{k_2}\cdot10^{5}`
      * - 0
        - ``1.1587120371368607``
        - +6751.24
        - +6186.99
      * - 1
        - ``1.0771093258323200``
        - −1409.03
        - −1291.27
      * - 2
        - ``1.0911996566537725``
        - —
        - —
      * - 3
        - ``1.0888490194116345``
        - −235.06
        - −215.42
      * - 4
        - ``1.0895262273204490``
        - −167.34
        - −153.36
      * - 5
        - ``1.0894040682370552``
        - −179.56
        - −164.55
      * - 6
        - ``1.0894248749300022``
        - −177.48
        - −162.65

   So the rows above are the :math:`(n,2n)` anisotropy **at**
   :math:`P_2`, never "the converged anisotropy answer": the elastic
   :math:`\ell \ge 3` moments are worth more than the :math:`(n,2n)`
   :math:`\ell \ge 1` ones on this fixture, and the two are not
   separable at a common order.  Whether the solve's order should be
   raised is a different question, tracked separately; that
   ``scattering_order`` is the ONLY remaining truncation is what
   ``tests/sn/solve/test_scattering_order_is_the_only_truncation.py``
   pins.

   ⚠ **Do not quote step 1's version of this ladder here.**  It read
   :math:`-229 / -163 / -175 / -173` at :math:`L = 3\ldots6` and was
   correct **on the pre-step-2 tree**, where the :math:`L = 2` baseline
   was ``1.0953221881419453`` and the :math:`(n,2n)` moments never
   entered at any order.  Both legs of that ratio moved on 2026-09-04,
   so the numbers above are a different measurement of a different
   tree, not a re-rounding of the same one.

**The forward action, per** :math:`\ell`.  The :math:`(n,2n)` gain is
the angular binding of its channel's Legendre stack at the solve's
order — the SAME product :math:`S` has always been, over a middle
factor that carries the channel's yield:

.. math::
   :label: sn-n2n-transfer-binding

   N_{2n} \;=\; \tfrac1W\,R\,\Lambda_{2n}\,M ,
   \qquad
   \Lambda_{2n} \;=\; \nu_{2n} \sum_{\ell=0}^{L}
                     \mathbf{P}_\ell \otimes \Sigma_{2n,\ell} ,
   \qquad \nu_{2n} = 2 ,

.. (vv-status rationale) Structural identity: the (n,2n) gain written on
   the ONE transfer-binding shape #426 step 2 installed — a
   composition-site fact about how the binding is built (which stack,
   which yield, which order), not a solver claim.  Its verifiable
   content is threefold and all three are ``@pytest.mark.foundation`` in
   ``tests/sn/operators/test_n2n_operator.py::TestTheBindingAtTheSolveOrder``:
   the realized operator reproduces its own conjugated product at the
   solve's order (``test_apply_equals_the_frame_conjugation_at_the_solve_order``),
   the ℓ = 1 moment REACHES the action (``test_the_first_moment_reaches_the_action``
   — the activation leg a P0 twin would leave at exactly 0.0), and the
   two terms differ by the yield ALONE, bit-exactly
   (``test_the_two_terms_differ_by_the_yield_alone``: ``N2N.apply(ψ) ==
   2·S'.apply(ψ)`` over the same stack — exact because scaling by 2 is
   exact in binary floating point).  The EIGENVALUE consequence is
   ``@pytest.mark.l2`` in
   ``tests/sn/verification/analytical/test_be_reflected_n2n_anisotropy.py``.
.. vv-status: sn-n2n-transfer-binding documented

with :math:`\mathbf{P}_\ell` the projector onto the degree-:math:`\ell`
harmonic block and :math:`\Sigma_{2n,\ell}` the per-material
group-to-group matrix of that order.  Setting :math:`\nu_{2n} \to 1`
and :math:`\Sigma_{2n,\ell} \to \Sigma_{s,\ell}` gives :math:`S`
exactly; there is no third form on this page, and no
:math:`(n,2n)`-specific arithmetic anywhere below it.

⛔ **Until 2026-09-04 this section read that the kernel was a single**
:math:`\ell = 0` **transfer matrix** :math:`K = \nu_{2n}
\Sigma_{2n}^{\mathsf T}` **per cell**, that the single :math:`\ell` was
"a *choice the kernel makes*", and that restoring the moments was "what
step 2 changes".  All three were true when written; step 2 landed.  The
prediction the old text made about itself was also right, and is worth
keeping because it is the reason this section barely moved: *"very
little below has to be re-derived, and the reason is structural rather
than lucky — the transpose here is not hand-written arithmetic but the
factor reversal of a frame conjugation, and a product reversal does not
care how many* :math:`\ell`\ *-blocks the middle factor has."*  What
was written as a one-block conjugation is now the :math:`L`-block one
:math:`S` already used
(:eq:`sn-scattering-adjoint-kernel-transpose`), with the same
derivation and not one new line of algebra.

**The** :math:`\ell = 0` **half is still the isotropic lift, and it is
still the fast path.**  Write :math:`K = \nu_{2n}\Sigma_{2n,0}^{\mathsf
T}` for the :math:`\ell = 0` block's per-cell emission matrix.  That
block's composite action is the **isotropic lift** of an energy
operator,

.. math::
   :label: sn-n2n-isotropic-lift

   \bigl(N_{2n}\,\psi\bigr)_{n,g}
   \;=\; \frac{1}{W}\,\sum_{g'} K_{g'\to g}\,
         \underbrace{\sum_{m} w_m\,\psi_{m,g'}}_{\textstyle \phi_{g'}} ,
   \qquad W \;=\; \sum_m w_m ,

.. (vv-status rationale) Definitional identity: the ℓ = 0 BLOCK of the
   (n,2n) composite action, written as the isotropic lift of its energy
   binding (the ℓ=0 frame conjugation realized on the reaction-rate fast
   path).  It is the P0 half of :eq:`sn-n2n-transfer-binding`, which is
   the operator's full statement; this equation is a block of that one
   and is stated separately because it is the block the FAST PATH
   evaluates.  Its verifiable content is the identity with the
   conjugation it realizes at the solve's order,
   ``N2N.apply(ψ) == (1/W)·frame.conjugate(Λ₂ₙ).apply(ψ)``, pinned by
   the ``@pytest.mark.foundation`` gate
   ``tests/sn/operators/test_n2n_operator.py::TestTheBindingAtTheSolveOrder::test_apply_equals_the_frame_conjugation_at_the_solve_order``.
   ⚠ Scope, stated so the row is not over-read: that gate's fixture runs
   at L = 1, so it pins the FULL conjugation, of which this equation is
   the ℓ = 0 block — it is a necessary condition on this equation and
   not a gate written for it, and no shipped gate isolates the ℓ = 0
   block alone.  The block's own arithmetic (the P0 verb, yield applied
   once) is pinned at the term tier by
   ``tests/transport/test_material_field.py``.
.. vv-status: sn-n2n-isotropic-lift documented

which is *literally* the frame's :math:`\ell = 0` conjugation
:math:`\tfrac{1}{W}\,R_0\,K\,M_0` — measured on the shipped faces, the
analysis face's :math:`\ell = 0` row **is** the weight vector
:math:`w_n` (the no-prefactor :math:`Y_0^0 = 1` convention,
:doc:`/theory/foundations/spherical_harmonics`) and the reconstruction
face's :math:`\ell = 0` column **is** the all-ones broadcast.  The
operator evaluates this block on the reaction-rate fast path (one
``einsum`` over groups, through the P0 energy binding
:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicN2N`,
with no moment tensor allocated) and reconstructs the
:math:`\ell \ge 1` blocks through the frame, combining the two in the
one producer-side :math:`(\text{iso}/W) + \text{aniso}` verb.  That is
the *algebra eager, performance lazy* ruling: the algebra is stated and
gated as the conjugation of :eq:`sn-n2n-transfer-binding`, the
:math:`\ell = 0` evaluation takes the cheap route, and the gate above is
what keeps the two from drifting.

⚠ **Ranking the two equations.**  :eq:`sn-n2n-transfer-binding` is the
operator's statement; this one states its :math:`\ell = 0` block.  They
coincide exactly when the binding is isotropic — order 0, an ``NL = 1``
evaluation, or an absent MT=16 section — which is a **property of the
datum**, exposed as
:attr:`~orpheus.transport.operators.transfer.TransferOperator.is_isotropic`
and used to skip the :math:`R\Lambda M` product rather than to select a
different formula.  Citing this equation for the operator as a whole was
correct until 2026-09-04 and is not correct now.

**The transpose falls out of the lift by differentiation** — no new
derivation, and no per-geometry work.  From :eq:`sn-n2n-isotropic-lift`,

.. math::
   :label: sn-n2n-adjoint-source

   \bigl(N_{2n}^{T}\chi\bigr)_{m,g'}
   \;=\; \sum_{n,g}
         \frac{\partial \bigl(N_{2n}\psi\bigr)_{n,g}}{\partial \psi_{m,g'}}\,
         \chi_{n,g}
   \;=\; \frac{w_m}{W}\,\sum_{g} K_{g'\to g}\,\sum_{n}\chi_{n,g}
   \;=\; \frac{w_m}{W}\,\bigl(K^{\mathsf T}\textstyle\sum_n \chi_n\bigr)_{g'} ,

.. (vv-status rationale) Definitional identity: the Euclidean transpose
   of the ℓ = 0 block, obtained by differentiating
   :eq:`sn-n2n-isotropic-lift` (the forward's constant embedding
   transposes to a sum over ordinates; the forward's w-weighted
   integration transposes to the w-weighted embedding).  The OPERATOR's
   transpose is the per-ℓ product reversal below, of which this is the
   ℓ = 0 block.  Its verifiable content is the per-group Euclidean
   reciprocity ⟨N₂ₙψ,χ⟩ = ⟨ψ,N₂ₙᵀχ⟩ and its group-flip mutation leg,
   both ``@pytest.mark.foundation`` in
   ``tests/sn/operators/test_n2n_operator.py::TestTheBindingAtTheSolveOrder``
   (``test_euclidean_reciprocity``, ``test_transpose_reds_on_group_flip``)
   — and since #426 step 2 that reciprocity is measured on an ℓ ≥ 1
   fixture, so it now pins the per-ℓ chain and not only this block.
.. vv-status: sn-n2n-adjoint-source documented

i.e. the lift run backwards: the forward **broadcasts** one scalar
emission onto every ordinate and the transpose **sums** the per-ordinate
cotangent back; the forward **integrates** :math:`\psi` against
:math:`w`, the transpose **embeds** with :math:`w`.  Read against the
frame, this is :math:`\tfrac{1}{W}M_0^{T}K^{T}R_0^{T}` — :math:`R_0^{T}`
the plain ordinate sum, :math:`M_0^{T}` the :math:`w`-weighted embedding.

**And the operator's transpose is the same expression with every
block in it.**  From :eq:`sn-n2n-transfer-binding`, and by exactly the
argument :math:`S` uses (:eq:`sn-scattering-adjoint-kernel-transpose`),

.. math::
   :label: sn-n2n-adjoint-per-ell

   N_{2n}^{\mathsf T} \;=\; \tfrac1W\,M^{\mathsf T}\,
   \Lambda_{2n}^{\mathsf T}\,R^{\mathsf T} ,
   \qquad
   \Lambda_{2n}^{\mathsf T} \;=\; \nu_{2n} \sum_{\ell=0}^{L}
   \mathbf{P}_\ell \otimes \Sigma_{2n,\ell}^{\mathsf T} ,

.. (vv-status rationale) Structural identity: the (n,2n) Euclidean
   transpose as the reversal of the product :eq:`sn-n2n-transfer-binding`
   — a statement about how the transpose is COMPOSED, not a new
   derivation (Λ is the only group-asymmetric factor, so the whole
   reversal is one expression).  Its verifiable content is the per-group
   Euclidean reciprocity on an ℓ ≥ 1 fixture plus the group-flip
   mutation leg, both ``@pytest.mark.foundation`` in
   ``tests/sn/operators/test_n2n_operator.py::TestTheBindingAtTheSolveOrder``.
.. vv-status: sn-n2n-adjoint-per-ell documented

:math:`\Lambda_{2n}` being block-diagonal on the harmonic axis and the
ONLY group-asymmetric factor, exactly as for :math:`S` — which is why
the whole product transpose is one expression and no per-:math:`\ell`
case analysis appears anywhere.  ⛔ The paragraph above ended, until
2026-09-04, with *"the same product-transpose shape*
:eq:`sn-scattering-adjoint-kernel-transpose` *has, one* :math:`\ell`
*-block wide"*.  The shape claim was right and the width was the
model: it is :math:`L+1` blocks wide now.

The reciprocity is a real cross-check rather than a tautology for the
same reason :math:`S`'s is: the forward evaluates the :math:`\ell = 0`
block on the reaction-rate fast path and the transpose rides the frame
form, two different float programs.

.. note::

   **How the transpose is SPELLED changed at CS4c step 4 (2026-08-30);
   what it computes did not.**  Until the step-4 harmonization the
   operator retained a bare ``weights`` array and evaluated
   :eq:`sn-n2n-adjoint-source` as hand-written arithmetic — a
   :math:`w`-broadcast followed by a division by :math:`W`.  It now
   retains the :math:`L = 0` **frame** instead and spells the transpose
   as factor reversal of the cached conjugated product
   ``N2NOperator.full_n2n_kernel = frame.conjugate(N2NMomentOperator)``,
   followed by the producer-side :math:`/W` — exactly the shape
   :math:`S` and :math:`F` use (:ref:`sn-fission-binding-adjoint`).  So
   the equation above became the **product chain's** identity rather
   than this module's transcription, and the last hand-rolled
   :math:`w` arithmetic on any adjoint path went with it.
   ``N2NOperator.weights`` no longer exists;
   :attr:`~orpheus.transport.operators.transfer.TransferOperator.total_weight`
   derives :math:`W` from the retained frame's measure, as :math:`S`
   does.

   ⛔ Both names in that sentence are themselves history now.  #426
   step 2 (2026-09-04) collapsed the two channels onto one binding, so
   the product is
   :attr:`~orpheus.transport.operators.transfer.TransferOperator.full_transfer_kernel`
   ``= frame.conjugate(Λ_c)`` for either channel and
   ``N2NMomentOperator`` — the :math:`\ell = 0`-only moment factor —
   retired into
   :class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`.
   The step-4 *claim* is unaffected and is the reason step 2 was cheap:
   once the transpose is a product reversal, widening the middle factor
   costs nothing on this page.

   **The re-spelling costs nothing, and that is a theorem of**
   :math:`\ell = 0` **rather than a lucky fixture** — measured, and
   worth writing down because the neighbouring :math:`F`
   harmonization is *not* free (:ref:`sn-fission-binding-adjoint`).  At
   one moment the two outer factors degenerate: :math:`R_0^{\mathsf T}`
   is the plain ordinate sum (the reconstruction face's
   :math:`\ell = 0` column is the all-ones broadcast) and
   :math:`M_0^{\mathsf T}` is a per-ordinate multiply by :math:`w_n`
   (the analysis face's :math:`\ell = 0` row **is** the weight vector),
   so the product chain performs the *same* float operations, in the
   *same* order, as the retired broadcast did — there is no summation
   over :math:`\ell` to re-associate.  ``[M]`` 2026-08-31, 1000 draws
   (200 seeds :math:`\times` ``gauss_legendre`` :math:`n = 2, 4, 6, 8,
   16`) on the shipped 2-group slab fixture: ``np.array_equal``
   **1000 / 1000**, :math:`\max|\Delta| = 0`.  A bit-exact reading on
   one draw would have been a property of the draw
   (``vv-principles`` #31); the sweep, and the structural reason above,
   make it a property of the binding.

   Note what this does to the cross-check's *strength*: the two float
   programs are now the reaction-rate fast path and the frame-form
   product, which are structurally further apart than the fast path and
   a hand-written :math:`w`-embedding were.  The retirement therefore
   **promoted** the reciprocity gate rather than demoting it
   (``coding-standards``, the silent-promotion mirror) — its blindness
   analysis in the warning below is unchanged, because the
   :math:`w`-embedding it warns about is still what the product
   *computes*.

.. warning::

   The transpose's :math:`w_m` factor is the trap this channel offers,
   and it has a fixture that cannot see it.  The forward's per-ordinate
   output is a **constant in** :math:`n` (:math:`q_n = q^{\rm iso}/W`
   for every ordinate), which invites writing the transpose as the
   *uniform average* :math:`\tfrac{1}{N}\,K^{\mathsf T}\sum_n\chi_n`
   instead of the :math:`w`-embedding :math:`\tfrac{w_m}{W}\,
   K^{\mathsf T}\sum_n\chi_n`.  On an **equal-weight rule** the two are
   algebraically identical — :math:`w_m \equiv c` gives :math:`W = Nc`
   and hence :math:`w_m/W = 1/N` — so an equal-weight fixture is
   *structurally* blind to the substitution (``vv-principles`` Mode 12
   at the fixture), while every non-uniform rule separates them.  The
   shipped gate accordingly builds on ``Quadrature.gauss_legendre(4)``,
   whose weights are :math:`(0.3479, 0.6521, 0.6521, 0.3479)`, and
   carries a group-flip mutation leg
   (``test_transpose_reds_on_group_flip``) so that the
   :math:`\Sigma_{2n}` transpose convention — ERR-002's habitat — is
   pinned as well.

**What survives, and what it is for.**  The moment-space channel object
survived the CS4c extraction and was retired by #426 step 2 — into its
sibling, not into nothing, and the distinction is what a future reader
greping ``N2NMomentOperator`` needs.  It was the **algebra of record**
for the fast path: its ``apply`` was :math:`\nu_{2n}\Sigma_{2n}` on the
:math:`\ell = 0` moment and its ``apply_transpose`` the group transpose
of that, so ``frame.conjugate(N2NMomentOperator)`` reconstructed
:eq:`sn-n2n-isotropic-lift` as a literal
:class:`~orpheus.numerics.operator.OperatorProduct` — precisely the
reference the lift was gated against.  Retiring it as dead weight would
have retired the oracle (the ``coding-standards`` *fuller-view oracle*
exception).

That is not what happened.  The class was a one-block special case of
:class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`,
which now carries both channels — so the oracle is **stronger**, not
gone: ``frame.conjugate(LegendreMomentTransfer(n2n_field, skip_l0=False))``
reconstructs :eq:`sn-n2n-transfer-binding` at the solve's order, and the
gate that consumes it
(``test_apply_equals_the_frame_conjugation_at_the_solve_order``) now
exercises :math:`L+1` blocks where its predecessor exercised one.  A
retirement that *widens* the reference it removes is the case
``coding-standards`` calls the silent **promotion**, and it is recorded
here because nothing in a green suite would say so.

.. _sn-fission-binding-adjoint:

The fission adjoint — :math:`F` is the third member of one shape
-----------------------------------------------------------------

CS4c step 4 (design record §16.2, landed 2026-08-30) finished what the
:math:`(n,2n)` extraction started.  Before it, each of the three gain
channels reached its transpose a different way: :math:`S` by product
reversal of a frame conjugation, :math:`N_{2n}` by a hand-written
:math:`w`-broadcast, :math:`F` by a hand-written
``np.multiply.outer(w, ·)/W``.  After it, **all three are the same
sentence**, and the sentence is worth stating once because it is what
makes the adjoint chain free of channel-specific arithmetic.

Write :math:`R_\ell` / :math:`M_\ell` for the frame's reconstruction /
analysis faces truncated at order :math:`\ell`, and :math:`W = \sum_n
w_n`.  The three gains are

.. math::
   :label: sn-gain-channels-one-shape

   S \;=\; \tfrac{1}{W}\,R\,\Lambda_{\ell \le L}\,M ,
   \qquad
   N_{2n} \;=\; \tfrac{1}{W}\,R_0\,
                \bigl(\nu_{2n}\Sigma_{2n}^{\mathsf T}\bigr)\,M_0 ,
   \qquad
   F \;=\; \tfrac{1}{W}\,R_0\,
           \bigl(|\chi\rangle\langle\nu\Sigma_f|\bigr)\,M_0 ,

.. (vv-status rationale) Structural identity: the three gain channels
   written on the ONE frame-conjugation shape the CS4c step-4 rebind
   installed — a composition-site fact about how the bindings are built,
   not a solver claim.  Its verifiable content is that each realized
   operator reproduces its own conjugated product: the
   ``@pytest.mark.foundation`` gates
   ``tests/sn/operators/test_n2n_operator.py::TestLiftIsTheConjugation::test_apply_equals_l0_conjugation``
   (N₂ₙ), ``tests/sn/operators/test_scattering_adjoint.py::TestFullScatterKernel``
   (S), and for F the two-binding split gated by
   ``tests/sn/operators/test_isotropic_fission.py`` (the energy binding's
   forward/transpose against hand-rolled dyads, with a swapped-factor
   control) plus ``tests/sn/operators/test_fission_adjoint.py::TestCompositeTransposeArm``
   (the angular binding's composite arm against an independent spelling,
   with a weight-swap discriminator).
.. vv-status: sn-gain-channels-one-shape documented

differing **only** in the middle factor and in how far the faces are
truncated: a dense per-:math:`\ell` Legendre stack over
:math:`(L{+}1)^2` moments for :math:`S`, a dense one-group-to-group
matrix on one moment for :math:`N_{2n}`, a **rank-1 dyad** on one moment
for :math:`F`.  Fission is thus the :math:`\ell = 0`, rank-1 degenerate
of the scattering binding — the reading
:ref:`operator_algebra <emission-kernels-btd>` had carried as a lens,
now a shared-code fact.  Critically the frame is *hub-interned*
(:meth:`HarmonicFrame.for_space
<orpheus.transport.frames.harmonic_frame.HarmonicFrame.for_space>`), so
an :math:`S`, an :math:`N_{2n}` and an :math:`F` posed on one space
reach the **same frame object** and therefore agree on the angular
metric by construction rather than by convention — which is the
precondition every reciprocity statement on this page quietly needs.

The consequence for the adjoint is immediate.  Each transpose is the
reversal of its own product,

.. math::
   :label: sn-gain-transposes-one-shape

   X^{\mathsf T}\psi^{*}
   \;=\;
   \tfrac{1}{W}\,M_\ell^{\mathsf T}\,K^{\mathsf T}\,
   R_\ell^{\mathsf T}\,\psi^{*} ,
   \qquad X \in \{S,\;N_{2n},\;F\},

.. (vv-status rationale) Structural identity: the Euclidean transpose of
   each gain as the OperatorProduct reversal of its own conjugation —
   the shared shape, not a per-channel derivation.  Its verifiable
   content is the per-group Euclidean reciprocity ⟨Xψ,χ⟩ = ⟨ψ,Xᵀχ⟩ on
   each channel, the ``@pytest.mark.foundation`` rows named in
   :eq:`sn-scattering-adjoint-source` (S), ``TestLiftIsTheConjugation``
   (N₂ₙ) and
   ``tests/sn/operators/test_fission_adjoint.py::TestForwardAdjointReciprocity``
   + ``::TestCompositeTransposeArm`` (F, scalar and composite arms).
.. vv-status: sn-gain-transposes-one-shape documented

assembled by :meth:`OperatorProduct.apply_transpose
<orpheus.numerics.operator.OperatorProduct.apply_transpose>` from the
leaf transposes.  For :math:`F` the middle transpose
:math:`K^{\mathsf T} = |\nu\Sigma_f\rangle\langle\chi|` is the
:math:`\chi \leftrightarrow \nu\Sigma_f` **role swap** — a theorem of
the rank-1 ``outer`` primitive (:meth:`RankOneOperator.apply_transpose
<orpheus.numerics.operator.RankOneOperator.apply_transpose>`), never
re-derived — so *no line of fission code computes an adjoint*.  The
metric Hilbert adjoint ``F.H`` then composes
:math:`\sharp_V \circ F^{\mathsf T} \circ \flat_W` out of the bound
spaces' own Riesz legs; nothing fission-specific appears on that path
either.

.. important::

   **The dyad and the lift live on different objects, and a reader of
   the daggered-posing bullets needs the distinction.**  Since step 4
   the fission channel is *two bindings of one datum*
   (:ref:`fission-as-dyad`): the **energy** binding
   :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`,
   whose ``apply_transpose`` is exactly the bare dual dyad
   :math:`F^{\mathsf T}\psi^{*} = \nu\Sigma_f\,(\chi\cdot\psi^{*})` on
   the scalar flux, and the **angular** binding
   :class:`~orpheus.transport.operators.fission.FissionOperator`, whose
   transpose is :eq:`sn-gain-transposes-one-shape` — the dual dyad
   *wrapped* in the ordinate sum :math:`R_0^{\mathsf T}` and the
   :math:`w`-weighted embedding :math:`M_0^{\mathsf T}`.  Writing the
   bare dyad as "the fission transpose" without saying which binding is
   meant drops those two factors.  The daggered eigen-pencil below poses
   on the composite, so it is the angular one.

   ⭐ **Since CS4c step 5 (2026-09-04) the angular binding no longer
   *holds* the energy one — it DERIVES it.**
   :class:`~orpheus.transport.operators.fission.FissionOperator`'s exact
   constructor retains the representation-free
   :class:`~orpheus.transport.material_field.FissionMaterialField`, the
   two minted :math:`L = 0` faces and its two ends, and nothing else
   (#450); ``F.isotropic_energy`` is
   :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`
   built from *that* datum on the emitted interior's scalar sub-space,
   cached once at construction.  So "which binding" is no longer a
   question about two objects a caller might have wired inconsistently —
   there is one datum, and the second binding is a theorem of the first.
   `[M]` ``F.isotropic_energy.fission is F.fission`` and its domain is
   the ``(ng, *spatial)`` sub-space of the CODOMAIN's interior (read off
   the codomain because the moment-end sibling's domain interior is a
   tensor-product space with no axes to name it — F-1 of the step-5
   verification plan).

.. admonition:: The transposes' surface, per binding
   :class: note

   The angular lifts (:math:`S`, :math:`N_{2n}`, :math:`F`) expose
   exactly ``apply_transpose(FullField) -> FullField`` — there is **no**
   bare-``ndarray`` transpose arm on them, and no per-carrier family of
   transposes.  The cotangent lands on the **domain's** interior in that
   end's own source/sink class: an
   :class:`~orpheus.transport.source_sinks.AngularSourceSink` for an
   angular-end binding, a
   :class:`~orpheus.transport.source_sinks.HarmonicMomentSourceSink` for
   a moment-end one; the trace is the implicit zero (a volumetric gain's
   transpose is volumetric), emitted in the SOURCE role through the one
   lift verb.  `[M]` on a GL8 slab: ``F.apply_transpose(FullField)``
   returns interior ``AngularSourceSink`` + boundary
   ``AngularBoundarySourceSink``, the latter all-zero.

   The arithmetic is unchanged and stays factor reversal — the
   :class:`~orpheus.numerics.operator.OperatorProduct` chain reverses the
   cached conjugated product, then the producer :math:`1/W`.  What moved
   is only *who names the output carrier*: the operand's own class
   declares its source/sink partner
   (:ref:`role-partner-declaration`), so the transpose spells a ROLE and
   never a leaf.  The **bare-array** transposes live one tier down, on
   the ENERGY bindings, which are plain-bound by ruling R-4 —
   :meth:`IsotropicFission.apply_transpose
   <orpheus.transport.operators.isotropic_transfer.IsotropicFission.apply_transpose>`
   is the bare dual dyad on ``(ng, *spatial)`` arrays, and that is the
   object every scalar consumer (the k-outer, the ray seed, diffusion)
   reaches.

.. note::

   **Why the** :math:`F` **re-spelling moved values and the**
   :math:`N_{2n}` **one did not.**  Both harmonizations replaced hand
   arithmetic with the same product reversal, and it is tempting to
   describe them together — but they are different in kind, and the
   difference is the position of the :math:`1/W`.  The retired
   :math:`N_{2n}` spelling was ``(w * Kᵀ(Σχ)) / W``, which is
   *operation-for-operation* what the reversed chain does at
   :math:`\ell = 0`; measured, it is bit-identical (previous
   subsection).  The retired :math:`F` spelling divided **first** —
   ``outer(w, Kᵀ(Σχ / W))`` — so the chain now applies
   :math:`K^{\mathsf T}` before the division instead of after, a
   genuine IEEE-754 re-association.

   ``[M]`` 2026-08-31, 600 draws (200 seeds :math:`\times` three angular
   rules) on the shipped fissile 4-group heterogeneous 2-D fixture
   (asymmetric :math:`\chi = [0.6, 0.35, 0.05, 0]`, asymmetric
   :math:`\nu\Sigma_f`, fuel/moderator split):

   .. list-table::
      :header-rows: 1
      :widths: 30 14 20 20 16

      * - angular rule
        - :math:`N`
        - ``array_equal``
        - :math:`\max|\Delta|`
        - max ULP
      * - ``lebedev(17)``
        - 110
        - 0 / 200
        - :math:`8.33\times10^{-17}`
        - 5
      * - ``lebedev(11)``
        - 50
        - 0 / 200
        - :math:`8.33\times10^{-17}`
        - 4
      * - ``level_symmetric(4)``
        - 24
        - 0 / 200
        - :math:`8.33\times10^{-17}`
        - 5

   So the change is **principled-equivalent, not bit-identical**, and it
   meets all three ``vv-principles`` criteria: the new intermediate is a
   named object (the conjugated product, not an anonymous broadcast);
   the value is anchored by the reciprocity gate against the
   independently-evaluated forward fast path; and the drift is a
   single-reduction re-association at :math:`\le 5` ULP.  ⚠ Do **not**
   pin the :math:`F` row at ``array_equal`` on the strength of the
   :math:`N_{2n}` result — the two channels are exactly the case
   ``vv-principles`` #31 warns about, and here the sweep separates them.

.. _sn-adjoint-daggered-posing:

The daggered posing
===================

The adjoint entries are :func:`~orpheus.sn.solver.solve_sn_adjoint`
(eigenvalue) and
:func:`~orpheus.sn.solver.solve_sn_adjoint_fixed_source`
(importance / detector), module-level siblings of the forward family.
Neither spells adjoint physics: both hand
:func:`~orpheus.numerics.iteration.KEigenvalue` /
:class:`~orpheus.numerics.iteration.SourceIteration` a **daggered**
operator triple and let the operator algebra do the rest.

The daggered eigenproblem
-------------------------

The shared build ``_adjoint_posing_parts`` takes the forward
within-group system off the single construction site
:func:`~orpheus.sn.coupled_system.build_within_group_system` and returns
its daggerable parts — the invertible resolvent :math:`(L+C)`, the
summed coupling gain :math:`(S + N_{2n} + B)`, and the fission operator
:math:`F`.  It does not enumerate those gain members: it folds the
builder's own ``explicit_gains`` tuple with ``+``, so a member added to
the within-group algebra (as :math:`N_{2n}` was, CS4c §14.1) reaches the
adjoint posing without an edit here — the summation is over whatever the
one construction site composed.  The CALLER daggers each with ``.H`` and
poses

.. math::
   :label: sn-adjoint-eigenproblem

   A_{\rm loss}^{\dagger}\,\psi^* \;=\; \frac1k\,F^{\dagger}\,\psi^*
   \qquad\Longleftrightarrow\qquad
   \mathtt{KEigenvalue}\bigl((L{+}C).\mathtt{H},\;
   (S{+}N_{2n}{+}B).\mathtt{H},\;
   F.\mathtt{H}\bigr),

with :math:`A_{\rm loss}^{\dagger} = (L+C).\mathtt{H} -
(S+N_{2n}+B).\mathtt{H}`
fed to the **unchanged** :func:`~orpheus.numerics.eigenvalue.power_iteration`
(the adjoint row of the eigenvalue-posing table,
:mod:`orpheus.numerics.eigenvalue`).  The within-group loss splits and
each term daggers independently:

* :math:`(L+C).\mathtt{H}` is invertible **for free** by the swap law
  (:eq:`loss-rep-adjoint-inverse-swap`): ``(L+C).H.inverse()`` routes to
  ``(L+C).inverse().H``, i.e. the reverse-scan transpose sweep
  (:ref:`loss-rep-orientation-two-frames`), so the daggered inner solve
  is the reversed walk with no new solver code.
* :math:`S.\mathtt{H}` is the frame-conjugated scattering transpose
  (:ref:`sn-scattering-adjoint`) wrapped in the angular metric.
* :math:`N_{2n}.\mathtt{H}` is the metric-wrapped
  :math:`w`-embedding of :eq:`sn-n2n-adjoint-source` — the lift run
  backwards (:ref:`sn-n2n-adjoint`).
* :math:`B.\mathtt{H}` is the boundary-law transpose; the reflective
  and vacuum traces transpose structurally — an adjoint vacuum is the
  transpose of the forward vacuum, never a user-facing BC flip.
* :math:`F.\mathtt{H}` is the metric-wrapped fission transpose.  Its
  Euclidean core is :eq:`sn-gain-transposes-one-shape` at
  :math:`\ell = 0` — the ordinate sum :math:`R_0^{\mathsf T}`, then the
  :math:`\chi\leftrightarrow\nu\Sigma_f` **role swap**
  :math:`K^{\mathsf T}\phi^* = \nu\Sigma_f\,(\chi\cdot\phi^*)`, then the
  :math:`w`-weighted embedding :math:`M_0^{\mathsf T}` and the
  producer-side :math:`1/W`
  (:class:`~orpheus.transport.operators.fission.FissionOperator`; the
  bare role swap alone, without the two angular factors, is the
  **energy** binding's transpose —
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`,
  see :ref:`sn-fission-binding-adjoint`).

Because :math:`k^{\dagger} = k` is exact, the entry returns
``keff`` equal to the forward eigenvalue to iteration tolerance, and
its ``angular_flux`` is the true discrete adjoint (importance) flux
:math:`\psi^*` — verified against the closed-form
:math:`(A^{\mathsf T})^{-1}F^{\mathsf T}` spectrum
(:ref:`sn-adjoint-verification`).

.. _sn-adjoint-coupled-posing:

The coupled (sphere) posing
---------------------------

On a carrying mesh — the sphere, whose half-angle starting-direction
seed is first-class System-B state — the posing is a 2×2 block operator
over System A (the transport bulk ⊕ trace) and System B (the
radial-characteristic ray).  The gain is the builder's own coupled gain
grid :math:`N`; the fission operator is lifted to the coupled grid:

.. math::

   F_{\rm posed} \;=\;
   \begin{pmatrix}
     F & 0 \\[2pt]
     A_{BA}^{\rm fis} & 0_{BB}
   \end{pmatrix},

where the :math:`(B,A)` block :math:`A_{BA}^{\rm fis}` is the **fission
ray fold** — the kernel-generic
:class:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicEmission`
carrying the fission channel's **energy binding**
``F.isotropic_energy`` (the operator spelling of the coupled fission
seed's :math:`q_{1/2}` assembly).

.. note::

   **One slot, one level — and it closed two adjoint bypasses (CS4c step
   5, 2026-09-04).**  The emission's ``emission_kernel`` slot is a
   dependency injection over *any* ``ndarray → ndarray`` emitter, and
   until this step the daggered posing filled it with ``F.kernel`` — the
   rank-1 :class:`~orpheus.numerics.operator.TensorProductOperator`
   *inside* the energy binding — while the within-group site filled the
   same slot with the **operators** ``S.isotropic_energy +
   N2N.isotropic_energy``.  Two levels of one abstraction in one slot.

   The consequence was invisible in every value: `[M]` the census
   recorded
   :meth:`IsotropicFission.apply_transpose
   <orpheus.transport.operators.isotropic_transfer.IsotropicFission.apply_transpose>`
   at **zero** calls, bypassed on **two** routes — the posing reaching
   past it into ``F.kernel``, and the moment factor
   ``FissionMomentOperator`` reading ``energy.kernel`` rather than
   calling the operator.  Both routes computed the *same numbers*, which
   is exactly why neither could be caught by an adjoint value record: a
   route claim is a claim about the PATH, and only an instrument on the
   path can see it (``vv-principles`` #26).  Both now call the operator's
   own verb, one level of delegation, and the energy binding's transpose
   is REACHED on the production adjoint path rather than stepped over.

   ⚠ Fission still does **not** flow through :math:`A_{BA}` in the
   *within-group* gain (HAZARD 5 in
   :doc:`/theory/foundations/coupled_block_operator` — routing it there
   would double-apply :math:`K \circ \int d\mu`).  This note is about
   the eigen-:math:`M` posing's own :math:`(B, A)` block, where the fold
   belongs.

On the eigen-:math:`M`
operator that fold **belongs** in the posing; the within-group gain
keeps it out (HAZARD 5), not the eigenproblem.  The :math:`(B,B)` block
is the genuine **zero map** ray-flux → ray-source: the :math:`w = 0`
closed rays carry no quadrature weight, so they never source fission —
spelled with the space-typed
:class:`~orpheus.numerics.operator.ZeroOperator` (``codomain_zero`` and
its dual ``transpose_zero`` hook) so both the forward grid and its
dagger emit the source-classed ray zero.  The ray-leg's
``solve_transpose`` output is duality-typed to the adjoint FLUX (the
dual of a source under the G-pairing is the adjoint flux), the exact
sibling of the within-group ``StreamingCollisionOperator.solve_transpose``
fix.

The dual lift asymmetry
-----------------------

The fixed-source entry
(:func:`~orpheus.sn.solver.solve_sn_adjoint_fixed_source`) exposes the
sharpest test of "adjoint ≠ forward-with-a-sign-flipped-source": the
source lift.  The forward isotropic-source lift divides by the
quadrature weight sum, :math:`q \mapsto \tfrac1W\,\mathbf 1_\Omega\,q`;
the **adjoint** detector-response lift does **not** —
:math:`\Sigma_d \mapsto \mathbf 1_\Omega\,\Sigma_d`, a plain angle-flat
broadcast with **no** :math:`w_n`, **no** :math:`1/W`.  The two lifts
are duals of *different* maps.  Under the bulk metric :math:`G = V\,w_n`
the plain broadcast is exactly the dual of the scalar-flux extraction,

.. math::

   \langle \mathbf 1_\Omega\,\Sigma_d,\,\psi\rangle_G
   \;=\; \sum_{\rm cells} V\,\Sigma_d\,\varphi
   \;=\; \langle \Sigma_d,\,\varphi\rangle_V ,

the detector-response functional — whereas the forward :math:`1/W` lift
is the dual of *source injection*.  This asymmetry IS the content of the
P1.2 reciprocity gate: the entries duality row cross-checks the
detector side against the hand volume sum :math:`\sum V\,\Sigma_d\,
\varphi`, pinning the angle-flat lift as exactly the adjoint of the
extraction.  (The daggered **coupled** fixed-source arm — a carrying
mesh with System B — is a typed, loud refusal at #276 A4: it has no
consumer or gate yet and lands with its first consumer rather than
shipping unexercised.  The eigenvalue entry covers carrying meshes.)

.. _sn-adjoint-carrier:

The adjoint flux carrier — ``AdjointSolution``
==============================================

Where does :math:`\varphi^*` live?  Not on the forward
:class:`~orpheus.sn.solution.Solution`.  Campaign #276 A5 split the
solution carrier along a **role axis** into sibling leaves under a
role-agnostic base
(:class:`~orpheus.sn.solution.SolutionBase` →
{:class:`~orpheus.sn.solution.Solution`,
:class:`~orpheus.sn.solution.AdjointSolution`}).

Role is a type; problem kind is a property
------------------------------------------

The solution family discriminates along **two independent axes that use
deliberately different mechanisms**:

* **Problem kind** (fixed-source vs eigenvalue) is a **property** — one
  carrier covers both via the optional ``keff``, because the two kinds
  share every realisation *and* every operation (homogenising a
  fixed-source flux is as meaningful as homogenising an eigenmode).  A
  type here would be ceremony.
* **Solution role** (forward vs adjoint) is a **type**.  The roles
  share the carrier — same fields, same packaging convention (both
  route through the one scalar- and role-agnostic ``_package_solution``
  tail) — but **not the operation set**.

The base is deliberately non-instantiable (a role-less solution is not
a value that exists); a capability-*removing* subclass
(``AdjointSolution`` inheriting ``Solution`` and hiding
``homogenize``) would violate Liskov substitutability, so the two roles
are **siblings** under the base, not parent/child.

The forward physics is structurally absent
------------------------------------------

The forward-physics operations —
:meth:`~orpheus.sn.solution.Solution.homogenize`,
:meth:`~orpheus.sn.solution.Solution.condense`,
:meth:`~orpheus.sn.solution.Solution.reaction_rate_density` — live on
:class:`~orpheus.sn.solution.Solution` **and only there**.  They
interpret ``scalar_flux`` as the flux :math:`\phi` and collapse cross
sections *preserving reaction rates* — an operation whose subject is
the forward flux.  **An importance map has no reaction rate to
preserve**, so those methods do not exist on
:class:`~orpheus.sn.solution.AdjointSolution` at all.  The absence is
**structural**, not a runtime refusal: ``adj.homogenize`` is an
``AttributeError``, and the wrong physics is *unspellable*
(``coding-elegance`` Pattern 4 — illegal states unrepresentable).  The
adjoint's ``scalar_flux`` is the importance :math:`\varphi^* = \sum_n
w_n\,\psi^*_n`, exposed under the domain-named alias
:attr:`~orpheus.sn.solution.AdjointSolution.importance` — one storage,
two vocabularies.

The design tension, ruled forward-looking
-----------------------------------------

The type-minting criterion (``coding-standards``: mint a type iff
:math:`\ge 2` non-isomorphic realisations **and** a non-identity
morphism is applied to it) technically **fails** for :math:`\varphi^*`
in isolation: it is byte-for-byte the same storage as :math:`\phi`, and
the adjoint-weighting :math:`\langle\varphi^*, \Sigma\varphi\rangle` is
a bilinear applied to the *pair* at a call site, not a morphism on
:math:`\varphi^*` alone.  The testability axis therefore favoured
leaving :math:`\varphi^*` an unmarked ``Solution``.  The USER **ruled
otherwise (#276 A5, Option 3 — mint the type)**, on the *trajectory*:
the forthcoming adjoint-machinery family — perturbation theory
:math:`\langle\varphi^*, \delta A\,\varphi\rangle`, generalised
perturbation / response estimation, and #281 adjoint-weighted
homogenisation — earns :math:`\varphi^*` a signature-level carrier that
makes its role legible at every boundary.  This is a *correctness /
forward-design* judgement that overrode the local testability
recommendation, and it is recorded here so a future reader does not
"simplify" the type back into a property and lose the intent.

The #281 adjoint-weighted homogenisation API
--------------------------------------------

The homogenize / condense asymmetry is resolved cleanly by the ratified
#281 (P6-B2) API: the forward machinery stays forward-only, and the
adjoint enters as an **optional test weight**,

.. code-block:: python

   # LANDED (P6 #281) — with its Petrov-Galerkin implementation and gates
   Solution.homogenize(coarse, *, adjoint: AdjointSolution | None = None)

— ``None`` keeps today's flux-weighted (Galerkin-degenerate) collapse
bit-identically; a real :math:`\varphi^*` makes the collapse
eigenvalue-consistent per the worth-zeroing taxonomy of the algebra of
record (:mod:`orpheus.derivations.common.homogenization` — the test
weight is the bilinear PRODUCT :math:`\varphi^*\!\odot\varphi` for the
vector channels, the exact angular pairing for :math:`\Sigma_t`, the
per-pair sink×source rule for the matrices, and the mixed-fold factored
fission rule; the same parameter on :meth:`~orpheus.sn.solution.Solution.condense`
runs the B&G-convention bilinear condensation).  The adjoint is the
*test weight* of the forward collapse, never its subject — which is
exactly why the forward trio is structurally absent on the adjoint
leaf.  The full taxonomy narrative and gates live in the frame chapter
(:ref:`sn-homogenization-why-petrov-galerkin`) and the SN verification
slice.

.. _sn-adjoint-verification:

Verification — how :math:`\psi^*` is certified
==============================================

The adjoint flux is **not** verified by MMS.  MMS is a source-driven
pillar that reaches flux-shape and convergence-order but **cannot**
verify an eigenvalue (``vv-principles`` — the pillars); the daggered
:math:`k` and :math:`\varphi^*` need **closed-form** references.  Every
Phase-1 value gate is L1, anchored to a reference that terminates in
``np.linalg.eig`` or the reciprocity identity — never in another
ORPHEUS solver.  φ\* is "correct" only when the whole chain below is
green and every named mutation reddens its named gate under
``python -O``.

The gate map
------------

.. list-table:: The adjoint certification battery (measured on Mixture A)
   :header-rows: 1
   :widths: 14 20 30 36

   * - Gate
     - Claim layer / pillar
     - Reference (structurally independent)
     - What it pins (measured)
   * - **P1.2** duality
     - model (fixed-source) / closed-form
     - the reciprocity identity :eq:`sn-adjoint-duality`; two
       independent solves
     - :math:`\langle\Sigma_d,\psi\rangle = \langle\psi^*,q\rangle` on a
       2G asymmetric-SigS vacuum slab, source and detector in
       DIFFERENT groups AND regions; detector side hand-checked against
       :math:`\sum V\Sigma_d\varphi` (pins the angle-flat lift)
   * - **P1.3** :math:`k^{\dagger}=k`
     - eigenvalue / closed-form
     - ``kinf_homogeneous`` (triple equality, terminates in
       ``np.linalg.eig``)
     - :math:`k^{\dagger}=k_{\rm fwd}=k_\infty` on ∞ 2G+4G, a
       heterogeneous reflective slab, AND the coupled sphere; teeth:
       :math:`F^{\dagger}\!\to\!F`, :math:`S^{\dagger}\!\to\!S`,
       :math:`L^{\dagger}\!\to\!L` each shift :math:`k`
   * - **P1.4** spectrum
     - eigenvalue + flux-shape / closed-form
     - the dominant right eigenvector of
       :math:`(A^{\mathsf T})^{-1}F^{\mathsf T}`
       (``kinf_and_adjoint_spectrum_homogeneous``)
     - the 4G adjoint energy spectrum :math:`\varphi^*_{\rm cf} =
       [0.470, 0.486, 0.518, 0.524]` (:math:`\ne\varphi`, asserted);
       :math:`F^{\dagger}\!\to\!F` reds the spectrum O(1)
   * - **P1.5** bi-orthogonality
     - flux-shape (intrinsic law) / closed-form
     - the spectral decomposition of :math:`M` and :math:`M^{\dagger}`
       (both from ``np.linalg.eig``)
     - the cross-Gram :math:`\langle\psi^*_i, F\varphi_j\rangle` is
       diagonal; for the rank-1 :math:`F=\chi\otimes\nu\Sigma_f` this is
       the degenerate one-nonzero-entry form (both zero mechanisms
       :math:`F\varphi_j=0` and :math:`\chi\cdot\psi^*_i=0` asserted)
   * - **sphere** :math:`\varphi^*`-shape
     - flux-shape / dense forward-probe
     - a dense FORWARD-probed :math:`(A_{\rm loss}, F)` + a raw-data
       coupled :math:`G` (both structurally independent of the ``.H``
       reverse-scan under test)
     - the coupled defining-law residual
       :math:`\|A_{\rm loss}^{\mathsf T}(G\psi^*) -
       F^{\mathsf T}(G\psi^*)/k\|` at rel floor :math:`1.2\times10^{-10}`
       vs gate :math:`10^{-7}` (:math:`n=140`); anti-vacuity
       :math:`|\Delta k| = 3.3\times10^{-11}`

The k rows verify the daggered **eigenproblem**
:eq:`sn-adjoint-eigenproblem`; the P1.2 row verifies the reciprocity
**duality** :eq:`sn-adjoint-duality`.  The full narrative and mutation
teeth live in the V&V slice (:ref:`sn-adjoint-verification-slice`); the
gate code is
``tests/sn/solve/test_sn_adjoint_certification.py`` and
``tests/sn/solve/test_sn_adjoint_entries.py``.

The Mode-12 accounting — what :math:`k` can and cannot see
----------------------------------------------------------

Because :math:`\operatorname{eig}(M^{\dagger}) =
\operatorname{eig}(M)` by construction — the identity lives on the
ITERATION operator :math:`M = A_{\rm loss}^{-1}F` (every factor
daggered; the derivation above) — a :math:`k^{\dagger}=k` gate is
**designed-green** (``vv-principles`` Mode 12) on entire classes of
error.  Getting the boundary exactly right is load-bearing — this
campaign twice caught a wrong "why" here:

* :math:`k` **is EXACTLY blind** to (i) the **factor-order / similarity
  family** (:math:`\operatorname{eig}(M^{\mathsf T}) =
  \operatorname{eig}(M)` — transposing *all* factors is a similarity),
  (ii) **all vector content**, and (iii) **the G-metric itself**
  (:math:`G'^{-1}A^{\mathsf T}G'` is metric-similar to
  :math:`A^{\mathsf T}` for any invertible :math:`G'`).  No tolerance,
  mesh refinement, or regime change can expose these through
  :math:`k`; the committed catchers are the spectrum row, the
  bi-orthogonality row, the duality pairing, and the sphere vector row
  — functionals **outside** the eigenvalue stabiliser.
* :math:`k` **is NOT blind** to a single **leaf-transpose drop**
  (:math:`F^{\dagger}\!\to\!F`, :math:`S^{\dagger}\!\to\!S`,
  :math:`L^{\dagger}\!\to\!L`): transposing *one* factor is **not** a
  similarity of the pencil, and :math:`k` measurably moves —
  :math:`F^{\dagger}=F` shifts :math:`k` from :math:`1.488` to
  :math:`0.171` on the 4G ∞ fixture (the FULL SN-solve measurement; the angular-collapsed 0-D closed-form proxy of the same mutation gives 0.153 — cite the solve, not the proxy).  So the k-equality rows *are*
  legitimate teeth for drops (with the visibility preconditions:
  asymmetric SigS, :math:`\chi\not\parallel\nu\Sigma_f`, spatial
  structure), while the factor-order and metric classes stay the vector
  rows' exclusive province.

**The factor-order trap.**  The P1.4 reference must be the dominant
right eigenvector of :math:`(A^{\mathsf T})^{-1}F^{\mathsf T}`, **not**
:math:`\operatorname{eig}(M^{\mathsf T})`.  The two are similar
(conjugation by :math:`A^{\mathsf T}`) so every :math:`k` check passes
on both — but for the rank-1 :math:`F`, the dominant eigenvector of
:math:`M^{\mathsf T} = F^{\mathsf T}A^{-\mathsf T}` degenerates to
**exactly** :math:`\widehat{\nu\Sigma_f}` (:math:`F^{\mathsf T}x \propto
\nu\Sigma_f` for all :math:`x`), a reference carrying **zero A-physics**.
The wrong reference was caught by the SN daggered solve on first contact
(the corrected law is recorded in
:func:`~orpheus.derivations.common.eigenvalue.kinf_and_adjoint_spectrum_homogeneous`).
And **the metric is caught by nothing but the sphere vector row**: the
:math:`G_{\rm sd} = V_{\rm cell} \to 1` drop leaves
:math:`|k^{\dagger}_{\rm mut}-k_{\rm fwd}| = 2.6\times10^{-11}`
(EXACTLY k-blind) while the residual reds to :math:`2.35` (the ERR-067
family — a metric bug in production that no eigenvalue gate could see).

.. _sn-adjoint-consumers:

Consumers and horizon
=====================

The adjoint flux is a means, not an end.  With :math:`\varphi^*` landed,
its consumers are unblocked:

* **Adjoint-weighted (eigenvalue-consistent) homogenisation** (#281 P6,
  frame-machinery :ref:`galerkin-projection` / #51).  Flux-weighted
  homogenisation is the Galerkin degenerate :math:`\varphi^*=\varphi`;
  the adjoint weight makes it genuinely Petrov-Galerkin, so the coarse
  :math:`k` becomes **first-order stationary** — the homogenisation
  error is :math:`\mathcal O(\delta\varphi^2)` rather than
  :math:`\mathcal O(\delta\varphi)`.  This lands with its C1/C2/C3 gates
  in P6 via the ``Solution.homogenize(..., adjoint=...)`` parameter
  (:ref:`sn-adjoint-carrier`).
* **Perturbation theory and GPT.**  The first-order worth
  :math:`\delta k \propto \langle\varphi^*, \delta A\,\varphi\rangle /
  \langle\varphi^*, F\varphi\rangle` and generalised perturbation /
  response-sensitivity estimation are the adjoint eigenmode's native
  applications — the trajectory the A5 type ruling was made for.

The **honest deferral ledger.**  Two arms are callable-but-deferred by
design, each a typed refusal rather than an unexercised path:

* the daggered **coupled fixed-source** arm (a carrying mesh with
  System B) refuses loud in
  :func:`~orpheus.sn.solver.solve_sn_adjoint_fixed_source` — no consumer
  or gate yet; the eigenvalue entry covers carrying meshes;
* the Gauss–Seidel **schedule-reverse** transpose (#310 R7) has no
  consumer, so a
  :class:`~orpheus.sn.operators.scheduled_invertible.ScheduledInvertibleOperator`
  over it stays non-adjointable (:ref:`loss-rep-orientation-two-frames`,
  the deferral ledger).

Development history
===================

* **#276 P3** (commit ``15185e5``, closes #118) — the **scattering
  adjoint** :math:`S^{\mathsf T}`, free from the harmonic frame.  The
  modernised in-scatter source became one frame-conjugated operator, so
  its transpose falls out as the product transpose with no per-geometry
  derivation.  :class:`~orpheus.transport.operators.scattering.ScatteringOperator`
  gained ``apply_transpose`` and now reports ``is_adjointable = True``.
* **#280 / #310** — the **walk adjoint** / orientation axis.  #280 added
  the reverse-scan inner solve :math:`(L+C)^{-\mathsf T}b` and the swap
  law :math:`(A^{\dagger})^{-1} = (A^{-1})^{\dagger}`; #310 completed the
  ``loss_action_transpose`` grid over every registered
  scheme × representation (DD 1-D/2-D/3-D, LD 1-D/2-D), retiring the
  transpose residue (:ref:`loss-rep-orientation-two-frames`).
* **#276 A4** (merged @ ``065a0e5d``) — the **daggered posing
  activation**.  ``KEigenvalue((L+C).H, (S+B).H, F.H)`` runs through the
  unchanged ``power_iteration``; the entries
  :func:`~orpheus.sn.solver.solve_sn_adjoint` /
  :func:`~orpheus.sn.solver.solve_sn_adjoint_fixed_source` land; the
  coupled sphere posing (fission ray fold + space-typed
  ``ZeroOperator`` + duality-typed ``solve_transpose``) lands; the P1.4
  reference is corrected from the factor-order-degenerate
  :math:`\operatorname{eig}(M^{\mathsf T})` to
  :math:`(A^{\mathsf T})^{-1}F^{\mathsf T}`.
* **#276 A5** (merged @ ``a24380ca``) — the **role-typed carrier**.
  :class:`~orpheus.sn.solution.SolutionBase` →
  {:class:`~orpheus.sn.solution.Solution`,
  :class:`~orpheus.sn.solution.AdjointSolution`}, the forward physics
  structurally absent on the adjoint, ``importance`` alias, and the
  coupled-sphere :math:`\varphi^*`-shape row closing the sphere
  :math:`k` row's honest-scope gap.
* **#276 A6** — this chapter: the route decision, the three-transposes
  taxonomy, the daggered-posing mechanics, the carrier ruling, and the
  verification narrative, closing #276.
* **CS4c step 3** (2026-08-30, branch ``refactor/cs4c-s-rebind``) — the
  :math:`(n,2n)` **channel becomes a first-class operator with its own
  transpose** (:ref:`sn-n2n-adjoint`).  Two consequences reach this
  chapter: :math:`\mathrm{full\_scatter\_kernel}` loses its
  :math:`N_{2n}` summand and is scattering-only
  (:eq:`sn-scattering-adjoint-kernel`), and the daggered gain the
  posing folds is :math:`(S + N_{2n} + B)` — read off the builder's
  ``explicit_gains``, so the A4 bullet's ``(S+B)`` above is the
  spelling of its own day, not a member list this chapter maintains by
  hand.

References
==========

The adjoint transport equation, importance interpretation, and
reciprocity duality follow :cite:`BellGlasstone1970` (§6, importance
and the adjoint) and :cite:`LewisMiller1984` (§6, adjoint transport,
reciprocity, and perturbation theory); the Monte-Carlo importance
reading is :cite:`Lux1991`.  The discrete Hilbert (G-metric) adjoint is
derived in :ref:`operator-adjoint`; the reverse-scan walk transpose and
the swap law in :ref:`loss-rep-orientation-two-frames`.
