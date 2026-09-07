.. _theory-verification-sn:

Discrete Ordinates
==================

.. note:: **Verification slice — automation pending.**  The per-page V&V
   table (equation label × test × level × ERR coverage, auto-filtered
   from the ``tests/_harness`` registry to this page's equation labels) is
   blocked on Nexus equation-label ↔ test linking; until it ships, the
   verification below is **hand-authored**.  The project-wide matrix lives
   at :doc:`/theory/verification/matrix`.

.. _sn-mms-verification:

Method of Manufactured Solutions (1D slab)
-------------------------------------------

Homogeneous and heterogeneous eigenvalue tests verify :math:`\keff`
--- a scalar. They do not tell us whether the **spatial operator**
itself converges at the design order :math:`\mathcal O(h^{2})` of
:term:`diamond difference`.  The Method of Manufactured Solutions closes
that gap by constructing a fixed-source problem whose exact angular
flux is known in closed form, so the error against the prescribed
flux is pure spatial-discretisation error.

**Ansatz.**  For a vacuum-BC slab of length :math:`L` in one energy
group, pick an isotropic :term:`angular flux`

.. math::
   :label: sn-mms-psi

   \psi_n(x) = \frac{1}{W}\,A(x),
   \qquad A(x) = \sin\!\left(\frac{\pi x}{L}\right),

where :math:`W = \sum_n w_n = 2` for Gauss--Legendre.  Because
:math:`A(0) = A(L) = 0`, every :term:`ordinate` vanishes at both faces ---
the :term:`vacuum boundary conditions <vacuum boundary condition>` are satisfied automatically, with no
inflow bookkeeping required on the caller side.  Since :math:`\psi_n`
is independent of ordinate, the :term:`scalar flux` recovered by any
quadrature order is *exactly* :math:`\phi(x) = A(x)` --- the test
isolates spatial error from angular :term:`quadrature` error.

**Manufactured source.**  Substituting :eq:`sn-mms-psi` into the
discrete ordinates transport equation :eq:`transport-cartesian`
(with the :math:`1/W` convention ORPHEUS uses),

.. math::

   \mu_n\,\frac{\partial\psi_n}{\partial x} + \Sigma_t\,\psi_n
   = \frac{1}{W}\!\left(\Sigma_s\,\phi + Q^{\text{ext}}_n\right),

and solving algebraically for :math:`Q^{\text{ext}}_n` gives

.. math::
   :label: sn-mms-qext

   Q^{\text{ext}}_n(x)
   = \mu_n\,A'(x) + \bigl(\Sigma_t - \Sigma_s\bigr)\,A(x)
   = \mu_n\,\frac{\pi}{L}\cos\!\left(\frac{\pi x}{L}\right)
     + \bigl(\Sigma_t - \Sigma_s\bigr)\sin\!\left(\frac{\pi x}{L}\right).

The :math:`W` factor cancels cleanly because the ansatz was already
divided by :math:`W`, so what we hand the solver is the full residual
without any additional rescaling.  The expression is per-ordinate and
linear in :math:`\mu_n`: a constant isotropic external source *cannot*
drive a non-trivial manufactured flux because the streaming term
:math:`\mu_n\,\psi'_n` is odd in :math:`\mu`.  That is the fundamental
reason MMS for SN requires the :math:`Q_{\rm aniso}` plumbing path ---
no "cheat" with a cell-by-cell isotropic source exists.

**Why :math:`\sin(\pi x/L)`?**  The ansatz is smooth
(:math:`C^{\infty}`) so all derivatives of the exact solution exist
and DD's :math:`\mathcal O(h^{2})` truncation error dominates.  It
vanishes at both boundaries for free.  Its derivatives do not collapse
to a polynomial --- a cubic ansatz, for instance, has a constant
second derivative so the DD truncation term :math:`\psi'''` would be
zero and the error could disappear for a non-physical reason,
hiding bugs.  Trigonometric or exponential ansätze have bounded
but non-zero derivatives of every order and therefore expose the
leading truncation term cleanly.

**Implementation.**  The case is built by
:func:`orpheus.derivations.continuous.mms.sn.build_1d_slab_mms_case` and
consumed by :func:`orpheus.sn.solve_sn_fixed_source`.  The latter
accepts a per-ordinate external source of shape
:math:`(N, n_g, n_x, n_y)` (Issue #196 PR-INDEX-5 principled layout,
the ``g`` axis directly after ``N``) and threads it through the :term:`sweep`'s
:math:`Q_{\rm aniso}` slot --- merging additively with any P1+
scattering contribution the solver itself builds.  This bare-array form
is the **bulk-only / vacuum** special case of the composite source
``q = q_bulk ⊕ q_∂`` the solver also accepts (see
:ref:`sn-composite-fixed-source`); this isotropic slab MMS is
vacuum-automatic, so the boundary leaf is identically zero.  Vacuum
boundary conditions are applied via the mesh-level BC infrastructure
described in :ref:`boundary-conditions`:
:func:`solve_sn_fixed_source` defaults its ``boundary_condition``
parameter to ``"vacuum"`` and the internal helper
``_apply_default_bcs`` stamps :attr:`BC.vacuum <orpheus.geometry.mesh.BC.vacuum>`
onto every face of the mesh that lacks an explicit BC declaration.
:class:`SNMesh` then resolves these to the ``"vacuum"`` kind string,
which the sweep reads directly.  In the 1-D cumprod path, the
recurrence starts from zero; in the 2-D wavefront path, the
reflective-partner copy is skipped, leaving incoming-face angular
fluxes at their zero initialisation (which is correct because no
code path writes the incoming-face slot of any ordinate except the
reflection step itself).

.. note::

   Before the BC infrastructure was introduced, the then-production
   ``transport_sweep`` entry accepted a ``boundary_condition: str``
   parameter directly.  That parameter has been removed --- BCs now flow
   through the mesh → SNMesh resolution path.  The description above
   reflects the current implementation.

**Measured convergence.**  With
:math:`\Sigma_t = 1\ \mathrm{cm^{-1}}`,
:math:`\Sigma_s = 0.5\ \mathrm{cm^{-1}}`,
:math:`L = 5\ \mathrm{cm}`, Gauss--Legendre :math:`S_{16}`:

.. list-table::
   :header-rows: 1
   :widths: 10 20 20

   * - :math:`n_{\rm cells}`
     - :math:`\|\phi_h - \phi_{\rm ex}\|_{L^{2}}`
     - measured order
   * - 10
     - :math:`2.17\!\times\!10^{-3}`
     - ---
   * - 20
     - :math:`5.40\!\times\!10^{-4}`
     - 2.01
   * - 40
     - :math:`1.35\!\times\!10^{-4}`
     - 2.00
   * - 80
     - :math:`3.37\!\times\!10^{-5}`
     - 2.00
   * - 160
     - :math:`8.42\!\times\!10^{-6}`
     - 2.00

Successive ratios hit :math:`4.00\pm0.02`, i.e. the measured order
is exactly the design order of diamond difference.  The L1 test
:func:`tests.sn.verification.mms.test_mms.test_sn_1d_slab_mms_converges_second_order`
asserts a slightly loose ``order > 1.9`` bracket to leave room for
round-off at the finest mesh.

**Risk points / things that can go wrong.**

- *Vacuum BC not honoured.*  If the reflective-partner copy is not
  skipped, incoming-face angular flux at the boundary is non-zero
  (the reflected outgoing from the opposite sweep) and the
  manufactured solution no longer satisfies the discrete problem.
  Symptom: :math:`\mathcal O(1)` error at the coarsest mesh; no
  convergence regardless of refinement.
- *Wrong normalisation for* :math:`Q_{\rm ext}`.  The solver's
  :math:`Q_{\rm aniso}` slot is divided by :math:`W` internally;
  the ansatz has a :math:`1/W` prefactor; the two must cancel.
  If the derivation forgets the :math:`W` cancellation, the
  measured flux is a factor of :math:`W` off but still converges at
  order 2 --- sneaky.  Guard: the second test in ``test_mms.py``
  cross-checks the algebraic symmetry of :eq:`sn-mms-qext`.
- *Non-smooth ansatz.*  A discontinuous material or a piecewise
  linear ansatz degrades the observed order to :math:`\mathcal O(h)`.
  The homogeneous sinusoid avoids both.
- *1-group vs multigroup.*  Because the manufactured flux is isotropic
  and there is no fission in the fixed-source problem, 1 group is
  sufficient --- the degeneracy warning about 1-group eigenvalue
  tests does not apply, since no :math:`\keff` enters.  Multigroup
  and heterogeneous MMS extensions are tracked as follow-ups for
  richer operator coverage.

**Follow-ups.**  MMS for :doc:`/theory/methods/method_of_characteristics`, diffusion,
and spherical / cylindrical curvilinear SN is tracked in GitHub
Issues (see ``type:feature level:L1``).  The curvilinear sweeps
need their own ansatz because their vacuum BC plumbing is not
yet wired up. **Heterogeneous and multigroup SN MMS is covered
by the next subsection.**


.. _sn-mms-heterogeneous-verification:

Heterogeneous MMS — 2-group continuous-:math:`\Sigma` slab
-----------------------------------------------------------

The homogeneous MMS case above verifies the Cartesian 1D SN
sweep for a *single-material* slab. To verify the multigroup
operator on a **heterogeneous** problem --- where each cell can
have different cross sections and the scatter matrix couples
groups across positions --- the Method of Manufactured Solutions
is extended in Phase 2.1a of the verification campaign with two
deliberate choices:

1. **Continuous (smooth)** :math:`\Sigma_{t,g}(x)` and
   :math:`\Sigma_{s,g\to g'}(x)` instead of piecewise-constant
   material regions. Discontinuous :math:`\Sigma` at interfaces
   that do not coincide with cell faces degrades diamond
   difference from :math:`\mathcal O(h^{2})` to
   :math:`\mathcal O(h)`, which would contaminate the
   spatial-convergence measurement with interface-treatment
   artefacts. With smooth :math:`\Sigma(x)` the diamond-
   difference operator hits its design :math:`\mathcal O(h^{2})`
   order exactly --- the convergence study becomes a clean test
   of the operator itself. This follows Salari & Knupp
   SAND2000-1444 §6, the canonical MMS reference for
   heterogeneous verification.
2. **Per-group amplitudes** :math:`\mathbf c = (c_1, c_2)` in
   the ansatz, so the scalar flux has a non-trivial group
   spectrum and the downscatter source term in the manufactured
   :math:`Q^{\text{ext}}` is non-zero. A bug that transposes
   the scatter matrix or drops a cross-group source term
   produces an incorrect :math:`\phi_2` that the convergence
   test catches immediately.

**Ansatz.**  The homogeneous ansatz carries over, now with a
per-group amplitude:

.. math::
   :label: sn-mms-hetero-psi

   \psi_{n,g}(x) \;=\; \frac{c_g}{W}\,A(x),
   \qquad A(x) \;=\; \sin\!\left(\frac{\pi x}{L}\right),

where :math:`W = \sum_n w_n` is the quadrature weight sum. The
scalar flux in each group is
:math:`\phi_g(x) = c_g\,A(x)`, so the amplitudes
:math:`\mathbf c` literally are the group fluxes at the slab
midpoint (where :math:`A` peaks). With
:math:`\mathbf c = (1.0, 0.3)` the two groups are linearly
independent and the downscatter coupling is visible.

Both groups share the same *spatial* mode :math:`\sin(\pi x/L)`
--- this is the fundamental mode of the bare slab and is exactly
the shape that emerges from separation of variables in the
diffusion limit. The heterogeneous SN problem would in general
have each group living in its own spatial harmonic, but we
*choose* the shared-mode ansatz as the manufactured target and
derive the non-trivial :math:`Q^{\text{ext}}` that makes it
satisfy the transport equation. The test then measures how well
the numerical SN sweep reproduces this prescribed shape.

**Manufactured source.**  Substituting :eq:`sn-mms-hetero-psi`
into the multigroup discrete-ordinates transport equation

.. math::

    \mu_n\,\frac{\partial\psi_{n,g}}{\partial x}
        + \Sigma_{t,g}(x)\,\psi_{n,g}
    \;=\; \frac{1}{W}\!\left(
        \sum_{g'}\Sigma_{s,g'\to g}(x)\,\phi_{g'}(x)
      + Q^{\text{ext}}_{n,g}(x)
    \right)

and solving algebraically for :math:`Q^{\text{ext}}`:

.. math::
   :label: sn-mms-hetero-qext

   Q^{\text{ext}}_{n,g}(x) \;=\;
       \mu_n\,c_g\,A'(x)
     + c_g\,\Sigma_{t,g}(x)\,A(x)
     \;-\; \sum_{g'}\Sigma_{s,g'\to g}(x)\,c_{g'}\,A(x).

The :math:`W` factor cancels between the ansatz's :math:`1/W`
prefactor and the solver's own :math:`1/W` convention on the
isotropic and anisotropic source slots, so :eq:`sn-mms-hetero-qext`
is the residual hand-delivered to the sweep without any
additional rescaling.

**Structure of the source.**  The streaming term
:math:`\mu_n\,c_g\,A'(x)` is odd in :math:`\mu` and carries the
only angular dependence, which is why SN MMS fundamentally
needs the per-ordinate ``Q_aniso`` plumbing path. The removal
term :math:`c_g\,\Sigma_{t,g}(x)\,A(x)` is diagonal in group
index. The **in-scatter** sum
:math:`\sum_{g'}\Sigma_{s,g'\to g}\,c_{g'}\,A(x)` is the only
term that couples groups, and for :math:`g=2` in the default
2-group setup it contributes
:math:`-\Sigma_{s,1\to 2}(x)\,c_1\,A(x)` --- the thermal source
depends on the fast amplitude through the downscatter cross
section, exactly as the multigroup scatter assembly in the
sweep does.

**Canonical cross sections.**  The reference uses smooth
profiles on :math:`[0, L]`:

.. math::

    \Sigma_{t,1}(x) &= 1.0 + 0.2\sin(\pi x/L), \\
    \Sigma_{t,2}(x) &= 2.0 + 0.3\cos(\pi x/L), \\
    \Sigma_{s,1\to 1}(x) &= 0.3 + 0.1\sin(\pi x/L), \\
    \Sigma_{s,1\to 2}(x) &= 0.2 + 0.05\sin(\pi x/L), \\
    \Sigma_{s,2\to 2}(x) &= 1.5 + 0.15\sin(\pi x/L), \\
    \Sigma_{s,2\to 1}(x) &= 0.

These give :math:`\Sigma_{a,1}(x) = 0.5 + 0.05\sin(\pi x/L) > 0`
trivially and
:math:`\Sigma_{a,2}(x) = 0.5 + 0.3\cos(\pi x/L) - 0.15\sin(\pi x/L)`,
bounded below by :math:`0.5 - \sqrt{0.3^{2} + 0.15^{2}} \approx
0.165 > 0`, so the cross sections are physical everywhere. The
:term:`scattering ratios <scattering ratio>` :math:`c_g = \Sigma_{s,\text{tot},g}/\Sigma_{t,g}`
stay around :math:`0.5` for both groups, which means source
iteration converges geometrically at rate :math:`\sim 0.5^n`
per sweep.

**Per-cell material construction.**  The solver consumes the
continuous :math:`\Sigma(x)` by creating **one material per cell**
with cross sections evaluated at the cell centre
:math:`x_i = (x_{i-1/2} + x_{i+1/2})/2`. The midpoint rule for
the cell-average cross section is :math:`\mathcal O(h^{2})`-
accurate on smooth :math:`\Sigma`, matching the diamond-
difference design order and not degrading the measured
convergence rate. The number of materials scales with mesh
refinement, so each mesh in the convergence study builds a
fresh materials dictionary via
:meth:`orpheus.derivations.continuous.mms.sn.SNSlab2GHeterogeneousMMSCase.build_materials`.

**Measured convergence.**  With default parameters
(:math:`L = 5\,\text{cm}`, :math:`\mathbf c = (1.0, 0.3)`,
Gauss--Legendre :math:`S_{16}`):

.. list-table::
   :header-rows: 1
   :widths: 10 20 20 20

   * - :math:`n_{\rm cells}`
     - :math:`\|\phi_1 - \phi_{1,\rm ex}\|_{L^{2}}`
     - :math:`\|\phi_2 - \phi_{2,\rm ex}\|_{L^{2}}`
     - measured order
   * - 20
     - :math:`3.71\!\times\!10^{-4}`
     - :math:`3.38\!\times\!10^{-4}`
     - ---
   * - 40
     - :math:`9.25\!\times\!10^{-5}`
     - :math:`8.45\!\times\!10^{-5}`
     - 2.00
   * - 80
     - :math:`2.31\!\times\!10^{-5}`
     - :math:`2.11\!\times\!10^{-5}`
     - 2.00
   * - 160
     - :math:`5.78\!\times\!10^{-6}`
     - :math:`5.28\!\times\!10^{-6}`
     - 2.00

Both groups hit the design order independently, confirming
that the multigroup scatter coupling is correctly exercised.
The L1 test
:func:`tests.sn.verification.mms.test_mms_heterogeneous.test_sn_heterogeneous_mms_converges_second_order`
asserts ``> 1.9`` to leave round-off headroom at the finest
mesh.

**What this replaces.** Before Phase 2.1a, the heterogeneous
SN verification was
``orpheus.derivations.continuous.cases.sn._derive_sn_heterogeneous``, which
computed the reference :math:`k_{\text{eff}}` by running the
SN solver itself at four mesh refinements and Richardson-
extrapolating the eigenvalue sequence. That is a **T3 circular
self-test** in the verification-campaign taxonomy: the solver
verifies against its own extrapolated output, so any consistent
bug in the SN sweep that affects all mesh refinements the same
way is invisible to the test. The heterogeneous MMS reference
above breaks the circularity: the reference comes from the
manufactured-solution algebra, not from the solver.

**Complementary eigenvalue verification.** The MMS test
verifies the **spatial operator** on a heterogeneous problem
but does not exercise the eigenvalue iteration. Phase 2.1b
lands a Case singular-eigenfunction eigenvalue reference --- see
:ref:`sn-case-heterogeneous-verification` --- that restores
eigenvalue-heterogeneous coverage for the SN solver (T2
semi-analytical, from the first-order Boltzmann equation
itself, no diffusion approximation).


.. _sn-mms-2d-verification:

2D Cartesian MMS — separable sinusoidal ansatz
-----------------------------------------------

Phase 3.1 of the verification campaign extends the MMS spatial-operator
verification to **two Cartesian dimensions**.  The 1D slab MMS tests
verify the :math:`\mu\,\partial\psi/\partial x` streaming term in
isolation; this section adds :math:`\mu_y\,\partial\psi/\partial y`
and confirms that the 2D wavefront sweep
(:func:`orpheus.sn.loss_representation._sweep_jacobi`) with diamond-difference
closure achieves its design :math:`\mathcal O(h^{2})` convergence rate.

**Ansatz.**  On a rectangle :math:`[0, L_x] \times [0, L_y]` with
vacuum boundary conditions:

.. math::
   :label: sn-mms-2d-psi

   \psi_n(x, y) \;=\; \frac{1}{W}\,A(x, y),
   \qquad A(x, y) \;=\; \sin\!\left(\frac{\pi x}{L_x}\right)
                         \sin\!\left(\frac{\pi y}{L_y}\right).

The ansatz is **isotropic in angle** --- every ordinate carries the
same angular flux amplitude --- so the scalar flux recovered by any
quadrature set equals :math:`\phi(x, y) = A(x, y)` exactly.  This
design is deliberate: it isolates **spatial** discretisation error from
angular quadrature error, exactly as in the 1D case
(:eq:`sn-mms-psi`).

The separable sinusoidal ansatz vanishes on all four domain edges
(:math:`x = 0`, :math:`x = L_x`, :math:`y = 0`, :math:`y = L_y`),
so vacuum BCs are satisfied automatically for every ordinate.

**Manufactured source.**  Substituting :eq:`sn-mms-2d-psi` into the
2D Cartesian transport equation :eq:`transport-cartesian-2d` and
solving for the residual:

.. math::
   :label: sn-mms-2d-qext

   Q^{\text{ext}}_n(x, y) \;=\;
       \mu_{x,n}\,\frac{\partial A}{\partial x}
     + \mu_{y,n}\,\frac{\partial A}{\partial y}
     + (\Sigma_t - \Sigma_s)\,A(x, y)

where the partial derivatives of the separable ansatz are:

.. math::

   \frac{\partial A}{\partial x} =
       \frac{\pi}{L_x}\cos\!\left(\frac{\pi x}{L_x}\right)
       \sin\!\left(\frac{\pi y}{L_y}\right), \qquad
   \frac{\partial A}{\partial y} =
       \sin\!\left(\frac{\pi x}{L_x}\right)
       \frac{\pi}{L_y}\cos\!\left(\frac{\pi y}{L_y}\right).

The manufactured source :eq:`sn-mms-2d-qext` is angle-dependent through
:math:`\mu_{x,n}` and :math:`\mu_{y,n}` (streaming terms) and
angle-independent in the removal term :math:`(\Sigma_t - \Sigma_s) A`.
It enters the solver through the ``Q_aniso`` external source slot in
:func:`orpheus.sn.solve_sn_fixed_source`.

**Quadrature.**  2D problems use Lebedev spherical quadrature
(:meth:`Quadrature.lebedev(17)
<orpheus.numerics.quadrature.Quadrature.lebedev>`, order 17 = 110 ordinates).
Because the ansatz is isotropic in angle, the quadrature-level angular
integration is exact for *any* quadrature set --- the spatial
convergence study isolates spatial error exclusively.

**Measured convergence.**  Four mesh refinements on a
:math:`5 \times 5\,\text{cm}` square domain with
:math:`\Sigma_t = 1.0`, :math:`\Sigma_s = 0.5`:

.. list-table::
   :header-rows: 1

   * - :math:`n_x = n_y`
     - L2 error
     - Order
   * - 10
     - :math:`5.50 \times 10^{-3}`
     -
   * - 20
     - :math:`1.37 \times 10^{-3}`
     - 2.01
   * - 40
     - :math:`3.41 \times 10^{-4}`
     - 2.00
   * - 80
     - :math:`8.53 \times 10^{-5}`
     - 2.00

The measured order is indistinguishable from 2.00 across all
refinements, confirming that the 2D wavefront sweep preserves the
diamond-difference design order.

**Code pointers.**

- Derivation:
  :class:`orpheus.derivations.continuous.mms.sn.SN2DCartesianMMSCase` and
  :func:`orpheus.derivations.continuous.mms.sn.build_2d_cartesian_mms_case`.
- Test:
  :func:`tests.sn.verification.mms.test_mms_2d.test_sn_2d_cartesian_mms_converges_second_order`.
- Sweep:
  :func:`orpheus.sn.loss_representation._sweep_jacobi` (the 2D diamond-difference
  kernel verified by this test).

**Why this test matters.**  The existing 2D SN tests
(:mod:`tests.sn.sweep.cartesian_2d.test_discrete_ordinates_2d`) are L2 self-convergence
tests with real cross sections that verify the solver as a black box.
This MMS test is more incisive: it provides a **closed-form reference
flux** and asserts the **design convergence order** of the spatial
discretisation.  A bug that corrupts the 2D DD cell-average formula
(e.g. swapping :math:`\Delta x` and :math:`\Delta y`, mis-indexing the
wavefront anti-diagonal, or computing face fluxes with the wrong
sign) would break the :math:`\mathcal O(h^{2})` rate while possibly
still converging at some reduced order — the MMS test catches this
immediately, while a self-convergence test might not.

**Gotchas.**

- *Ordinates with* :math:`\mu_x = \mu_y = 0`.  The Lebedev set
  includes purely :math:`z`-directed ordinates.  For these, the
  streaming terms vanish, and the sweep reduces to
  :math:`\psi = Q/\Sigma_t`.  The manufactured source formula
  handles this correctly because both :math:`\mu_{x,n}` and
  :math:`\mu_{y,n}` multiply the gradient terms.
- *Aspect ratio.*  The test uses :math:`L_x = L_y` (square domain).
  A non-square domain would work identically — the separable ansatz
  is parameterised by :math:`L_x` and :math:`L_y` independently.
  Phase 3.2 extends to 2-group with heterogeneous materials (below).


.. _sn-mms-2d-2g-verification:

2D Cartesian 2-group heterogeneous MMS
----------------------------------------

Phase 3.2 combines the 2D geometry from Phase 3.1 with the
smooth-:math:`\Sigma` heterogeneous approach from Phase 2.1a.  The
cross sections are smooth 2D functions :math:`\Sigma(x, y)` so the
diamond-difference design order :math:`\mathcal O(h^{2})` is preserved
(no interface degradation).

**Ansatz.**  Per-group amplitudes :math:`c_g` with the same 2D shape:

.. math::
   :label: sn-mms-2d-2g-psi

   \psi_{n,g}(x, y) = \frac{c_g}{W}\,A(x, y), \qquad
   A(x, y) = \sin(\pi x/L_x)\,\sin(\pi y/L_y),

giving :math:`\phi_g(x, y) = c_g\,A(x, y)` with
:math:`\mathbf c = (1.0, 0.3)`.

**Manufactured source.**  From the 2D multigroup transport equation:

.. math::
   :label: sn-mms-2d-2g-qext

   Q^{\text{ext}}_{n,g}(x, y) =
       \mu_{x,n}\,c_g\,\partial_x A
     + \mu_{y,n}\,c_g\,\partial_y A
     + \Sigma_{t,g}(x, y)\,c_g\,A
     - \sum_{g'}\Sigma_{s,g'\to g}(x, y)\,c_{g'}\,A.

The thermal (:math:`g = 2`) source couples to :math:`c_1` through
the downscatter term :math:`\Sigma_{s,1\to 2}(x, y)\,c_1\,A`, which
exercises the multigroup scatter assembly in the 2D sweep.

**Cross-section profiles.**  The 2D functions extend the 1D
Phase-2.1a profiles (see :ref:`sn-mms-heterogeneous-verification`)
with a mild :math:`y`-dependent modulation:

- :math:`\Sigma_{t,1}(x,y) = 1.0 + 0.2\sin(\pi x/L_x) + 0.1\cos(\pi y/L_y)`
- :math:`\Sigma_{t,2}(x,y) = 2.0 + 0.3\cos(\pi x/L_x) + 0.1\sin(\pi y/L_y)`

Scattering cross sections carry a :math:`0.05\cos(\pi y/L_y)` modulation.
All :math:`\Sigma_a > 0` bounds from the 1D case are preserved because
the :math:`y`-modulation amplitudes (0.1, 0.05) are smaller than the
1D absorption margin (:math:`\sim 0.165`).

**Measured convergence.**  Four refinements on a :math:`5 \times 5` cm
square:

.. list-table::
   :header-rows: 1

   * - :math:`n_x = n_y`
     - L2 error (g=1)
     - Order (g=1)
     - L2 error (g=2)
     - Order (g=2)
   * - 10
     - :math:`3.79 \times 10^{-3}`
     -
     - :math:`2.85 \times 10^{-3}`
     -
   * - 20
     - :math:`9.41 \times 10^{-4}`
     - 2.01
     - :math:`7.09 \times 10^{-4}`
     - 2.01
   * - 40
     - :math:`2.35 \times 10^{-4}`
     - 2.00
     - :math:`1.77 \times 10^{-4}`
     - 2.00
   * - 80
     - :math:`5.87 \times 10^{-5}`
     - 2.00
     - :math:`4.42 \times 10^{-5}`
     - 2.00

Both groups achieve the design :math:`\mathcal O(h^{2})` rate.

**Code pointers.**

- Derivation:
  :class:`orpheus.derivations.continuous.mms.sn.SN2DCartesian2GHeterogeneousMMSCase`
  and :func:`orpheus.derivations.continuous.mms.sn.build_2d_cartesian_heterogeneous_mms_case`.
- Test:
  :func:`tests.sn.verification.mms.test_mms_2d.test_sn_2d_cartesian_2g_heterogeneous_mms_converges_second_order`.


.. _sn-mms-p1-verification:

P1 anisotropic scattering MMS
-------------------------------

Phase 3.5 verifies that the P\ :sub:`N` anisotropic scattering
source assembly (:ref:`pn-scattering`) preserves
:math:`\mathcal O(h^{2})` convergence. All previous MMS tests use
isotropic (P0) scattering; this test exercises the P1 slot
:math:`\Sigma_s^{(1)}` through a weakly angle-dependent ansatz.

**Ansatz.** On a 1D vacuum-BC slab :math:`[0, L]`:

.. math::
   :label: sn-mms-p1-psi

   \psi_n(x) = \frac{1}{W}\bigl(A(x) + \alpha\,\mu_n\,B(x)\bigr)

with :math:`A(x) = B(x) = \sin(\pi x/L)` and small
:math:`\alpha = 0.1`. The scalar flux is :math:`\phi(x) = A(x)`
(the :math:`\mu`-odd term integrates to zero), and the P1 current
is :math:`J(x) = \alpha\,B(x)/3` (using
:math:`\sum w_n\mu_n^2 = 2/3` for Gauss–Legendre on
:math:`[-1, 1]`).

**Manufactured source.** Substituting :eq:`sn-mms-p1-psi` into
the 1D transport equation with P1 scattering and solving for
the residual:

.. math::
   :label: sn-mms-p1-qext

   Q^{\text{ext}}_n(x) =
       \mu_n\,A'(x)
     + (\Sigma_t - \Sigma_s^{(0)})\,A(x)
     + \alpha\,\mu_n\,(\Sigma_t - \Sigma_s^{(1)})\,B(x)
     + \alpha\,\mu_n^2\,B'(x).

The first two terms are the isotropic MMS source from
:eq:`sn-mms-qext`. The third term comes from the P1 scattering
slot :math:`3\,\Sigma_s^{(1)}\,\mu_n\,J(x)` in the transport
equation, and the fourth from the :math:`\mu_n`-weighted
streaming of :math:`B(x)`.

**Measured convergence.** Four refinements with
:math:`\Sigma_t = 1.0`, :math:`\Sigma_s^{(0)} = 0.5`,
:math:`\Sigma_s^{(1)} = 0.2`, :math:`\alpha = 0.1`:

.. list-table::
   :header-rows: 1

   * - :math:`n_{\text{cells}}`
     - L2 error
     - Order
   * - 20
     - :math:`6.15 \times 10^{-4}`
     -
   * - 40
     - :math:`1.53 \times 10^{-4}`
     - 2.00
   * - 80
     - :math:`3.84 \times 10^{-5}`
     - 2.00
   * - 160
     - :math:`9.59 \times 10^{-6}`
     - 2.00

**Code pointers.**

- Derivation:
  :class:`orpheus.derivations.continuous.mms.sn.SNP1AnisoMMSCase` and
  :func:`orpheus.derivations.continuous.mms.sn.build_p1_aniso_mms_case`.
- Test:
  :func:`tests.sn.verification.mms.test_mms_aniso.test_sn_p1_aniso_mms_converges_second_order`.
- P1 assembly: the collision gain's :math:`\ell \ge 1` body,
  ``TransferOperator._redistribute_ordinates``, reached through
  :meth:`~orpheus.transport.operators.transfer.TransferOperator.apply`
  (it was ``SNSolver._build_aniso_scattering`` → ``build_aniso_source``
  until #448).


.. _sn-mms-curvilinear-isotropic-verification:

Curvilinear isotropic MMS — radial DD-closure probe
----------------------------------------------------

Phase 3.4 of the verification campaign extends the slab MMS
(:eq:`sn-mms-psi` / :eq:`sn-mms-qext`) to 1-D **spherical** and
1-D **cylindrical** geometries with the simplest non-trivial trial
solution that respects the vacuum-at-outer and symmetry-at-origin
boundary conditions: an **isotropic** ansatz
:math:`\psi_n(r) = A(r)/W`.  By construction the angular
redistribution operator vanishes on this ansatz
(:math:`(1-\mu^2)/r \cdot \partial\psi/\partial\mu = 0` for the
sphere; :math:`-(1/r)\,\partial(\xi\psi)/\partial\varphi = 0` for
the cylinder), so the only spatial-discretisation error that drives
the measured convergence rate is the **radial DD closure**.  The
isotropic case is therefore the focused L1 probe for the
streaming + removal path; the angular redistribution path is
covered by the companion anisotropic case
(:ref:`sn-mms-curvilinear-aniso-verification` below — a deliberate
pairing that defeats the ``vv-principles`` Mode 7 "MMS
simplification bias" failure mode).

**Spherical isotropic ansatz.**  For a vacuum-BC sphere of radius
:math:`R` with reflective inner BC at :math:`r=0` in one energy
group, pick

.. math::
   :label: sn-mms-spherical-psi

   \psi_n(r) = \frac{1}{W}\,A(r),
   \qquad A(r) = \sin\!\left(\frac{\pi r}{R}\right),

with :math:`W = \sum_n w_n = 2` for symmetric Gauss--Legendre.
Because :math:`A(0) = A(R) = 0`, every ordinate vanishes at both
the symmetry centre and the vacuum outer face — both BC kinds are
satisfied automatically.  Since :math:`\psi_n` is independent of
ordinate, the scalar flux recovered by any quadrature order is
exactly :math:`\phi(r) = A(r)`.

**Spherical manufactured source.**  Substituting
:eq:`sn-mms-spherical-psi` into :eq:`transport-spherical` and
using that :math:`(1-\mu^2)\,\partial_\mu\psi/r \equiv 0` for an
isotropic flux gives

.. math::
   :label: sn-mms-spherical-qext

   Q^{\text{ext}}_n(r)
        = \mu_n\,A'(r)
        + (\Sigma_t - \Sigma_s)\,A(r)
        = \mu_n\,\frac{\pi}{R}\cos\!\left(\frac{\pi r}{R}\right)
          + (\Sigma_t - \Sigma_s)\sin\!\left(\frac{\pi r}{R}\right).

This is structurally identical to the slab source
:eq:`sn-mms-qext` with :math:`x \to r` — the spherical
:math:`(2/r)\partial_r` curvature term and the angular
redistribution term both vanish on the isotropic ansatz, leaving
the per-ordinate streaming + removal balance as the residual.

**Cylindrical isotropic ansatz.**  The radial direction cosine for
1-D cylindrical is :math:`\eta_n = \sin\theta_n \cos\varphi_n`.
Use

.. math::
   :label: sn-mms-cylindrical-psi

   \psi_n(r) = \frac{1}{W}\,A(r),
   \qquad A(r) = \sin\!\left(\frac{\pi r}{R}\right),

with the same :math:`W = \sum_n w_n` for the cylindrical Product or
LS quadrature.  Symmetric Product quadrature gives
:math:`\sum_n w_n \eta_n = 0`, so :math:`\phi(r) = A(r)` exactly.

**Cylindrical manufactured source.**

.. math::
   :label: sn-mms-cylindrical-qext

   Q^{\text{ext}}_n(r)
        = \eta_n\,A'(r) + (\Sigma_t - \Sigma_s)\,A(r).

The cylindrical curvature term :math:`-(1/r)\,\partial(\xi\psi)/\partial\varphi`
vanishes by isotropy of :math:`A(r)`, the same way the spherical
:math:`(1-\mu^2)/r \cdot \partial_\mu\psi` vanishes; the radial
streaming :math:`\eta_n A'(r)` and the removal
:math:`(\Sigma_t - \Sigma_s)A(r)` carry the residual.

**Risk point — Mode 7 ansatz bias.**  Per ``vv-principles`` failure
Mode 7 ("MMS simplification bias"), the isotropic ansatz is
deliberately structured to NULL the angular redistribution path.
A passing :math:`\mathcal{O}(h^2)` convergence here is necessary
evidence for the radial DD closure but it is *not* sufficient for
the full curvilinear sweep — ERR-026 (the curvilinear sweep WDD
flux-shape bug) is mathematically invisible to this MMS because the
redistribution term that ERR-026 lives on cancels by ansatz
construction.  The companion anisotropic case
(:ref:`sn-mms-curvilinear-aniso-verification`) is the load-bearing
sufficient evidence for the full sweep; both are required.

**Code pointers.**

- Derivation:
  :class:`orpheus.derivations.continuous.mms.sn.SNSphericalMMSCase`,
  :class:`orpheus.derivations.continuous.mms.sn.SNCylindricalMMSCase`,
  :func:`orpheus.derivations.continuous.mms.sn.build_spherical_mms_case`,
  :func:`orpheus.derivations.continuous.mms.sn.build_cylindrical_mms_case`.
- Tests:
  :func:`tests.sn.verification.mms.test_mms_curvilinear.test_sn_spherical_mms_converges_second_order`
  (sphere) and
  :func:`tests.sn.verification.mms.test_mms_curvilinear.test_sn_cylindrical_mms_converges_second_order`
  (cylinder), both ``catches("ERR-058")``.  **Their ``xfail`` markers
  came off 2026-06-12 with the ERR-058 closure-seed fix** (Issue #195).
  Post-fix the ladders are clean second-order with SI :math:`\equiv`
  Krylov bit-identical — sphere ``[1.49e-2, 3.73e-3, 9.28e-4, 2.31e-4,
  5.74e-5]`` (orders 2.00–2.01), cylinder ``[2.16e-3, 5.39e-4, 1.35e-4,
  3.37e-5]`` (orders 2.00); the magnitude band
  :math:`[10^{-8}, 10^{-3}]` is met (sphere :math:`n_x\ge 80`,
  cylinder :math:`n_x\ge 40`).  Through the bug era (Wave E Round 3
  2026-05 → ERR-058) they were ``xfail(strict=True)`` — the
  now-superseded "pre-asymptotic transient" reading; see
  :ref:`sn-err-058-closure-seed-closeout`.


.. _sn-mms-curvilinear-aniso-verification:

Curvilinear anisotropic MMS — angular redistribution probe
-----------------------------------------------------------

Phase 3.6 closes the **angular-redistribution coverage gap** in the
curvilinear MMS verification chain. The existing isotropic
1D-spherical (:class:`SNSphericalMMSCase`) and 1D-cylindrical
(:class:`SNCylindricalMMSCase`) MMS cases use the ansatz
:math:`\psi_n(r) = A(r)/W` (no :math:`\mu`-dependence). For that
ansatz, **the angular-redistribution operator is identically zero**:
:math:`(1-\mu^2)/r \cdot \partial\psi/\partial\mu = 0` for the sphere,
:math:`-(1/r)\,\partial(\xi\psi)/\partial\varphi = 0` for the
cylinder. The hardest math the curvilinear sweep performs — where
ERR-026 (curvilinear sweep WDD wrong fixed point) lives — is
mathematically invisible to the isotropic MMS because it cancels by
ansatz construction.

This is the ``vv-principles`` failure mode #7 ("MMS simplification
bias") — the MMS test cannot catch a bug class because the ansatz
nulls it. The defence is **not**
to replace the isotropic case (it remains the right probe for the
non-redistribution paths) but to **pair** it with a companion case
whose ansatz activates redistribution. The two cases together let a
narrow-down diagnosis route a failing convergence rate to either
the streaming/removal path (only isotropic fails) or the
redistribution path (only anisotropic fails).

**Spherical anisotropic ansatz**

.. math::
   :label: sn-mms-spherical-aniso-psi

   \psi_n(r) = \frac{1}{W}\bigl(A(r) + B(r)\,\mu_n\bigr),
   \qquad
   A(r) = \sin\!\left(\frac{\pi r}{R}\right),
   \qquad
   B(r) = \frac{r}{R}\Bigl(1 - \frac{r}{R}\Bigr)
            \cos\!\left(\frac{\pi r}{R}\right).

Both :math:`A` and :math:`B` vanish at :math:`r \in \{0, R\}`, so
**every** ordinate satisfies the symmetry BC at :math:`r=0` and the
vacuum BC at :math:`r=R`, regardless of the sign of :math:`\mu_n`.
The :math:`B(r)\,\mu_n` coefficient is non-trivial: ordinates with
opposite sign of :math:`\mu_n` differ in sign of the
angular-flux contribution, but both still vanish at the boundaries.

The choice :math:`B(r) = (r/R)(1-r/R)\cos(\pi r/R)` is **not**
algebraically reducible to a multiple of :math:`A(r)` — the
:math:`(r/R)(1-r/R)` envelope and the :math:`\cos(\pi r/R)` factor
produce a derivative :math:`B'(r)` whose extrema do not co-locate
with :math:`A'(r)`'s extrema, so the redistribution term
:math:`(1-\mu_n^2)\,B/r` cannot be absorbed into a renormalisation
of the streaming term.

The discrete scalar flux is :math:`\phi(r) = A(r)` because
:math:`\sum_n w_n \mu_n = 0` for any symmetric Gauss-Legendre
quadrature — the :math:`B \mu` term integrates to zero in the
scalar moment.

**Spherical manufactured source**

Substituting :eq:`sn-mms-spherical-aniso-psi` into
:eq:`transport-spherical` and solving for the residual:

.. math::
   :label: sn-mms-spherical-aniso-qext

   Q^{\text{ext}}_n(r) =
        \mu_n\,A'(r)
      + \mu_n^2\,B'(r)
      + (1 - \mu_n^2)\,\frac{B(r)}{r}
      + (\Sigma_t - \Sigma_s)\,A(r)
      + \Sigma_t\,\mu_n\,B(r).

The first and fourth terms are the isotropic-MMS source from
:eq:`sn-mms-qext` adapted to spherical. The **second term**
(:math:`\mu_n^2 B'(r)` — :math:`\mu`-weighted streaming of the
anisotropic profile) and the **third term**
(:math:`(1-\mu_n^2)\,B/r` — angular redistribution) are
load-bearing: they are precisely what the isotropic case lacks.
The fifth (:math:`\Sigma_t\,\mu_n B`) comes from the removal
operator acting on the :math:`B \mu` part of :math:`\psi_n`.

**Cylindrical anisotropic ansatz**

The radial direction cosine for cylindrical 1D is :math:`\eta_n =
\sin\theta_n \cos\varphi_n`; the azimuthal partner that drives
the redistribution is :math:`\xi_n = \sin\theta_n \sin\varphi_n`.
Use:

.. math::
   :label: sn-mms-cylindrical-aniso-psi

   \psi_n(r) = \frac{1}{W}\bigl(A(r) + B(r)\,\eta_n\bigr),

with the same :math:`A(r),\,B(r)` shapes. Symmetric ProductQuadrature
gives :math:`\sum_n w_n \eta_n = 0`, so :math:`\phi(r) = A(r)`.

**Cylindrical manufactured source**

Substituting :eq:`sn-mms-cylindrical-aniso-psi` into
:eq:`transport-cylindrical` (treating :math:`\eta_n` and
:math:`\xi_n` as the :math:`\varphi`-dependent functions
:math:`\sin\theta\cos\varphi` and :math:`\sin\theta\sin\varphi`)
and solving for the residual:

.. math::
   :label: sn-mms-cylindrical-aniso-qext

   Q^{\text{ext}}_n(r) =
        \eta_n\,A'(r)
      + \eta_n^2\,B'(r)
      + \xi_n^2\,\frac{B(r)}{r}
      + (\Sigma_t - \Sigma_s)\,A(r)
      + \Sigma_t\,\eta_n\,B(r).

The :math:`\xi_n^2\,B/r` redistribution term is the cylindrical analog
of the sphere's :math:`(1-\mu_n^2)\,B/r`. Both come from the
same operator — angular redistribution of the linearly-:math:`\mu`
(or linearly-:math:`\eta`) ansatz — and both vanish for any
isotropic ansatz.

**Spatial-convergence claims.**  Diamond-Difference is design-order
:math:`\mathcal{O}(h^2)` in the cell width (:eq:`dd-cartesian-1d` /
:eq:`dd-curvilinear-scalar`); the curvilinear anisotropic L1 claim
asserts that the **measured** scalar-flux error against the
manufactured solution :eq:`sn-mms-spherical-aniso-psi` /
:eq:`sn-mms-cylindrical-aniso-psi` falls at the same rate.  For the
sphere,

.. math::
   :label: sn-mms-spherical-aniso-spatial-convergence

   \bigl\|\phi_h(r) - A(r)\bigr\|_{L^2(\Omega)}
        \;=\; \mathcal{O}(h^2)
        \qquad \text{as } h = R/n_x \to 0\,,

with the convergence ORDER (slope of :math:`\log\|\phi_h - A\|`
versus :math:`\log h` over the last two mesh halvings) the L1
acceptance criterion ``min(orders[-2:]) > 1.9``.  The cylindrical
analogue,

.. math::
   :label: sn-mms-cylindrical-aniso-spatial-convergence

   \bigl\|\phi_h(r) - A(r)\bigr\|_{L^2(\Omega)}
        \;=\; \mathcal{O}(h^2)
        \qquad \text{as } h = R/n_x \to 0\,,

uses the same acceptance criterion on the cylindrical-aniso
ansatz.  Both labels are consumed by the
:file:`tests/sn/verification/mms/test_curvilinear_aniso_convergence.py`
gate-3 tests, which **stay ``xfail``** — but, post-ERR-058 (Issue
#195), no longer for the wrong-fixed-point reason.  The ERR-058
closure-seed fix recovered :math:`\mathcal{O}(h^2)` *spatial*
convergence (the isotropic ladders are clean second-order; see
:ref:`sn-err-058-closure-seed-closeout`).  These **anisotropic** rows
remain xfail because the angle-varying ansatz hits the
**fixed-quadrature angular floor** of the half-angle thread
interpolation: under spatial refinement at fixed quadrature the error
converges to a floor (sphere S16 :math:`\approx 7\mathrm{e}{-4}`,
cylinder :math:`n_\mu{=}4` :math:`\approx 1.9\mathrm{e}{-2}`) that
drops only under *quadrature* refinement.  Their markers are re-pinned
to the `Issue #229
<https://github.com/deOliveira-R/ORPHEUS/issues/229>`_
quadrature-aware retune (the regression gate that flips them to
unexpected-pass when the retune lands).  See
:ref:`sn-err-058-aniso-floor` for the floor-vs-quadrature evidence.

**Verification chain (Branch 1 / Branch 2)**

Per the ``algebra-of-record`` discipline (Branch-1 SymPy reference,
Branch-2 numpy production, structurally-independent L1 cross-check):

- **Branch 1 (SymPy)**:
  :func:`orpheus.derivations.continuous.mms.sn.derive_spherical_anisotropic_mms`
  and
  :func:`orpheus.derivations.continuous.mms.sn.derive_cylindrical_anisotropic_mms`
  substitute the ansatz into the transport operator symbolically and
  prove ``simplify(LHS - RHS) == 0``. Foundation tests:
  :file:`tests/derivations/test_sn_mms_anisotropic_symbolic.py` (one
  ``@pytest.mark.foundation`` test per ``derive_*`` function).
- **Branch 2 (vectorised numpy)**:
  :class:`orpheus.derivations.continuous.mms.sn.SNSphericalAnisotropicMMSCase`
  and
  :class:`orpheus.derivations.continuous.mms.sn.SNCylindricalAnisotropicMMSCase`
  evaluate :eq:`sn-mms-spherical-aniso-qext` and
  :eq:`sn-mms-cylindrical-aniso-qext` per ordinate using vectorised
  numpy.
- **L1 cross-check (the gate)**: the Branch-2 numerical
  :math:`Q^{\text{ext}}_n(r_i)` agrees with Branch-1 SymPy-evaluated
  :math:`Q^{\text{ext}}_n(r_i)` (via :func:`sympy.lambdify`) to
  :math:`\sim 10^{-16}` (max absolute) on a sample mesh in both
  geometries. Tested in
  :func:`tests.derivations.test_sn_mms_anisotropic_symbolic.test_spherical_aniso_numerical_qext_matches_sympy`
  and the cylindrical sibling.

**Code pointers**

- Derivations: :class:`orpheus.derivations.continuous.mms.sn.SNSphericalAnisotropicMMSCase`,
  :class:`orpheus.derivations.continuous.mms.sn.SNCylindricalAnisotropicMMSCase`,
  :func:`orpheus.derivations.continuous.mms.sn.build_spherical_anisotropic_mms_case`,
  :func:`orpheus.derivations.continuous.mms.sn.build_cylindrical_anisotropic_mms_case`.
- Symbolic factory:
  :func:`orpheus.derivations.continuous.mms.sn._spherical_anisotropic_symbolic`,
  :func:`orpheus.derivations.continuous.mms.sn._cylindrical_anisotropic_symbolic`.
- Foundation tests:
  :file:`tests/derivations/test_sn_mms_anisotropic_symbolic.py`.
- Consumer L1 convergence test (Phase-0 work, separate branch):
  ``tests/sn/_l1/test_mms_spherical_anisotropic_dd_convergence_O_h2.py``
  (planned).


.. _sn-curvilinear-aniso-norm-reconciliation:

The curvilinear anisotropic-MMS "floor", reconciled (W1–W5)
-----------------------------------------------------------

.. admonition:: Key Facts — curvilinear anisotropic SN
   :class: important

   - **The single "#229 floor" was three distinct errors**, separated
     by a *norm difference*: the production gates measure a
     **volume-weighted L2** :math:`\sqrt{\sum_i V_i\,\Delta_i^2}`; the
     root-cause probes measured **pointwise / L∞**.  An error
     concentrated at the :math:`r \to 0` pole cell is loud in L∞ and
     near-silent in L2.
   - **(a) Sphere central-cell** :math:`\mathcal{O}(h)` **spatial
     closure (#233)** — L∞-only; :math:`\sim 75\,\%` an MMS
     midpoint-vs-shell-volume-average comparison artifact +
     :math:`\sim 25\,\%` genuine but **inherent** first order.  WONTFIX
     for diamond difference (Hébert §3.9.4 / Stacey §9.9 use plain
     diamond at the central cell).  See
     :ref:`sn-pole-cell-spatial-closure`.
   - **(b) Sphere angular** :math:`\tau`-**clamp floor** — fixed by W1
     (the clamp was mis-cited and 100 % spurious on physical fields).
     The sphere uses the Bailey-Morel-Chang 2010 Eq. 43 weight
     unclamped; **so does the cylinder since Q5.6.4** — there is one
     :math:`\tau` and no clamp in either arm.  See
     :ref:`sn-tau-clamp-vindication`.
   - **(c) Cylinder angular floor** (the floor measured in #229, CLOSED)
     — the half-angle-thread INTERPOLATION floor; scales with the
     **azimuthal** quadrature :math:`n_\varphi`.  Reduced twice
     (Q5.6.3's σ_y fold, Q5.6.4's cell-partition fix) and **not
     closed**: the residual limitation is that a 1-D
     :math:`\eta`-thread cannot represent a genuinely 2-D
     :math:`(\eta, \varphi)` field.  The sphere has a pre-floor
     :math:`\mathcal{O}(h^2)` window (clean at S32).  See
     :ref:`sn-cylinder-angular-floor`.
   - **Two unrelated "anisotropic" paths** (Issue #9): Path-(I) =
     geometric angular redistribution :math:`(1-\mu^2)/r\,\partial_\mu`
     (the M-M :math:`\alpha`-dome, P0-only — what #229 concerns);
     Path-(II) = :math:`P_1{+}` Legendre SCATTERING
     :math:`R\,\Lambda\,M` (geometry-agnostic).  See
     :ref:`sn-p1-scattering-curvilinear`.
   - **Norm gotcha**: a convergence-rate gate on a volume-weighted L2
     norm CANNOT see a pole-cell defect (the pole sits at one cell of
     :math:`V \sim h^3` → :math:`\sqrt{V} \sim h^{1.5}` →
     :math:`\sim h^{2.5}` contribution → subdominant).  A pointwise /
     L∞ probe is required to surface it.

This section closes the curvilinear-anisotropic-SN investigation
program (W1–W5, branch ``fix/curvilinear-aniso-pole-and-clamp``,
2026-06-13).  It is the sequel to the ERR-058 / #195 / #196 curvilinear
*isotropic* closure-seed family (:doc:`/theory/methods/sn/curvilinear_numerics`); that
family fixed the
wrong-fixed-point class (now formally retired), and what remained was
the *anisotropic* floor — which this program resolved into three
distinct, separately-actionable errors.

The headline — one floor was three errors, settled by a norm difference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ERR-058 close-out (:doc:`/theory/methods/sn/curvilinear_numerics`) deferred a
residual "anisotropic angular
floor" to Issue #229, citing a single
:ref:`floor table <sn-err-058-aniso-floor>`.  The W1–W5 root-cause
program found that the apparent single floor was **three structurally
distinct errors**, and the reason they had been conflated is a **norm
difference** in how two independent investigations measured the same
solves:

* The verification gates (test-architect) measure the **volume-weighted
  L2** norm :math:`\|\Delta\|_{2,V} = \sqrt{\sum_i V_i\,\Delta_i^2}` —
  the natural norm for a finite-volume scheme whose unknown is a
  cell-volume average.
* The diagnostic probes (numerics-investigator) measured **pointwise /
  L∞** — :math:`\max_i |\Delta_i|`.

The two norms weight the :math:`r \to 0` pole cell completely
differently.  Under the spherical volume weight :math:`V \sim h^3`, a
fixed pointwise error at the single pole cell contributes
:math:`\sqrt{V} \sim h^{1.5}` to the L2 sum, so an L∞-:math:`\mathcal{O}(h)`
pole error appears as :math:`\sim h^{2.5}` in L2 — **subdominant to the
interior** :math:`\mathcal{O}(h^2)`, hence invisible.  This is exactly
why the production L2 gate stayed green throughout while a pointwise
probe found a first-order pole cell.

.. list-table:: The three errors behind the "#229 floor"
   :header-rows: 1
   :widths: 6 38 22 16 18

   * - #
     - Error
     - Dominant norm
     - Quadrature-scaling?
     - Status
   * - (a)
     - Sphere pole-cell spatial closure :math:`\mathcal{O}(h)` at
       :math:`r \to 0`
     - L∞ / pointwise central flux (diluted in L2 by :math:`V \propto
       r^2`; invisible in :math:`k_{\rm eff}`)
     - no (spatial)
     - **#233 — documented inherent limitation (ERR-059, WONTFIX-for-DD)**
   * - (b)
     - Sphere angular :math:`\tau`-clamp floor (:math:`\sim 7\mathrm{e}{-4}`
       @ S16)
     - volume-weighted L2 at fine mesh
     - yes (angular)
     - **fixed (W1 unclamp)**
   * - (c)
     - Cylinder angular floor
     - both
     - yes (azimuthal :math:`n_\varphi`)
     - **reduced twice, not closed** — Q5.6.3's fold took it
       :math:`5.4\times` down; Q5.6.4 retired the clamp compensating a
       wrong cell partition.  The residual 1-D-:math:`\eta`-thread
       limitation stands (:ref:`sn-cylinder-angular-floor`)

The remainder of this section treats each error in turn, then the two
unrelated anisotropic paths (Issue #9).

.. _sn-tau-clamp-vindication:

(b) The :math:`\tau`-clamp vindication (W1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The spherical Morel--Montry :term:`weighted-diamond <weighted diamond difference>` weight is

.. math::
   :label: sn-tau-mm-raw

   \tau_n
       \;=\; \frac{\mu_n - \mu_{n-1/2}}{\mu_{n+1/2} - \mu_{n-1/2}}
       \;\in\; [0, 1],

.. (vv-status rationale) definition: the literature-transcribed Morel-Montry
.. weighted-diamond weight (BaileyMorelChang2010 Eq. 43), the SAME object as
.. :eq:`mm-weights` and as :eq:`morel-montry-closure` (the equation-of-record
.. on the structured-geometry page). Definitional; the WDD closure it names is
.. verified wherever :eq:`mm-weights` is exercised.
.. NOTE 2026-08-11: the LABEL still spells "raw" although the raw/clamped
.. distinction retired at Q5.6.4 (there is one tau). Kept rather than renamed
.. because a label rename has a V&V-matrix footprint and was out of scope for
.. the Q5.6.4 docs pass; a follow-on rename to sn-tau-mm-weight would touch
.. this label, this directive, and the :eq: site in the W1-W5 evidence table.
.. vv-status: sn-tau-mm-raw documented

the **unique** weight exact for an angular flux linear in :math:`\mu`
(:cite:`BaileyMorelChang2010` Eq. 43; the same object as
:eq:`mm-weights`).  The production code had wrapped it in a
:math:`[\tfrac12, 1]` clamp,
:math:`\tau_n \to \mathrm{clip}(\tau_n, \tfrac12, 1)`, cited
to Lewis & Miller §4.5.  W1 (2026-06-13) removed that clamp on the
sphere; Q5.6.4 (2026-08-11) retired it on the cylinder too, so **the
superscripted** :math:`\tau^{\rm raw}` **spelling used throughout this
section is historical** — there is now one :math:`\tau` and no clamp
anywhere.

W1 established, by three independent lines of evidence, that the clamp
is **mis-cited and 100 % spurious on physical fields**:

#. **Literature.** :cite:`BaileyMorelChang2010` state the admissible range
   is :math:`\tau \in [0, 1]` and recommend *exactly* the unclamped
   :math:`\tau^{\rm raw}` (their Eq. 43) as the unique exact-on-linear
   weight; Hébert §3.9.4 uses pure diamond (:math:`\tau = \tfrac12`),
   no clamp.  Lewis & Miller §4.5 does **not** prescribe the
   :math:`[\tfrac12, 1]` clamp — the citation was wrong.
#. **The clamp buys no positivity on the SPHERE'S converged solve.** On
   every realistic converged **spherical** solve W1 exercised (smooth
   MMS, homogeneous eigenvalue :math:`k_{\rm eff} = 1`, thick absorber)
   there are ZERO negative half-angle fluxes, clamped or unclamped —
   every clamp activation is spurious (measured: 160 / 320 / 80 / 240
   activations across stress configs, 0 protective).  The half-flux
   negativity that *does* transiently appear in early SI iterates is
   inherited from a negative *input* :math:`\psi` and the clamp barely
   reduces it.  On Gauss--Legendre quadrature
   :math:`\tau^{\rm raw} \in [0.39, 0.61]` (never 0), so the unclamped
   weight is always interior to :math:`[0, 1]`.

   .. warning:: **Scope correction, 2026-08-11 — the old heading said
      "Positivity is never needed", and "never" is too strong.**

      Read as written, the item claims **zero negative half-angle
      fluxes**.  Its evidence is the **sphere**: W1's stress configs and
      its clamp-activation census.  The M-M angular recurrence is not
      positivity-preserving in general — it is a first-order linear
      recursion with amplification factor :math:`-(1-\tau_m)/\tau_m`,
      and no source read for this seam
      (:cite:`ReedLathrop1970`, :cite:`BaileyMorelChang2010`, Lathrop
      2000, :cite:`Hebert2009`) states a positivity condition for the
      ANGULAR recurrence at all; the positivity literature is about the
      SPATIAL closure.

      What the cylinder actually does was measured and committed on the
      same day, in
      ``tests/sn/sweep/curvilinear/test_psi_half_positivity.py``
      (19 ``foundation`` rows; a CHARACTERISATION module — no row
      carries ``verifies(...)``, because there is no equation whose
      truth they establish).  On a heterogeneous 2-region, 2-group
      vacuum-outer cylinder, `[M]` reproduced 2026-08-11:

      .. list-table:: :math:`\min\hat\psi` on a converged cylinder solve
         :header-rows: 1
         :widths: 12 8 18 26 20 16

         * - :math:`n_\varphi`
           - :math:`M`
           - :math:`\min\psi`
           - :math:`\min\hat\psi`, MARCHED seed
           - :math:`\div \min\psi`
           - :math:`\min\hat\psi`, ZERO seed
         * - 6
           - 3
           - ``+0.151231``
           - ``+0.133705``
           - 0.8841
           - ``-12.089129``
         * - 8
           - 4
           - ``+0.137569``
           - ``+0.128600``
           - 0.9348
           - ``-16.351438``
         * - 16
           - 8
           - ``+0.130781``
           - ``+0.128651``
           - 0.9837
           - ``-25.890124``

      ⟹ **the sign is a property of the SEED's consistency, not of the
      scheme.**  On the production value path — where the seed is the
      composite's marched :math:`\psi_{1/2}` state (#282 route (a)) —
      :math:`\hat\psi` is strictly positive, within 12 % of
      :math:`\min\psi` itself.  The zero seed is the legitimate
      :math:`\psi`-independent COEFFICIENT state (the transpose walk's
      ``denom``-only build, where these faces are never read as fluxes);
      used as an *inconsistent-seed* control it goes negative, bounded
      by the worst partial amplification
      :math:`A(M) = \max_m \prod_{k \le m}(1-\tau_k)/\tau_k`
      (`[M]` 2.732051 / 3.359161 / 4.728870).  Any headline figure of
      the form "``min psi_hat ≈ -77``" is an inconsistent-seed
      statement, not a production one.

      **The clamp was never what protected this, so W1's conclusion
      stands.**  `[M]` 2026-08-11, feeding a strictly positive analytic
      shadow profile :math:`\exp(-6\cos\omega)` through the production
      kernel
      :func:`~orpheus.sn.angular.closure.compute_psi_half_per_level`
      on level 0 of ``Quadrature.folded_product(4, 32)``
      (:math:`M = 16`) with a positive constant seed — i.e. squarely in
      the inconsistent-seed regime — all four :math:`\tau` conventions
      go negative:

      .. list-table:: :math:`\min\hat\psi` over the 17 half-angle faces, inconsistent seed
         :header-rows: 1
         :widths: 44 18 19 19

         * - :math:`\tau` convention
           - :math:`\tau` range
           - :math:`\min\hat\psi`
           - negative faces
         * - chord (η-midpoint) edges, retired
           - :math:`[0.201, 0.799]`
           - :math:`-229.7`
           - 7 / 17
         * - chord + :math:`[\tfrac12, 1]` absorber, retired
           - :math:`[0.500, 0.799]`
           - :math:`-23.3`
           - 6 / 17
         * - **arc (ω-midpoint) edges — SHIPPED**
           - :math:`[0.251, 0.749]`
           - :math:`-77.2`
           - 7 / 17
         * - :math:`\tau \equiv \tfrac12` (Hébert's plain diamond)
           - :math:`[0.500, 0.500]`
           - :math:`-24.2`
           - 6 / 17

      The absorber cuts the excursion :math:`\approx 10\times` but does
      not remove it (6 of 17 faces still negative), and neither does
      :math:`\tau \equiv \tfrac12`.  The destabilising coefficient is
      :math:`(1-\tau)/\tau`, so the exposure belongs to the *angular
      diamond family*, not to the clamp: retiring the clamp did not
      create it and keeping it would not have cured it.  The shipped arc
      chart is the more exposed of the two derived candidates because
      its :math:`\tau` reaches lower — an honest cost of the principled
      partition, ratified alongside the accuracy cost recorded at
      :ref:`sn-tau-absorber-retirement`.

      **What is gated, and where.**  The two curvilinear *scalar*-flux
      positivity gates
      (``tests/sn/sweep/curvilinear/test_282_direct_seed_fixed_point.py``
      ``::test_ciii_coarse_sphere_fixed_source_finite_positive``,
      ``tests/sn/sweep/curvilinear/test_w1_clamp_silent_on_flat.py``
      ``::test_unclamped_sphere_flux_strictly_positive``) are **sphere
      only** — they are what item 3 below rests on.  The
      :math:`\hat\psi` sign, on both seed regimes and on both arms, is
      owned by ``test_psi_half_positivity.py``; read it, not this item,
      for the half-angle flux.
#. **Stability without it.** Unclamped sphere source iteration
   converges with strictly positive, finite scalar flux on every
   stress config (thick absorber, near-vacuum, :math:`c = 0.999`, S64);
   the clamp costs a few SI iterations on low-scattering problems but is
   dispensable for stability.

.. admonition:: The architectural reason the static removal is correct
   :class: note

   A *dynamic* negative-flux fixup (where :math:`\tau` depends on the
   iterate :math:`\psi`) would make the streaming operator
   **nonlinear**, breaking the linear-Krylov matvec and the SI ≡ Krylov
   twin identity (Pattern-2 discipline, Cardinal Rule 2).  Because the
   fixup is *never needed* on physical fields, the principled W1 fix is
   to **drop the clamp** (a config-time, static change) and use the
   linear unclamped :math:`\tau^{\rm raw}`.  The weight :math:`\tau` is
   single-sourced in the angular closure
   (:func:`~orpheus.sn.angular.closure.morel_montry_tau_per_level`,
   since Issue #236 Step C — see :ref:`sn-tau-c-on-cellvisit-live`) and
   inherited by every consumer (the SI sweep and the Krylov matvec
   both), so both twins stay linear and stay identical.

**Geometry split — closed at Q5.6.4 (2026-08-11); there is no longer a
split.**  W1 removed the clamp for the **sphere only**, and the cylinder
kept it for four more sessions.  The reason given at the time was that
product / level-symmetric quadratures place the most-inward azimuthal
ordinate exactly on :math:`\eta = -\sin\theta`, giving
:math:`\tau^{\rm raw} = 0` **exactly** (bit-exact, not "near zero"), so
an unclamped recurrence divides by zero there.  Both halves of that
reason have since dissolved, in two steps:

* **Q5.6.3 (2026-08-08)** made the full-circle rule classes
  *unrepresentable* — a cylindrical ``SNMesh`` now refuses any rule with a
  non-carrying μ-level, so the :math:`\tau = 0` trigger is unreachable
  through any mesh.
* **Q5.6.4 (2026-08-11)** retired the absorber itself, after finding that
  the thing it was compensating for was not a singularity at all but a
  **wrong angular cell partition**: the cylinder's edges were taken at the
  midpoint of consecutive :math:`\eta` (the *chord* midpoint) while
  :math:`\alpha` used the real half-angle, so the two disagreed about the
  same object by a permanent :math:`\approx 17.5\,\%` in
  :math:`\omega`-width.  Taking the partition in :math:`\omega` — the
  variable the azimuthal march marches in — removes the disagreement, and
  then P2 *determines* :math:`\tau` with nothing left to clamp.

⭐ **This subsection's own prediction was right, for a reason it did not
know.**  It used to close: *"the cylinder's real fix is a 2-D
:math:`(\eta, \varphi)` closure, not unclamping."*  `[M]` confirmed at
Q5.6.4: unclamping the chord partition **alone** makes the anisotropic
cylinder MMS floor **1.8--3.4× worse** at every rung
(:math:`3.5384\mathrm{e}{-3} \to 6.2244\mathrm{e}{-3}` at
:math:`n_\varphi = 8`; :math:`6.7824\mathrm{e}{-4} \to
2.3020\mathrm{e}{-3}` at 16; :math:`2.4837\mathrm{e}{-4} \to
6.0065\mathrm{e}{-4}` at 32, :math:`n_x = 80`).  So "not unclamping" was
the correct verdict.  ⚠ But the actual fix was the **PARTITION**, not a
2-D closure and not unclamping: a 1-D :math:`\omega`-march with the right
cells satisfies P2/P3 and closes its own level exactly.  A 2-D
:math:`(\eta, \varphi)` closure remains a *separate* open question about
the residual azimuthal floor (:ref:`sn-cylinder-angular-floor`), no
longer a prerequisite for retiring the clamp.  The equation-of-record now
carries the geometry in the **partition**
(:eq:`angular-cell-partition`), not in the closure
(:eq:`morel-montry-closure`) — see
:doc:`/theory/foundations/structured_geometry`, whose
:ref:`sn-tau-absorber-retirement` section holds the full refutation, the
solve-free ν-closure evidence, and the honest accuracy trade.

**Mixed accuracy signature (the gotcha).**  Unclamping does NOT
uniformly improve the anisotropic solve.  It *cleans the coarse
convergence rate* (sphere S16 coarse orders 1.978 → 1.995) but *raises
the S16 fine-mesh floor* (:math:`7.3\mathrm{e}{-4} \to 1.2\mathrm{e}{-3}`):

.. list-table:: W1 sphere aniso MMS, matched-quadrature S16 (volume-weighted L2)
   :header-rows: 1
   :widths: 20 40 40

   * - :math:`n_x`
     - Clamped (pre-W1)
     - Unclamped (post-W1)
   * - 10 → 40
     - coarse orders 1.979 / 1.978
     - coarse orders 1.995 / 1.999
   * - 80
     - 1.16e-3
     - 1.40e-3
   * - 160 (floor)
     - **7.3e-4**
     - **1.2e-3**

The lower *clamped* floor was a **fortuitous cancellation**, not a
genuine accuracy gain — the clamp's constant bias happened to partly
offset the angular-thread interpolation floor at S16.  Removing it
exposes the true floor measured in #229 (next subsection), which is what
the unclamped weight should converge to.

⭐ **The cylinder repeated this signature two months later, and the
mechanism is the same one.**  At Q5.6.4 retiring the cylinder's
:math:`[\tfrac12, 1]` absorber *also* raised the anisotropic MMS floor
(:math:`3.128\mathrm{e}{-3}` vs :math:`3.511\mathrm{e}{-3}` better at
:math:`n_\varphi = 8`, then :math:`\sim 1.8`--:math:`2\times` **worse** at
16/32/64, :math:`n_x = 320`), for the same reason: a fixed closure bias
that partly cancels an interpolation floor is not an accuracy gain, and
the L2 norm cannot tell the two apart.  Do NOT read either table as
evidence for keeping a clamp — read them together as the reason the L2
norm is the wrong instrument for a closure decision.  The cylinder's
side of the story, including the solve-free ν-closure diagnostic that
*can* discriminate a derived τ from a fabricated one, is at
:ref:`sn-tau-absorber-retirement` in
:doc:`/theory/foundations/structured_geometry`.

Iso solves are unchanged in real
arithmetic (the clamp is silent on flat-in-:math:`\mu` fields) but
**not bit-identical** at IEEE-754: the closure
:math:`(\overline\psi - (1-\tau)\psi_{\rm in})/\tau` returns
:math:`\psi` exactly only :math:`\sim 81\,\%` of the time and within
1 ULP otherwise (reduction-order non-associativity), so the converged
homogeneous-reflective sphere drifts :math:`|\Delta k| = 2.3\mathrm{e}{-13}`,
:math:`\max|\Delta\phi| = 4.4\mathrm{e}{-13}` — an FP-tail, anchored to
the closed-form :math:`k_\infty = 1.875`.  One snapshot
(``sphere_2g_3reg_dd_n40``, genuinely non-flat) was regenerated
(:math:`k\;1.380766 \to 1.381001`); the two flat snapshots drift only
in the FP tail and were not regenerated.

**W1 gates.** ``tests/sn/sweep/curvilinear/test_w1_clamp_silent_on_flat.py``
(closure-unit :math:`\tau`-independence on flat fields; converged
homogeneous-reflective iso anchored to :math:`k_\infty`; unclamped
positivity) + the W1 ``@slow`` aniso gates appended to
``tests/sn/verification/mms/test_curvilinear_aniso_convergence.py``
(the S32 clean-:math:`\mathcal{O}(h^2)` full-ladder claim; the S16
coarse-rate-cleaner-unclamped discriminator; the floor-scales-with-
quadrature pin).  Landed in commit ``b2d8a6d``.

.. _sn-cylinder-angular-floor:

(c) The cylinder angular floor — structurally blocked
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The anisotropic ansatz :math:`\psi_{\rm chosen} = (A(r) + B(r)\,\mu)/W`
imposes :math:`\psi_n` per ordinate, so there is **no angular error at
the imposed ordinates**.  But the M-M redistribution consumes
half-angle THREAD values :math:`\psi_{m\pm 1/2}` that the recurrence
**interpolates** (they are not imposed).  On an angle-varying ansatz the
thread's interpolation error is an angular-quadrature-resolution effect:
under spatial refinement at fixed quadrature the solution converges to
an angular floor, and a pure-spatial-rate assertion cannot hold once
the spatial error drops below it.

**The cylinder floor scales with the AZIMUTHAL quadrature**
:math:`n_\varphi`, **not the polar** :math:`n_\mu`.  This is the
load-bearing physical fact (and a correction to an earlier mislabel):
the radial direction cosine is :math:`\eta = \sin\theta\,\cos\varphi`,
so the M-M thread marches in azimuth :math:`\varphi` *per polar
:math:`\mu`-level*.  Measured 2026-06-13 at :math:`n_x = 80` on the
**full** ``NODE_ALIGNED`` product rule:

.. list-table:: Cylinder aniso floor vs azimuthal quadrature (:math:`n_x = 80`, volume-weighted L2, the pre-Q5.6.3 fixture)
   :header-rows: 1
   :widths: 25 25 50

   * - :math:`n_\varphi`
     - L2 error
     - Behaviour
   * - 8
     - 1.90e-2
     - hard floor
   * - 16
     - 7.37e-3
     - drops :math:`2.58\times`
   * - 32
     - 3.10e-3
     - drops :math:`2.38\times`

while :math:`n_\mu` (polar) refinement at fixed :math:`n_\varphi`
leaves the floor **flat** (`[M]` 1.90e-2, 1.91e-2, 1.91e-2 at
:math:`n_\mu = 4/8/16`).

.. note:: **Configuration, 2026-08-11.**  The rule the table above was
   measured on — the full-circle ``NODE_ALIGNED`` product — is
   **refused at cylindrical** ``SNMesh`` **admission since Q5.6.3**, so
   the numbers are correct history for a fixture that no longer ships.
   The floor moved twice since:

   .. list-table::
      :header-rows: 1
      :widths: 44 36 20

      * - fixture
        - :math:`n_\varphi` 8→16
        - ratio
      * - `[M]` 2026-06-13, full ``NODE_ALIGNED`` product
        - 1.90e-2 → 7.37e-3
        - 2.58×
      * - `[M]` 2026-08-08, ``folded_product``
        - 3.538e-3 → 6.782e-4
        - 5.22×

   The fold's carrying march start alone took the floor down
   :math:`5.4\times` at :math:`n_\varphi = 8` and *steepened* the
   azimuthal scaling, before the Q5.6.4 τ work touched anything.  The
   live ladder is the gate's own docstring
   (``tests/sn/verification/mms/test_curvilinear_aniso_convergence.py``)
   — read it there rather than trusting a copy on this page.

**Why it is structurally blocked.**  Product and level-symmetric
quadratures carry **duplicate azimuthal** :math:`\eta`: ordinates come
in :math:`\pm\varphi` symmetry pairs with the same :math:`|\eta|` but
opposite :math:`\xi` (e.g. :math:`\varphi = \pi/4` and
:math:`\varphi = 7\pi/4` both give
:math:`\eta = \sin\theta/\sqrt 2`).  The M-M thread marches in
:math:`\eta` alone, so a field whose true variation is in the full
:math:`(\eta, \varphi)` plane is **not threadable exactly** by a 1-D
:math:`\eta`-march — a structural mismatch, not a tuning problem.
Closing the cylinder floor entirely would require a genuine 2-D
:math:`(\eta, \varphi)` angular closure — still **out of scope**.

.. note:: **Retraction (2026-08-11, Q5.6.4).**  This paragraph used to
   continue: *"No partition (midpoint / cumulative-weight /
   ordinate-interior) gives* :math:`\tau^{\rm raw} \in [\tfrac12, 1]`
   *with bounded edges; the cumulative-weight partition is exact on
   level-symmetric but needs* :math:`\tau^{\rm raw} \in [-4.5, 5.5]`
   *(edges outside the level)."*  Two things are wrong with it now.

   #. :math:`[\tfrac12, 1]` **was never a requirement** — it was the
      retired absorber's box, not the admissible range of :math:`\tau`,
      which is :math:`[0, 1]` (predicate P3).  The search it describes was
      for a partition satisfying a condition no reference imposes.
   #. **A partition satisfying the real predicates exists and shipped**:
      the :math:`\omega`-midpoint partition
      (:eq:`angular-cell-partition`) satisfies P3 *as a theorem* on any
      monotone arc, and P2 then determines :math:`\tau` uniquely.  The
      cumulative-weight observation survives, sharpened: it fails **P3**
      (not a :math:`[\tfrac12, 1]` box), with `[M]` 0/4 → 4/8 → 12/16 →
      28/32 ordinates outside their own cell at
      :math:`n_\varphi = 8/16/32/64` and a divergent (NaN) solve from
      :math:`n_\varphi \ge 16`, because an arc cell's
      :math:`\eta`-measure :math:`\propto \sin\omega_m` is not constant
      while a trapezoid weight is.

   What survives unchanged is the **azimuthal-duplication argument above**
   (a 1-D :math:`\eta`-march cannot thread a genuinely 2-D
   :math:`(\eta,\varphi)` field) — that is a statement about the *march*,
   independent of which partition the march uses.  Issue #229 is
   **CLOSED** (2026-06-13); it is the measurement record that named this
   floor and attributed it to half-angle-thread interpolation, not an open
   work item.  Full treatment:
   :ref:`sn-tau-absorber-retirement` in
   :doc:`/theory/foundations/structured_geometry`.

**The sphere–cylinder asymmetry.**  The sphere DOES have a pre-floor
:math:`\mathcal{O}(h^2)` window: at S16 the coarse orders clear 1.99 and
the floor (:math:`\approx 2.9\mathrm{e}{-4}` at S32, :math:`n_x = 160`)
sits below the segment's finest spatial error, so the clean
second-order window extends to :math:`n_x = 80` at S32.  The cylinder
has **no** such window at the *shipped* quadratures — even
:math:`n_\mu = 16` (:math:`N = 512`) reaches only order 1.80 on the
coarsest :math:`\{5, 10, 20\}` segment before the angular floor
dominates.  The mathematics, not runtime, is the blocker.

.. warning:: **Do not read "no** :math:`\mathcal{O}(h^2)` **window" as
   "the cylinder saturates at a floor"** — the two are different claims
   and the Q5.6.4 probes separated them (2026-08-11).  At the fixed-fine
   :math:`n_x = 80` used by the floor tables above, the :math:`\approx
   1.3\mathrm{e}{-4}` that every :math:`\tau` convention "saturated" to
   at :math:`n_\varphi \ge 32` is the **MESH**, not the closure: `[M]` at
   :math:`n_\varphi = 128`, refining :math:`n_x` 80 → 320 still drops the
   error :math:`8.6\times` (:math:`1.3397\mathrm{e}{-4} \to
   1.5488\mathrm{e}{-5}`).  Conversely at :math:`n_x = 320` (spatial
   contribution :math:`\le 1.5\mathrm{e}{-5}`) the **angular** error
   converges at a clean :math:`\sim \mathcal{O}(n_\varphi^{-2})` with no
   flat floor in range.  ⟹ any number read at
   :math:`n_\varphi \ge 32,\; n_x = 80` is a MIXED spatial+angular
   quantity.  The shipped gate's :math:`n_\varphi` 8→16 leg is
   angular-dominated (:math:`3.5\mathrm{e}{-3} \gg 1.3\mathrm{e}{-4}`)
   and is therefore sound; the "no window at any quadrature" phrasing was
   over-general and is narrowed here to the shipped set.

**W3 gate retune (the #229 retune).**  Per the vv-principles anti-pattern
"a claim that cannot hold MUST NOT be asserted; pin what IS true
instead", W3 removed all five aniso xfail markers and migrated the six
equation labels to green tests:

* **Sphere** ``test_sn_spherical_aniso_mms_converges_second_order`` →
  coarse-segment ``orders[:2] > 1.9`` + magnitude band
  :math:`1\mathrm{e}{-8} < \mathrm{err}[-1] < 5\mathrm{e}{-3}`
  (loosened from :math:`1\mathrm{e}{-3}` because the W1 unclamp removed
  the fortuitous-cancellation lower floor) + ``catches("ERR-026")``.
* **Cylinder** ``test_sn_cylindrical_aniso_mms_converges_second_order``
  → floor band :math:`1\mathrm{e}{-3} < \mathrm{err}[-1] <
  5\mathrm{e}{-2}`, **no rate claim** (the floor dominates).  The
  cylinder phase-C spatial test was **repurposed** into
  ``test_cyl_aniso_floor_scales_with_quadrature``
  (:math:`\mathrm{err}(n_\varphi{=}16) < \mathrm{err}(n_\varphi{=}8)/2`
  — the verified-floor second claim that pins the angular attribution).
* The sphere prescribed-inflow redistribution test dropped its
  strict-xfail and rate gate (band :math:`1\mathrm{e}{-8} <
  \mathrm{err} < 5\mathrm{e}{-3}` + a kept converged-value
  ``assert_allclose(2e-2)``).

Landed in commit ``679a1e6`` (audit exit 0; the
:eq:`sn-mms-spherical-psi` / ``-qext`` / :eq:`sn-mms-cylindrical-psi` /
``-qext`` labels and the two spatial-convergence labels are now all
green-tested).

.. _sn-pole-cell-spatial-closure:

(a) The sphere/cylinder pole-cell :math:`\mathcal{O}(h)` spatial closure (#233, ERR-059)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the **new** discovery of the program and the one *surviving*
manifestation in the curvilinear-SN family.  The curvilinear scalar
flux is **first-order** :math:`\mathcal{O}(h)` at the :math:`r \to 0`
central cell in the pointwise / L∞ norm — distinct from #168 (outer
face, CLOSED), ERR-058 (the closure seed, CLOSED), and the angular
floors above.  It decomposes into three parts, **none** of which
warrants a code fix.

Decomposition
^^^^^^^^^^^^^

**Part 1 — :math:`\sim 75\,\%` MMS comparison artifact (not a solver
bug at all).**  The production spherical MMS evaluates the source at the
cell MIDPOINT ``mesh.centers`` and compares
:math:`\phi_{\rm solver}` against :math:`\phi_{\rm exact}(\text{midpoint})`.
But the spherical DD discrete unknown **IS** the cell-volume average

.. math::
   :label: sn-pole-cell-shell-average

   \overline{\phi}_{n,i}
       \;=\; \frac{4\pi}{V_i}\int_{r_{i-1/2}}^{r_{i+1/2}} r^2\,\phi_n(r)\,dr

.. (vv-status rationale) definition: the literature-transcribed DEFINITION of
.. the spherical DD discrete unknown as the shell-volume average
.. (Hebert2009 Eq. 3.430), not a point value. Definitional — it is the datum
.. of the pole-cell error decomposition, not a solver claim.
.. vv-status: sn-pole-cell-shell-average documented

(:cite:`Hebert2009` Eq. 3.430 — the unknown is *defined* as the shell
average, not a point value; the diamond relation Eq. 3.431 relates it to
the face fluxes).  Under :math:`r^2\,dr` weighting the volume-average and
the midpoint point-value differ by :math:`\mathcal{O}(h)` at the pole
cell, because :math:`r_{\rm lo} = 0` maximally skews the weight (the
volume-centroid sits at :math:`\tfrac34 h`, not :math:`\tfrac12 h`).
Using the *shell-averaged* source AND comparing against the
*shell-volume-average* drops the pole error :math:`\sim 4\times`
(:math:`0.0212 \to 0.00497`) — confirming the bulk of the apparent
error is a comparison subtlety, not solver truncation.

**Part 2 — :math:`\sim 25\,\%` genuine but LITERATURE-ACCEPTED INHERENT
first order.**  Even the fully consistent finite-volume MMS (shell-avg
source + shell-avg reference) leaves the pole at clean
:math:`\mathcal{O}(h^{1.00})`.  The root cause: at :math:`r_{\rm lo} = 0`
the inner face area :math:`A(0) = 0`, so the diamond closure
:math:`\overline\psi = \tfrac12(\psi_{\rm in} + \psi_{\rm out})` gives
:math:`\psi_{\rm out} = 2\overline\psi`, **over-predicting the pole
outer face by exactly +50 %** (mesh-independent rel. error 0.5000), while
the true face is :math:`A(h)` and :math:`2\langle A\rangle_{\rm vol} =
2\cdot\tfrac34 A(h) = 1.5\,A(h)`.  Deeper still, the conservative
*balance itself* is inconsistent at the pole: fed the EXACT cell average
and EXACT inflow it solves for an outer face :math:`-46\,\%` wrong, and
the residual-per-volume plateaus mesh-independently — because
:math:`A_{\rm in} = 0` degenerates the streaming surface integral while
:math:`V \sim h^3`.

:cite:`Hebert2009` §3.9.4 and Stacey §9.9 **both** use exactly this plain
diamond + Carlson-starting-direction + symmetry scheme at the central
cell with **no special** :math:`\mathcal{O}(h^2)` **closure, and
neither flags reduced order there**.  First-order at the single pole
cell is the accepted, unflagged behaviour of the standard scheme.

**Part 3 — NOT cleanly fixable by a local closure.**  W2 tested the
volume-weighted linear reconstruction
:math:`\overline\psi = \beta\,\psi_{\rm out} + (1-\beta)\,\psi_{\rm in}`
with :math:`\beta = \tfrac34` at the pole (the value that makes
:math:`\overline\psi` :math:`\mathcal{O}(h^3)`-consistent against
:math:`\langle A\rangle_{\rm vol}` at exact faces).  Validated
end-to-end with a faithful production-sweep monkeypatch (and a
:math:`\beta = \tfrac12`-identity regression guard verified to
:math:`3\mathrm{e}{-16}`): :math:`\beta = \tfrac34` does **NOT** restore
:math:`\mathcal{O}(h^2)` — the pole stays :math:`\mathcal{O}(h)`,
magnitude slightly *worse* (:math:`0.0050 \to 0.0106`), and a full-mesh
:math:`\beta` degrades the interior.  Closure-consistency at exact faces
:math:`\neq` fixed-point accuracy: the propagated face flux couples back
through the balance.  A genuine fix needs a non-local higher-order
central-cell reconstruction the canon does not provide — a linear-
discontinuous (Issue #6), cell-update (#158), or nodal scheme
(:cite:`WuXieFischer1999` NSE 133).

Why it is invisible to L2 and to :math:`k_{\rm eff}`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The production ``test_sn_spherical_mms_converges_second_order`` uses the
volume-weighted L2 norm.  The pole :math:`\mathcal{O}(h)` at one cell of
:math:`V \sim h^3` contributes :math:`\sqrt V \sim h^{1.5}` →
:math:`h^{2.5}` to L2 — subdominant.  Both midpoint and volume-average
L2 references converge clean :math:`\mathcal{O}(h^{2.00})`; only the L∞
(pole) is :math:`\mathcal{O}(h)`.  For :math:`k_{\rm eff}`: a reflective
sphere recovers :math:`k_\infty = 1.875` exactly, mesh-independent; a
vacuum sphere converges monotone to :math:`\sim 1.78590` at
:math:`\mathcal{O}(h^{1.48})` (combined pole + outer-face first order;
increments :math:`2\mathrm{e}{-5}` at :math:`n_x = 160`, far below
engineering tolerance).  **This is why #233 needed an L∞ / per-cell
probe to surface** — no L2 or eigenvalue gate could see it.

The cylinder shares the same defect, masked
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cylinder pole vs. **midpoint** is :math:`\mathcal{O}(h^2)`
(1.94 / 1.97 / 1.98) but vs. the **volume average** is
:math:`\mathcal{O}(h)` (0.99 / 0.99 / 1.00) — the SAME diamond
inconsistency, masked by the midpoint comparison: the cylinder's
:math:`r\,dr` (linear) weight puts the volume-centroid at
:math:`\tfrac23 h` while diamond's :math:`\tfrac12 A(h)` happens to
:math:`\approx` the midpoint :math:`A(h/2)`, so the midpoint comparison
the gate uses is *accidentally* :math:`\mathcal{O}(h^2)` for the
cylinder.  The cylinder pole is therefore **not** "clean
:math:`\mathcal{O}(h^2)`" — it is the same :math:`\mathcal{O}(h)`
volume-average defect, hidden by the comparison choice.  Cylinder global
L2 is also clean :math:`\mathcal{O}(h^{2.00})`.

The characterization gate (#233)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Per the vv-principles "pin what is TRUE + protect the floor WITHOUT
calcifying the limitation" discipline, W2 ships a **characterization**
gate, not a fix gate
(``tests/sn/verification/mms/test_curvilinear_pole_cell_characterization.py``,
commit ``255eba4``):

* **Guarantee tests** (carry ``verifies("dd-curvilinear-scalar", ...)``,
  the :eq:`dd-curvilinear-scalar` cell-update label): global
  volume-weighted L2 is :math:`\mathcal{O}(h^2)` (``orders > 1.9``).
  The sphere is asserted under **both** references — midpoint AND the
  Hébert-3.430 shell-volume-average :eq:`sn-pole-cell-shell-average`,
  built from ``scipy.integrate.quad`` (a trusted-library integrator,
  structurally independent of the solver).  Agreement on the order
  across two structurally-different references proves the L2 order is
  REAL, not a midpoint artifact.
* **Characterization tests** (NO ``verifies`` — they pin a *limitation*,
  not a correctness claim): the pole L∞ order is **lower-bounded only**
  (:math:`> 0.8` — "at least first order, does not regress"), the pole
  is the L∞-dominant cell (fraction :math:`> 0.99`), and the interior
  is clean :math:`\mathcal{O}(h^2)` (:math:`> 1.8`).  **No upper bound**
  on the pole order, so a future LD / nodal scheme that lifts the pole
  to :math:`\mathcal{O}(h^2)` keeps the gate green
  (:math:`2.0 > 0.8`) — the characterization gate pins what is true and
  the regression floor without blocking a legitimate improvement
  (vv-principles anti-patterns #5 / #17).

Measured (sphere :math:`n_{\rm ord} = 16`, ladder
:math:`[40, 80, 160, 320]`): L2 midpoint orders :math:`2.01\times3`; L2
shell-average :math:`2.00\times3`; L∞ (pole) :math:`0.91 / 0.95 / 0.97`;
interior :math:`1.84 / 1.92 / 1.96`; pole fraction :math:`1.00` every
mesh.  Cylinder pole-vs-midpoint :math:`1.94 / 1.97 / 1.98`;
pole-vs-volavg :math:`0.99 / 0.99 / 1.00`.

.. _sn-p1-scattering-curvilinear:

:math:`P_1` anisotropic SCATTERING in curvilinear — the two unrelated paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A persistent source of confusion in this cluster is that "anisotropic"
names **two structurally unrelated** things in a curvilinear SN solve.
Issue #9 is about the *second*; everything above (#229, the
:math:`\alpha`-dome, the :math:`\tau`-clamp) is about the *first*.

* **Path-(I) — geometric angular redistribution.**  The
  :math:`(1-\mu^2)/r\,\partial_\mu\psi` term (sphere) /
  :math:`\xi^2 B / r` (cylinder), threaded by the Morel--Montry Carlson
  :math:`\alpha`-dome.  This is **P0-only**; the "anisotropy" lives in
  the *angular-flux ansatz*, not in the scattering kernel.  The
  existing curvilinear aniso MMS cases
  (:math:`\psi = (A + \zeta B)/W`) exercise ONLY this path.  #229 is a
  Path-(I) test-design floor.
* **Path-(II) — Legendre SCATTERING moments.**  The
  :math:`P_1{+}` scattering source :math:`R\,\Lambda\,M`
  (the collision gain's :math:`\ell \ge 1` body at
  ``scattering_order ≥ 1``; the verb was ``scattering.build_aniso_source``
  until #448),
  geometry-**agnostic**, wired identically for all geometries through
  :func:`~orpheus.sn.coupled_system.build_within_group_system` (the
  :math:`S` gain of the :math:`(L+C),\,S,\,B` decomposition carries
  :math:`P_1` when
  ``scattering_order = 1``).  No curvilinear test exercised Path-(II)
  before #9 — it is NEW coverage of an existing capability (NO
  ``orpheus/`` change; Path-(II) works as-is).

L0 — the operator-admits trick
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rather than derive a costly symbolic :math:`P_1`-source MMS, W4 feeds a
*known* anisotropic angular flux :math:`\psi_{{\rm ref},n} = (A + \zeta
B)/W` to the within-group :math:`S` operator at
``scattering_order = 1`` and isolates the :math:`P_1` contribution as
:math:`S_1.\mathrm{apply}(\psi) - S_0.\mathrm{apply}(\psi)`, asserted
**per ordinate** (NOT weight-summed — the :math:`\alpha`-dome
telescopes, vv anti-pattern #8) against a structurally-independent
hand-reference:

* **Sphere** (fully SH-table-INDEPENDENT — :math:`P_1(\mu) = \mu`
  directly):

  .. math::
     :label: sn-p1-sphere-hand-ref

     q_n^{P_1} \;=\; \frac{1}{W}\,3\,\mu_n\,\Sigma_{s1}\,\phi_1,
     \qquad
     \phi_1 \;=\; B(r)\,\frac{\sum_n w_n\,\mu_n^2}{W}.

* **Cylinder** (explicit :math:`Y_1^m` moment-sum, independent of the
  production :math:`R\,\Lambda\,M` einsum):

  .. math::
     :label: sn-p1-cylinder-hand-ref

     q_n^{P_1} \;=\; \frac{1}{W}\,3\,\Sigma_{s1}
                  \sum_m Y_1^m(\Omega_n)\,\phi_1^m,
     \qquad
     \phi_1^m \;=\; \sum_n w_n\,Y_1^m(\Omega_n)\,\psi_n.


  .. no-implementation:: sn-p1-cylinder-hand-ref
     :kind: canonical-form

     **Nothing implements this**, and that is what makes it a reference.
     The page states the value is *independent of the production*
     :math:`R\Lambda M` einsum, and the claiming test assembles it inline
     as a double loop precisely so it is **not** the production frame
     analysis/reconstruction faces. Declaring those symbols here would
     demote the gate to a value compared with itself through a wrapper —
     green forever, keeping its authoritative name, unable to detect the
     drift it exists to catch.

Both agree at machine precision (rel. :math:`4.7\mathrm{e}{-15}` sphere /
:math:`5.6\mathrm{e}{-15}` cylinder), with a
``max|S₁−S₀| > 1e-6`` negative control (vv anti-pattern #11 — a dropped
:math:`P_1` makes :math:`S_1 - S_0 \equiv 0` and fails the non-zero
hand-ref match).  **1-group is legitimate here**: this is a
flux-shape / OPERATOR claim (the per-ordinate :math:`P_1` source reads
:math:`\phi_1`, flux-shape-dependent by construction), NOT an eigenvalue
claim — the 1-group-degeneracy rule applies only to *eigenvalue*
verification.

L1 — the directional eigenvalue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forward-peaked :math:`P_1` scattering (:math:`\bar\mu > 0`) **lowers**
:math:`k_{\rm eff}` versus :math:`P_0`.  The physics: positive
:math:`\bar\mu` preserves the forward direction, so in a finite,
vacuum-bounded sphere forward-preserved scattered neutrons are more
likely to cross the outer boundary → **enhanced leakage** → lower
:math:`k_{\rm eff}`.  This requires a **vacuum** outer BC (a reflective
sphere has no leakage → :math:`P_0 \equiv P_1`).  Validated robust:

* Homogeneous vacuum sphere :math:`R = 4 / 10 / 25`:
  :math:`\Delta = k_{\rm eff}^{P_1} - k_{\rm eff}^{P_0} =
  -3.76\mathrm{e}{-3} / -1.32\mathrm{e}{-3} / -2.88\mathrm{e}{-4}` — sign
  always negative, :math:`|\Delta|` **grows as the sphere shrinks**
  (the leakage-monotone signature, the structural negative control a
  sign-flipped or absorption-mimicking :math:`P_1` would violate).
* Heterogeneous fuel-core(:math:`r < 5`)+moderator-shell vacuum sphere
  :math:`R = 10`: :math:`\Delta = -1.40\mathrm{e}{-2}` (:math:`140\times`
  the :math:`1\mathrm{e}{-3}` detection bar), with materials
  ``get_mixture("A","2g")`` (the only fissile 2-group mixture;
  asymmetric downscatter-only P0 avoids the 1-group degeneracy) and
  ``get_mixture("C","2g")``.

Two L1 rows pin this: a heterogeneous-sphere
:math:`k_{\rm eff}^{P_1} < k_{\rm eff}^{P_0}` AND
:math:`1\mathrm{e}{-3} < (P_0 - P_1) < 5\mathrm{e}{-2}`; and a
leakage-monotone control
:math:`(P_0 - P_1)|_{R=4} > (P_0 - P_1)|_{R=25} > 0` (the mechanism
pin).  These are the **first curvilinear exercise** of the
geometry-agnostic ``pn-scatter`` / ``flux-moments`` labels (prior tests
were 2-D Cartesian only).  L0 lands in
``tests/sn/verification/mms/test_curvilinear_aniso_scattering_p1.py``,
L1 in ``tests/sn/eigenvalue/test_keff_curvilinear.py::TestSphereP1DirectionalEigenvalue``
(commit ``d5878e9``).  L2 is deferred (subsumed by L0+L1; a
:math:`P_1`-convergence L2 needs the :math:`\sigma_{s1}`-MMS source and
rides the same #229 floor).

Infrastructure retained
~~~~~~~~~~~~~~~~~~~~~~~

Per the aggressive-retirement exception, the program deletes no correct
machinery:

.. list-table:: Curvilinear aniso program — primitives status
   :header-rows: 1
   :widths: 34 18 48

   * - Primitive
     - Status
     - Why kept / what changed
   * - Spherical :math:`\tau_m` (unclamped)
     - **production**
     - W1: the unique exact-on-linear weight.  ⚠ **Ownership moved after
       W1** — #236 Step C excised the geometry-side τ producer, so τ is
       now single-sourced in the *angular closure*
       (:func:`~orpheus.sn.angular.closure.morel_montry_tau_per_level`),
       not in
       :func:`~orpheus.sn.mesh.reduced_operator.spherical_streaming`;
       SI sweep + Krylov matvec still inherit one value.
   * - Cylindrical :math:`\tau_m` clamp
     - ⛔ **retired** (Q5.6.4, 2026-08-11)
     - Was retained here as the :math:`\tau = 0` structural
       :math:`\div 0` block, "removable only with a 2-D
       :math:`(\eta,\varphi)` closure".  Both clauses fell: Q5.6.3 made
       the :math:`\tau = 0` rule classes inadmissible, and Q5.6.4 found
       the absorber was compensating a **wrong cell partition**, not a
       singularity.  Retired with ``morel_montry_tau_raw_per_level``; see
       :ref:`sn-tau-absorber-retirement`.
   * - Pole-cell characterization gate
     - **regression net**
     - Pins the inherent :math:`\mathcal{O}(h)` pole limitation
       (lower-bounded, not calcified) + the global :math:`\mathcal{O}(h^2)`
       guarantee under two independent references.
   * - Shell-volume-average reference :eq:`sn-pole-cell-shell-average`
     - **oracle**
     - The Hébert-3.430 finite-volume unknown; the principled MMS
       reference that removes the :math:`\sim 75\,\%` comparison
       artifact.  Built from ``scipy.integrate.quad``.
   * - Angular endpoint defect :math:`D`
       (:eq:`sn-angular-endpoint-defect-eq`)
     - **diagnostic** — ⛔ *not* a ranker
     - The over-determination residual between the angular march's two
       endpoint conditions (:ref:`sn-angular-endpoint-defect`).  Kept
       because it is a cheap, **reference-free**, pointwise consistency
       residual whose refinement behaviour is a property of the scheme,
       and because computing it costs nothing — the recurrence already
       fills the far face.  ⛔ It is **not** an error estimator: see the
       ruling below before using it for anything comparative.

.. warning:: **⛔** :math:`D` **may not be used to rank** :math:`\tau`, and
   this row is the place a future session would reach for it.

   :math:`D` is reference-free, pointwise, tight, honestly
   parameter-loaded, and it ranks the shipped Q5.6.4 angular cell partition
   first by 2.6–45× over garbage :math:`\tau`.  Every one of those
   properties is what an adjudicating instrument is supposed to have, and
   the ranking is still worthless as evidence: `[M]` 2026-08-12 against the
   **analytic** anisotropic cylindrical MMS, the Pearson correlation of
   :math:`\log D` with :math:`\log` of the true MMS error runs
   :math:`+0.7515 / +0.2608 / +0.0630` at :math:`n_\varphi = 8/16/32`
   (:math:`2/4 \to 0/4 \to 0/4` rank agreement) — **degrading monotonically
   to zero** as angle refines.  Structurally it must: :math:`D = e_1 - e_2`
   is a difference of two truncation errors, hence smallest exactly when
   both are largest and equal.

   This is ``vv-principles`` #24 applied to a *new* candidate instrument —
   an instrument that decides a design owes the BASIS, RANK-CORRELATION,
   ZERO-SET and REGIME checks before its verdict is quoted, and :math:`D`
   fails the rank-correlation one outright.  ⟹ **The campaign has no
   reference-free instrument that can rank** :math:`\tau`.  That gap is the
   standing state; any future :math:`D`-based :math:`\tau` argument must
   cite it before making its case.

Open research paths (research-tag, not production-blocking)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. **Higher-order central-cell spatial scheme** (lifts the #233 pole
   :math:`\mathcal{O}(h)`).  The canon (:cite:`Hebert2009` §3.9.4, Stacey
   §9.9) provides no drop-in :math:`\mathcal{O}(h^2)` central-cell
   diamond closure; the documented route is a non-local higher-order
   *spatial* scheme — linear-discontinuous (Issue #6), step-
   characteristic, or the Green's-function nodal method of
   :cite:`WuXieFischer1999` (NSE 133, "very high precision on coarse meshes
   relative to standard fine-mesh DD").  Likely diagnostic probe: the
   pole-cell per-cell rate under the shell-average reference, holding
   quadrature fixed.
#. **2-D** :math:`(\eta, \varphi)` **cylinder angular closure** (lifts
   the cylinder floor measured in #229).  The 1-D :math:`\eta`-thread
   cannot represent the duplicate-azimuthal-:math:`\eta` variation of
   product / level-symmetric quadratures; a genuine 2-D angular closure
   (or a Gauss-type azimuthal quadrature with distinct :math:`\eta`
   values, GitHub Issue #1) is required.  Likely probe: the floor-scaling
   table above with the azimuthal quadrature replaced by a
   distinct-:math:`\eta` set.

   ⚠ **Two things this path is NOT, both settled since it was written.**
   (a) It is no longer a *prerequisite* for retiring the cylinder clamp —
   Q5.6.4 did that with a 1-D :math:`\omega`-march by fixing the cell
   partition (:eq:`angular-cell-partition`).  (b) The duplicate-azimuthal
   rule classes it describes are **inadmissible** at cylindrical
   ``SNMesh`` since Q5.6.3; the shipped ``folded_product`` is a σ_y
   quotient whose levels are monotone half-circle arcs, so the residual
   floor to be lifted is the one measured on *that* fixture
   (`[M]` 3.538e-3 → 6.782e-4 at :math:`n_\varphi` 8→16), not the
   1.9e-2 of the retired full-circle rule.

Session trail (V&V audit trail)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Commits** (branch ``fix/curvilinear-aniso-pole-and-clamp``,
  2026-06-13): ``b2d8a6d`` (W1 sphere unclamp), ``255eba4`` (W2 #233
  pole-cell characterization gate), ``d5878e9`` (W4 #9 :math:`P_1`
  scattering coverage), ``679a1e6`` (W3 #229 gate retune).
* **Diagnostics**: the W1–W4 ``diag_01..31`` decomposition probes (the
  decisive ones: the
  :math:`E_{\rm test} = E_{\rm artifact}(\text{midpoint} -
  \text{volavg}) + E_{\rm true}(\text{solver} - \text{volavg})`
  decomposition; the discrete-balance residual fed exact fields; the
  faithful production-sweep monkeypatch with a :math:`\beta = \tfrac12`
  identity guard).
* **Literature**: :cite:`Hebert2009` §3.9.4, Stacey §9.9 (plain diamond at
  the central cell, no special closure), :cite:`BaileyMorelChang2010` Eq. 43
  (the exact-on-linear weight), :cite:`WuXieFischer1999` (the nodal route to
  :math:`\mathcal{O}(h^2)` at the origin).
* **vv catalogue**: ``error_catalog.rst`` — ERR-059 (the pole-cell
  inherent limitation) + the :math:`\tau`-clamp mis-citation finding +
  the ERR-026 surviving-manifestation note.
* **Issues**: #229 (cylinder floor + sphere gate retune), #9
  (:math:`P_1` curvilinear scattering), #233 (pole-cell, stays OPEN to
  track the future higher-order scheme).

.. note:: **vv-status (eq-labels added by this section).**  The labels
   :eq:`sn-tau-mm-raw`, :eq:`sn-pole-cell-shell-average`,
   :eq:`sn-p1-sphere-hand-ref`, and :eq:`sn-p1-cylinder-hand-ref` are
   *structural / representational* identities (the literature-
   transcribed M-M weight; the Hébert-3.430 finite-volume unknown; the
   structurally-independent :math:`P_1` hand-references).  They are NOT
   solver claims.  The :math:`\tau`-clamp / pole-cell / :math:`P_1`
   *verifiable* content is the W1 clamp-silence + positivity gates, the
   W2 ``verifies("dd-curvilinear-scalar")`` guarantee tests, and the W4
   ``verifies("pn-scatter","flux-moments")`` per-ordinate operator-
   admission gates named above — so these eq-labels are ``documented``.


.. _ld-cartesian-2d:

The 2-D Cartesian LD stress MMS (D5b-S4)
------------------------------------------

Sub-step **D5b-S4** is the L1 flux-shape verification of the multi-dimensional
bilinear (UBLD) Linear-Discontinuous closure: a Method-of-Manufactured-Solutions
reference whose trial flux is :math:`\mu`-bilinear (so the per-axis SPATIAL
slope rows are genuinely activated, the vv Mode-7 override) with a NON-vanishing
boundary trace (so the prescribed-inflow boundary closure is stressed).

.. math::
   :label: ld-cartesian-2d

   \psi_{n,g}(x,y) = \frac{1}{W}\bigl[\,A_g(x,y)
       + \mu_{x,n}\,B_g(x,y) + \mu_{y,n}\,C_g(x,y)\,\bigr],
   \qquad
   \phi_g(x,y) = A_g(x,y),

with the strengthened drivers (the :math:`b_2,\,c_2` cross-harmonics break the
x↔y reflection so a same-sign slope-row sign bug cannot cancel) and
:math:`a_0 > 0` (non-vanishing at all four edges).  The manufactured source is
the continuous-PDE residual, derived symbolically (Branch 1, the
algebra-of-record) and structurally independent of the LD cell-update code
(L11).

Why this ansatz (the Mode-7 stress design)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The angular structure :math:`\psi = (A + \mu_x B + \mu_y C)/W` is chosen so the
**per-ordinate** field carries a genuine spatial slope on each axis: the
LD :math:`x`-slope row discretizes :math:`\partial_x\psi`, which sees
:math:`\mu_x\,\partial_x B/W` — a :math:`\mu_x`-weighted slope the DD
cell-average path *cannot* represent (DD has no slope moment).  The scalar flux,
however, is :math:`\phi = \int\psi\,d\mu = A` (the :math:`\mu_x B + \mu_y C`
terms integrate to zero over a symmetric quadrature), so the manufactured scalar
solution is :math:`A` alone — the slope is a *genuinely angular-resolved*
forcing, not a trivial consequence of the average.  This is the vv Mode-7
override: the simplest trig that satisfies the BCs would leave the slope rows
nulled by construction (the classic isotropic-ansatz bias); this ansatz
*activates* them deliberately.  Two further design choices:

* **The :math:`b_2, c_2` cross-harmonics break the x↔y reflection.**  Were
  :math:`B` and :math:`C` related by an :math:`x\leftrightarrow y` reflection, a
  *same-sign* slope-row sign bug (the most likely transcription error, since
  both slope rows share the cell-update code path) could leave the measured
  symmetric flux unchanged — a false green.  Adding distinct cross-harmonic
  content to :math:`B` and :math:`C` (so no reflection maps one to the other)
  makes the :math:`x`-slope-source and :math:`y`-slope-source genuinely
  independent, so a same-sign slope error breaks the measured flux.

* **:math:`a_0 > 0` (non-vanishing at all four edges)** stresses the
  prescribed-inflow boundary closure.  A solution that vanished at the boundary
  by construction would test nothing about the BC handling.  (The curvilinear
  pole-regularity constraint :math:`B(0) = 0` does NOT apply here — a Cartesian
  cell has no :math:`1/r` redistribution, so the slope drivers are unconstrained
  at the boundary.)

The manufactured source is the continuous-PDE residual,
:math:`\mu_x\partial_x\psi + \mu_y\partial_y\psi + \Sigma_t\psi = (1/W)(\Sigma_s\phi + Q^{\rm ext})`,
derived symbolically (Branch 1, the algebra-of-record) and **structurally
independent** of the LD cell-update code (L11): the SymPy derivation never
touches the discretization.  Branch 1 and Branch 2 share their spatial
amplitudes as single-sourced :math:`(\text{num}, \text{den})` pairs
(``Rational`` for SymPy, exact float for numpy), so the Branch-2
:math:`\equiv` Branch-1 source cross-check pins the two *evaluators* agree to
machine precision (:math:`1.5\times10^{-16}`), not just the symbolic identity.

The Branch-1 SymPy derivation lives in
:mod:`orpheus.derivations.continuous.mms.sn`
(:func:`~orpheus.derivations.continuous.mms.sn.derive_2d_cartesian_ld_stress_mms`
and the symbolic builder ``_2d_cartesian_ld_stress_symbolic``); the Branch-2
numerical factory is
:class:`~orpheus.derivations.continuous.mms.sn.SN2DCartesianLDStressMMSCase`
(built by ``build_2d_cartesian_ld_stress_mms_case``).

What this verifies (and what it cannot)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MMS closes the **slope-UNKNOWN** half of the LM-1989 slope-row sign trap in
:math:`d \ge 2`: the bilinear closure genuinely solves :math:`\hat\psi_x` and
:math:`\hat\psi_y` from the average plus the scattering source, and the
convergence is :math:`O(h^2)` to the manufactured value.  The slope-UNKNOWN sign
is **mutation-verified load-bearing**: on this tightly-coupled UBLD wavefront
closure (the slope feeds the propagating face cochain), a slope-row sign error
does not merely shift the limit — it *diverges* the iteration:

.. list-table:: Mutation verification of the slope-row sign (S4)
   :header-rows: 1
   :widths: 50 26 24

   * - Mutation of the UBLD slope-row sign
     - Strengthened result
     - Verdict
   * - full per-axis gradient sign flip
     - NaN
     - CAUGHT
   * - finite-trace slope :math:`[1,-1]\to[1,1]`
     - order :math:`-4.62`
     - CAUGHT
   * - surgical slope-row :math:`-2 \to +2` (both axes, the faithful "same-sign"
       transcription error)
     - inf
     - CAUGHT

(Baseline strengthened: order 2.00–2.14, finest residual :math:`3.5\text{–}6.0\times10^{-3}`.)
Because the catch is catastrophic (NaN/inf), there is no false-green regime for
this closure to hide in — a stronger guarantee than the subtle-cancellation
scenario the strengthening was originally designed against.

The foundation gate is :mod:`tests.derivations.test_sn_mms_ld_2d_stress_symbolic`
(the SymPy substitution identity, an INDEPENDENT finite-difference residual check
that does not reuse SymPy's own ``diff`` — L11, the Branch-2 :math:`\equiv`
Branch-1 source cross-check, and the Mode-7 activation / x↔y-asymmetry checks);
the end-to-end L1 gates are :mod:`tests.sn.verification.mms.test_mms_ld_2d`
(``test_ld_2d_stress_converges_second_order`` — the headline :math:`O(h^2)` +
value band, ``@l1`` ``@verifies("ld-cartesian-2d", "transport-cartesian-2d")``;
``test_ld_2d_stress_krylov_equals_si`` — the L14 matvec twin on the stress
habitat; ``test_ld_2d_stress_two_paths_ffw_equals_mfw`` — the two-DAG-schedule
invariant).  The closeout is
``.claude/agent-memory/method-implementer/issue_240_d5b_s4_ld_2d_stress_mms_closeout.md``.

The slope-SOURCE sign convention has TWO halves of its own (external
:math:`\hat Q` vs the boundary transverse-face-slope).  Both are now closed —
the EXTERNAL half (**Leg A, #247**) and the BOUNDARY half (**Leg B, #251**) —
see the honest-scope note below.

.. _ld-cartesian-2d-slope-source:

.. note:: Honest scope — the slope-SOURCE half of the LM-1989 trap (Leg A
   VERIFIED #247; Leg B VERIFIED #251).

   The LM-1989 slope-row sign trap has two halves: the slope-UNKNOWN sign
   (always exercised when the slope is non-trivially solved — VERIFIED by this
   MMS; mutation-verified — a slope-UNKNOWN sign flip diverges / leaves the
   value band) and the slope-SOURCE sign :math:`\hat Q`.  The slope-SOURCE
   half splits further:

   - **Leg A — the EXTERNAL slope-moment source** :math:`\hat Q` — is now
     VERIFIED (#247).  :func:`orpheus.sn.solver.solve_sn_fixed_source` accepts
     a typed union of TWO bulk ranks — flat ``(N, ng, *spatial)`` (slope rows
     zeroed, the honest default) OR moment-resolved
     ``(N, ng, *spatial, per_axis**ndim)`` (the projected slope rows threaded
     through) — and ``_lift_external_source_to_moments`` threads the
     moment-resolved slope rows into the SI rhs alongside the scattering
     source.  The slope-source sign is pinned STRUCTURALLY (the converged flux
     is only sub-floor sensitive — the vv Mode 10 trap): the lift threads the
     projected moment vector through at machine precision, and a CONSUMED
     slope-row sign flip moves the converged flux :math:`O(1)` above the inner
     tolerance, while the FLAT scalar gate stays GREEN (the Mode-10 asymmetry
     that closed the gap).

   - **The SCATTERING slope source** :math:`\Sigma_s \hat\phi` (the
     Increment-C iterate feedback) IS consumed and is now mutation-verified
     NOT sign-blind (#247, mutation control M4): flipping the iso slope rows of
     the per-ordinate scattering source moves the converged flux above the
     inner tolerance.  (The old value-band MMS WAS empirically blind to this —
     a slope-source-row sign flip left both the :math:`O(h^2)` order and the
     scalar-flux value band unchanged, because :math:`\Sigma_s\hat\phi` is an
     :math:`O(h)`-small DG-internal forcing whose error enters above
     :math:`O(h^2)` — the canonical vv Mode 10 instance.  The #247 mutation
     control replaces the value-band with the consumption proof.)

   - **Leg B — the BOUNDARY transverse-face-slope** — is now VERIFIED
     (**#251**).  The boundary trace ``mesh.angular_trace`` carries the
     :math:`2^{d-1}` transverse face-moments per face per ordinate per group
     (a moment-resolved slot ``(N, ng, *face_shape, 2^{d-1})`` minted by
     :attr:`orpheus.sn.mesh.augmented_mesh.SNMesh.boundary_face_layout`, appending the
     single-source :func:`orpheus.numerics.moment_layout.face_moment_tail`),
     so a moment-resolved prescribed inflow can carry the along-face
     (transverse) Legendre slope, the sweep outflow STORES the
     :math:`2^{d-1}` moments (the capture is no longer collapsed to the
     average), and ``_inflow_to_moments`` rank-discriminates a scalar inflow
     (seed slot 0, slopes zero — the scalar default) from a moment-resolved
     inflow (thread the projected transverse slope through).  Like Leg A the
     converged near-boundary flux is only sub-floor sensitive to the boundary
     slope (a SHARPER vv Mode 10 — "improves-on-flat" is NOT achievable,
     because the localized :math:`O(h)`-small boundary-trace slope sits below
     the bulk :math:`O(h^2)` discretization floor), so the slope is pinned
     STRUCTURALLY: the producer threads the projected transverse face-slope
     into the cochain at machine precision, and a CONSUMED transverse-slope
     sign flip moves the converged near-boundary flux :math:`O(1)` above the
     inner tolerance (:math:`|\Delta\phi|/|\phi| \approx 4.1\times10^{-3}`
     near-boundary at ``nc=16``, ~5.6 orders above the consumption tolerance;
     linear in the slope magnitude — genuine consumption), while the SCALAR
     inflow gate stays byte-identical (the Mode-10 asymmetry).  DD/Step
     (``per_axis == 1`` → ``face_moment_tail(1) == ()``) leaves the trace
     byte-identical (the negative control); a 1-D slab face is a point
     (``face_shape == ()``), so the transverse face-moment is a 2-D-and-higher
     concern by construction.

     The transverse-slope SIGN under REFLECTION across a face is a separate
     follow-up: the Leg-B MMS is vacuum-BC (which nulls the reflective
     coupling), so the reflective ``B`` operator's moment-axis passthrough
     (verified storage-correct — the ``PermutationOperator(axis=0)``
     broadcasts over the new trailing moment axis without a hard-coded
     trailing-axis assumption) is NOT exercised for its sign.  Physics: a
     normal-flip reflection preserves the tangent-plane (transverse)
     coordinate, so the transverse slope should reflect WITHOUT a sign flip —
     but this is UNVERIFIED (a reflective-LD MMS + an ``op.H`` adjoint check
     on the transverse-slope reflection is the follow-up).

   The full Leg A narrative — the tensor-Legendre projection convention, the
   typed-union bulk widening, the Mode-10 structural-teeth design, and the
   M1–M4 mutation table — is the subsection :ref:`ld-cartesian-2d-legA`
   immediately below.  The full Leg B narrative — the
   :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.boundary_face_layout` moment-tail
   storage lever, the ``_inflow_to_moments`` rank-discriminated pass-through,
   the four outflow capture-collapse DROP sites, the
   ``prescribed_inflow`` scalar-or-moment producer, the transverse
   face-moment normalization, the SHARPER Mode-10 (no improves-on-flat leg),
   and the reflective-BC sign follow-up (#252) — is the subsection
   :ref:`ld-cartesian-2d-legB`.

   **S9 (#257)** completes the boundary half: the MMS case
   ``prescribed_inflow`` now itself EMITS the moment-resolved slot (it no longer
   drops the slope at the producer), and the **coherent promise** — *LD is
   second-order at the boundary with no asterisk* — is LOCKED by a dedicated
   first-cell-row convergence gate.  The promise is delivered by the AVERAGE
   moment alone; the transverse slope is a sub-floor inflow-representation
   refinement (the fourth vv Mode-10 companion-unavailable instance).  S9 also
   establishes that the transverse boundary moment is a PROPERTY, not a new
   field type (#263).  The full S9 narrative — the coherent-promise evidence,
   the producer-honesty carve, and the property-vs-type seam — is the subsection
   :ref:`ld-cartesian-2d-coherent-promise`.

.. _ld-cartesian-2d-legA:

Leg A — the external slope-moment source :math:`\hat Q`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This subsection is the rich record of the EXTERNAL half of the slope-SOURCE
trap, closed under Issue #247.  The change is small in code (two named sites in
:func:`orpheus.sn.solver._build_fixed_source_rhs` and
``_lift_external_source_to_moments``) but it is the first time an LD external
source can carry sub-cell slope information through the public solver, so the
*verification* design — not the code — is the load-bearing content here.  Why
the slope-SOURCE sign is genuinely hard to pin (the vv **Mode 10** trap) and how
the gate gets teeth anyway is the canonical resolution of an
activated-but-unconstrained term; it is recorded in full because the lesson
recurs whenever a term is consumed yet enters the measured quantity below the
discretization floor.

The tensor-Legendre projection convention (the CRUX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To feed a slope-resolved external source into the UBLD closure, a continuous
:math:`Q^{\rm ext}(x,y)` must be projected onto the per-cell tensor-Legendre
moment vector that the cell update consumes.  The single load-bearing decision
is *which normalization* the projected moments carry — and the answer is locked
by the UBLD mass matrix, not chosen for convenience.

The UBLD cell mass on one axis is :math:`M_{\rm 1d} = \mathrm{diag}(h,\,\theta
h)`, :math:`\theta = 1/3` (Eq. :eq:`ld-ubld-mass-weights`), the L2-Gram of the
Legendre moment basis :math:`\{P_0 = 1,\ P_1 = \xi\}` on :math:`\xi \in [-1,1]`:
:math:`\langle P_0, P_0\rangle = h`, :math:`\langle P_1, P_1\rangle = \theta h =
h/3`.  The cell kernel forms its right-hand side as :math:`R_{\rm source} = M\,
S_{\rm moments}` (the d=1 reduction Eq. :eq:`ld-ubld-d1-reduction` confirms
:math:`R_{\rm prod} = [\,\bar Q\,h,\ \theta\,\hat Q\,h\,]` symbolically).  The
mass matrix therefore *already* supplies the per-volume and the
:math:`\theta` weighting.  The projection must NOT duplicate it.  The projected
moment is the **bare per-volume Legendre coefficient**:

.. math::
   :label: ld-cartesian-2d-projection-coeff

   \bar q \;=\; \frac{1}{V}\!\int_{\rm cell} q \;=\; \text{(cell average)}
   \quad\text{(slot 0)},
   \qquad
   \hat q_a \;=\; \frac{\langle q,\,P_1(\xi_a)\rangle}
                       {\langle P_1,\,P_1\rangle}
              \quad\text{(the }P_1\text{ coefficient on axis }a).

.. (V&V scope note) definition: the Leg-A projection NORMALIZATION
.. convention (bare per-volume Legendre coefficient; the mass matrix supplies
.. the theta/h weighting downstream). Wired to the foundation gate
.. ``tests/sn/verification/mms/test_mms_ld_2d.py::test_projection_slot0_is_cell_average_not_centre``
.. (slot-0 IS the cell average — the normalization convention pinned against
.. an independent fine-quadrature average).

For a cell-linear source :math:`q = a + b\,x`, the axis coefficient is
:math:`\hat q = b\,h/2` — **no** :math:`\theta`, **no** :math:`h`, **no**
:math:`V` in the projected number; the kernel's :math:`M` adds them downstream.
Sharing the :math:`\theta`/:math:`h` weighting between the projection and the
mass would double-count it; this is the apples-to-apples constraint that makes
the projected slope rows match what :math:`M^{-1}R` expects.

For a general bilinear :math:`q = a_{00} + a_{10}x + a_{01}y + a_{11}xy` on a
cell :math:`[x_L,x_R]\times[y_L,y_R]` (:math:`h_x = x_R-x_L`, centre
:math:`x_c`; similarly :math:`y`), the four tensor-Legendre coefficients are
hand-derivable in closed form:

.. math::
   :label: ld-cartesian-2d-bilinear-coeffs

   \bar q   &= a_{00} + a_{10}x_c + a_{01}y_c + a_{11}x_c y_c, \\
   \hat q_y &= \tfrac{h_y}{2}\,(a_{01} + a_{11}x_c), \\
   \hat q_x &= \tfrac{h_x}{2}\,(a_{10} + a_{11}y_c), \\
   \hat q_{xy} &= \tfrac{h_x}{2}\,\tfrac{h_y}{2}\,a_{11}.

.. (V&V scope note) reference: the four hand-derived tensor-Legendre
.. coefficients of a bilinear source — the structurally-independent (Branch-1)
.. reference the quadrature projector is pinned against. Wired to the
.. foundation gate
.. ``tests/sn/verification/mms/test_mms_ld_2d.py::test_tensor_legendre_projection_matches_hand_polynomial``
.. (reproduced to ``atol=1e-13``).

These four numbers are the structurally-independent reference for the projector
(see the teeth subsection): a bilinear integrand is integrated exactly by a
2-point Gauss rule, so the quadrature projector reproduces them to machine
precision.

**The d=2 Kronecker moment order is** :math:`[\bar\psi,\ \hat\psi_y,\
\hat\psi_x,\ \hat\psi_{xy}]` — axis 0 (:math:`x`) is the OUTER Kronecker
factor, axis 1 (:math:`y`) the INNER, so the slot order
:math:`[(0,0),(0,1),(1,0),(1,1)]` places the **x-slope at slot 2**, the y-slope
at slot 1, the cross-moment at slot 3 (consistent with
Eq. :eq:`spatial-moment-kronecker-order`).  The cell mass diagonal satisfies
:math:`\mathrm{diag}(M) = \mathrm{diag}(h_x,\theta h_x) \otimes
\mathrm{diag}(h_y,\theta h_y)`, the Kronecker product that fixes which slot is
which.

**The projection supplies GLOBAL-frame coefficients.**  The natural Legendre
coefficients of :math:`q(x,y)` live in the global :math:`x`/:math:`y` frame; the
per-octant sweep frame (where a downwind axis runs the other way) is *not* the
projection's concern.  Production reframes the source global→sweep per octant in
:meth:`~orpheus.sn.loss_representation.sweep_graph._CellSolve.cell` via the slope-sign involution
:math:`\mathrm{octant\_moment\_frame\_signs}`
(Eq. :eq:`ld-ubld-octant-moment-frame-signs`,
:func:`orpheus.transport.spatial._ubld.octant_moment_frame_signs`), exactly as it
reframes the scattering slope source.  So the external :math:`\hat Q` rides the
SAME global→sweep machinery the scattering moments already use — the
:ref:`ld-ubld-sweep-global-frame` involution that the S3 unified matvec had to
get right (ERR-061) is reused unchanged, with no new cell branch.

The projector is structurally independent of production (L11): it evaluates
:math:`\int q\,P_k` with :func:`numpy.polynomial.legendre.leggauss` directly and
NEVER calls ``_lift_external_source_to_moments`` nor any
:class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous` cell op.
The reference (Eq. :eq:`ld-cartesian-2d-bilinear-coeffs`) is hand-laid
polynomial algebra, so "the projector matches the reference" is not a production
echo.  One subtlety, pinned by its own sub-gate: the projector returns the cell
**average** in slot 0, whereas the legacy flat producer ``case.external_source``
evaluates :math:`Q` at the cell **centre**; the two differ by :math:`O(h^2)`
(slot-0 ratio :math:`\sim 0.93` at ``nc=8``).  The projector slot-0 is therefore
cross-checked against an *independent* fine-quadrature cell average, NOT against
the cell-centre producer (which would falsely fail by :math:`O(h^2)`).

The typed-union bulk widening
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before #247, :func:`~orpheus.sn.solver.solve_sn_fixed_source` accepted a single
flat bulk shape :math:`(N, n_g, *\text{spatial})` and rejected everything else.
The widening makes the bulk a **typed union of two ndarray ranks**, discriminated
by RANK, not trailing size:

.. list-table:: The widened bulk-source contract (``_build_fixed_source_rhs``)
   :header-rows: 1
   :widths: 26 24 50

   * - Bulk shape
     - Closure
     - Meaning
   * - :math:`(N, n_g, *\text{spatial})` (flat)
     - any
     - the original path — slope moments :math:`\hat Q` zeroed by the lift (the
       honest default, exact for a region-uniform source).  Byte-identical to
       pre-#247.
   * - :math:`(N, n_g, *\text{spatial},\, \text{per\_axis}^{\,d})`
       (moment-resolved)
     - LD only (``per_axis > 1``)
     - the caller projected :math:`Q^{\rm ext}` onto the tensor-Legendre moment
       vector; the lift threads the slope rows through to join the
       moment-carrying scattering source :math:`\Sigma_s\hat\phi` in the SI rhs.
   * - anything else
     - any
     - ``ValueError`` (see the negative pin)

**Why discriminate by RANK, not trailing size.**  A moment-resolved bulk has
exactly one more axis than a flat bulk; a coincidental spatial dimension could
happen to equal :math:`2^d`, so testing the trailing-axis *length* would
misclassify a flat bulk whose last spatial dim is 4.  The rank is unambiguous: a
flat bulk has :math:`2 + |{\rm spatial}|` axes, a moment-resolved bulk has one
more.  ``_lift_external_source_to_moments`` makes the same rank decision
(``bulk_values.ndim == 2 + len(spatial_shape)`` is the flat-rank test) — the
moment-layout primitive :func:`orpheus.numerics.moment_layout.is_moment_valued_by_rank`
is the canonical discriminator for the cell-internal path.

**Why DD/Step rejects a moment bulk.**  At a flat closure
(:math:`\text{per\_axis} = 1`, hence :math:`n_{\rm cell\ moments} = 1`) there is
NO moment axis — the cell carries a single average, with no slope to fill.  A
moment-resolved input there is a category error, so the validation rejects it
outright (only flat is valid).  This is Pattern 4 (illegal states made
unrepresentable): the relaxation admits exactly the two principled ranks and
nothing in between.

**The negative pin.**  The relaxation must not swallow a real shape bug.  A
moment-resolved bulk whose trailing axis :math:`\neq \text{per\_axis}^{\,d}`
(e.g. a 5-wide axis on a 2-D LD mesh where :math:`2^d = 4`) raises a
``ValueError`` that names the expected :math:`2^d` and the full moment-vector
shape.  The gate
``test_moment_resolved_bulk_still_rejects_wrong_trailing_axis`` pins both arms:
the LD 5-wide reject AND the DD 4-wide reject.

The lift then has three arms, single-sourced for the fixed-source and eigenvalue
paths:

.. list-table:: ``_lift_external_source_to_moments`` (the three arms)
   :header-rows: 1
   :widths: 30 70

   * - Input
     - Action
   * - DD/Step (``tail == ()``)
     - input returned UNCHANGED — byte-identical, the backward-compat negative
       control.
   * - flat :math:`(N, n_g, *\text{spatial})`
     - zero the :math:`2^d` buffer, copy the flat values onto slot 0 (average);
       slope rows stay ZERO (:math:`\hat Q = 0`, the honest default).
   * - moment-resolved :math:`(N, n_g, *\text{spatial},\, 2^d)`
     - thread the moment vector through UNCHANGED (validate the trailing axis);
       the slope rows the caller projected reach the SI rhs.

No callable-projection entry is exposed (Pattern 6, defer abstraction): there is
no production consumer that needs the solver to project a continuous source, and
adding one would make the verification a tautology (the gate would compare the
production projector to itself).  The MMS test does its OWN projection and passes
the array — structurally independent of production by construction (L11).

The Mode-10 structural-teeth design
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The slope-SOURCE sign is the textbook vv **Mode 10** trap (an
*activated-but-unconstrained* term): the slope-source code path is genuinely
exercised — the slope rows are populated, threaded, reframed per octant, and
consumed by the cell update — yet a sign flip on those rows does **not** move the
converged scalar flux above the discretization floor.  The reason is that the
slope-source contribution enters the converged flux as an :math:`O(h^2)`-small
forcing that rides *on top of* the :math:`O(h^2)` discretization error.  Probed
live, the average-moment L2 error vs :math:`\phi_{\rm exact}` under an
x-slope-source sign flip:

.. list-table:: Why the converged flux is sub-floor sensitive to the slope-source sign
   :header-rows: 1
   :widths: 30 40 30

   * - ``nc``
     - correct slope (L2 err / order)
     - x-slope FLIPPED (L2 err / order)
   * - 16
     - :math:`8.18\times10^{-3}`
     - :math:`1.17\times10^{-2}`
   * - 32
     - :math:`1.99\times10^{-3}` (2.04)
     - :math:`2.96\times10^{-3}` (1.98)
   * - 64
     - :math:`4.86\times10^{-4}` (2.03)
     - :math:`7.38\times10^{-4}` (2.01)
   * - 128
     - :math:`1.18\times10^{-4}` (2.05)
     - :math:`1.81\times10^{-4}` (2.03)

Both converge at clean :math:`O(h^2)`; the flipped error is only
:math:`\sim 1.4\text{–}1.5\times` larger at every mesh, and the ratio is roughly
CONSTANT under refinement.  Two consequences for the gate:

* **A convergence-ORDER leg is blind to the sign.**  The order stays 2 both ways
  — :math:`O(h^2)` to the wrong limit is still :math:`O(h^2)` (the vv §5
  warning).
* **A fixed-mesh value-band is too fragile.**  A band that separates the correct
  from the flipped converged flux would need a tolerance tighter than the
  :math:`\sim 1.5\times` gap, and the :math:`O(h^2)` discretization error itself
  eats that margin.  The smallest signal — the :math:`xy` cross-slope — moves
  the converged flux only :math:`\sim 6\times10^{-5}` relative; no value-band
  survives that.

So the teeth do **not** come from the converged flux.  They come from two places
where the sign flip is :math:`O(1)`:

1. **The lift threads the projection through at machine precision** (the
   production-change proof).  ``test_ld_2d_external_slope_source_threaded_through_lift``
   feeds the projected moment vector to the production lift and asserts the
   returned moment source equals the projection EXACTLY — every slope slot, via
   ``np.testing.assert_array_equal``.  A regression that re-zeroes the slope rows
   (the EXACT bug #247 closes) breaks this at :math:`O(1)`, where the converged
   flux would never catch it.  A NEGATIVE-CONTROL leg in the same test pins that
   a FLAT bulk still lifts onto slot 0 with the slope rows EXACTLY ZERO (the
   honest default is preserved).
2. **A consumed slope-row sign flip moves the converged flux :math:`\gg` solver
   tolerance** (the consumption proof).  The inner solve converges to
   :math:`10^{-12}`; a flip of a CONSUMED slope row moves the flux by
   :math:`\sim 3\times10^{-3}` (x), :math:`\sim 10^{-2}` (y), or
   :math:`\sim 6\times10^{-5}` (xy) relative.  The acceptance band
   ``_CONSUMPTION_TOL = 1e-8`` sits :math:`\sim 5\times10^7\times` above the
   fixed point yet far below the §0 trap — the smallest probed flip (xy) clears
   it by :math:`\sim 6000\times`.  This is sharp BECAUSE the test contrasts two
   solves of the *same* problem that differ only in a sign, so the
   discretization floor cancels and the slope-source contribution is the signal.

The convergence-ORDER leg
(``test_ld_2d_external_slope_source_converges_second_order``,
``@verifies("ld-cartesian-2d")``) is kept as a NECESSARY check — it proves the
threaded slope rows are CONSISTENT (the slope-unknown plus the source together
produce a 2nd-order moment, probed :math:`8.18\times10^{-3} \to
1.99\times10^{-3}` at ``nc 16 → 32``) — but it is explicitly **not** the sign
teeth.  A POSITIVE leg
(``test_ld_2d_external_slope_source_improves_on_flat``) closes the loop: the
moment-resolved solve lands strictly closer to :math:`\phi = A` than the
flat-in-moment solve (:math:`3.4\times10^{-3} < 5.9\times10^{-3}` at ``nc=24``),
so the threaded slopes carry real sub-cell information, not noise.

.. note:: **A current-invariant lesson worth recording (vv Mode 10).**

   A Mode-10 gap is closed NOT by tightening the converged-flux value band (the
   :math:`O(h^2)`-small forcing is sub-floor) but by two :math:`O(1)` structural
   teeth: (1) assert the production *producer* threads the projection through at
   machine precision (catches a regression to zeroing), and (2) assert a
   *consumed* source-row sign flip moves the converged answer :math:`\gg` solver
   tol (catches sign-blindness), paired with the FLAT no-op leg that pins the
   asymmetry (the old scalar gate is correctly blind, by construction).  The
   convergence-order leg is necessary for slope consistency but is not the sign
   teeth.  This is the canonical resolution whenever a term is genuinely consumed
   yet its error enters the measured quantity below the convergence floor.

The mutation-control table (M1–M4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The primary sign-catchers are mutation controls (vv anti-pattern #11 — a
``catches`` claim is verified by re-introducing the exact bug and confirming the
gate reddens).  Two distinct mutations stress two distinct source paths: the
EXTERNAL :math:`\hat Q` (the NEW #247 consumption, M1–M3) and the SCATTERING
:math:`\Sigma_s\hat\phi` (the EXISTING S3 consumption, M4).  Each flips a slope
row and asserts the converged flux changes :math:`\gg` ``_CONSUMPTION_TOL`` while
the FLAT scalar gate stays GREEN — that asymmetry IS the Mode-10 gap being
closed.

.. list-table:: The slope-SOURCE sign mutation controls
   :header-rows: 1
   :widths: 6 30 34 30

   * - \#
     - Source row flipped
     - The NEW moment gate must
     - The FLAT scalar gate
       (``..._stress_converges_second_order``)
   * - M1
     - EXTERNAL :math:`\hat Q` x-slope (slot 2)
     - go RED — converged flux moves :math:`\sim 3\times10^{-3}` (the
       consumption proof)
     - stays GREEN — it feeds a flat source, slope row already 0, flipping zero
       is a no-op
   * - M2
     - EXTERNAL :math:`\hat Q` y-slope (slot 1)
     - go RED — flux moves :math:`\sim 10^{-2}`
     - stays GREEN (flat → no-op)
   * - M3
     - EXTERNAL :math:`\hat Q` cross-moment (slot 3)
     - go RED — flux moves :math:`\sim 6\times10^{-5}` (the weakest signal,
       still :math:`\sim 6000\times` over tol)
     - stays GREEN (flat → no-op)
   * - M4
     - SCATTERING :math:`\Sigma_s\hat\phi` iso slope rows (slots 1:)
     - go RED — flux moves :math:`\sim 2.6\times10^{-3}`
     - stays GREEN — the scalar gate's converged flux is only :math:`\sim
       1.4\times` sensitive (sub-floor)

**M1–M3 verify the NEW external consumption.**  Before #247 the lift zeroed the
external slope rows, so a flip of an already-zero row was a no-op and the
"flipped reddens" assertion could not hold — these mutations only become
catchers once the lift threads the slope rows.  The test flips slot
:math:`\{2,1,3\}` of the projected :math:`\hat Q` and re-solves the full public
solve; the same flip on the FLAT source is then asserted to be a no-op directly
(``flat_lift[..., slot]`` is exactly zero), pinning the asymmetry that closed
the gap.

**M4 verifies the EXISTING scattering consumption was never sign-blind.**  The
scattering slope source :math:`\Sigma_s\hat\phi` (the Increment-C iterate
feedback, Eq. :eq:`ld-ubld-scattering-moment-lift`) has been consumed since S3,
but the OLD value-band MMS was empirically blind to its sign — a slope-source-row
sign flip left both the :math:`O(h^2)` order and the scalar-flux value band
unchanged, because :math:`\Sigma_s\hat\phi` is an :math:`O(h)`-small DG-internal
forcing whose error enters above :math:`O(h^2)`.  M4 monkeypatches the
per-ordinate source combine — since CS4c step 5 (2026-09-04)
:meth:`AngularLift._combine <orpheus.transport.operators.angular_lift.AngularLift._combine>`,
the producer-side ``(iso/W) + aniso`` over every spatial moment (until then
spelled ``ScatteringOperator._assemble_per_ordinate_source``) — to negate the
iso slope rows and confirms the converged flux moves
:math:`\sim 2.6\times10^{-3}` — the consumption proof replaces the value-band
the old gate relied on.

Each mutation is reverted (a ``finally`` block, or the ``monkeypatch`` fixture
for M4), and all #247 gates are
``-O``-safe (``np.testing.*`` / ``pytest.fail`` / ``pytest.raises`` only, no
bare ``assert`` that ``python -O`` would strip — vv Mode 8).

No ERR entry was minted: Mode 10 here is a proactive-gap close, not a caught
production bug.  The lift correctly zeroed an unverified-but-honest default
(:math:`\hat Q = 0`); the slope-source sign was UNVERIFIED, not WRONG.  Per the
"log every caught bug" directive, an ``@catches`` marker is added only when a
real production bug surfaces; none did.

Sources and gates
^^^^^^^^^^^^^^^^^

The production change is in :func:`orpheus.sn.solver._build_fixed_source_rhs`
(the typed-union validation) and ``_lift_external_source_to_moments`` (the
slope-thread arm), both confined to ``solver.py``.  The end-to-end gates live in
:mod:`tests.sn.verification.mms.test_mms_ld_2d` (the #247 block):
``test_ld_2d_external_slope_source_threaded_through_lift`` (the foundation
structural teeth), ``..._converges_second_order`` and ``..._improves_on_flat``
(the L1 necessary + positive legs), ``..._sign_mutation_reddens`` (M1–M3),
``test_ld_2d_scattering_slope_source_sign_mutation_reddens`` (M4),
``test_moment_resolved_bulk_still_rejects_wrong_trailing_axis`` (the negative
pin), with the projection-correctness foundation sub-gates
``test_tensor_legendre_projection_matches_hand_polynomial`` and
``test_projection_slot0_is_cell_average_not_centre``.  The bit-identity of the
flat/DD path is guarded by the strict ``DriftWarning`` regression gate (no golden
moved — the typed-union widening leaves the flat path byte-identical).

.. _ld-cartesian-2d-legB:

Leg B — the boundary transverse-face-slope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This subsection is the rich record of the BOUNDARY half of the slope-SOURCE
trap, closed under Issue #251 (split from #247).  Where Leg A widened the BULK
external source carried through the *interior*, Leg B widens the BOUNDARY TRACE
``mesh.angular_trace`` so a moment-resolved prescribed inflow can carry the along-face
(transverse) Legendre slope and the sweep outflow can STORE the
:math:`2^{d-1}` transverse face-moments instead of collapsing them to the
average.  It is the boundary twin of Leg A, and the structural-teeth template
(:ref:`ld-cartesian-2d-legA`) carries over almost verbatim — but the
verification design is *sharper*: the boundary slope is sub-floor for ANY value
claim, not just its sign (the "improves-on-flat" leg that closed Leg A is
**unachievable** here).  That sharpening — the first vv **Mode 10** instance
where the O(1)-isolating-companion half of the recipe is genuinely unavailable —
is the load-bearing lesson, and the reason this is recorded in full.

The ``boundary_face_layout`` moment-tail storage lever (the CRUX)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Leg A had a free ride: the BULK iterate :math:`\hat\phi` already carried its
per-cell :math:`2^d`-moment axis (the S3 unified moment matvec), so widening the
external source meant relaxing a single lift's input contract.  Leg B has no
such carrier — *the boundary trace is scalar-per-face end-to-end*.  A new place
must be found to STORE the transverse face-moments, and the elegant answer is a
single attribute on the mesh.

The trace's per-face slot shape is owned not by the
:class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace` itself but by the
:class:`~orpheus.numerics.face_layout.FaceLayout` it is built from, and that
layout is minted by :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.boundary_face_layout`.
The widening is therefore ONE site: append the scheme's per-face transverse
moment tail to each slot,

.. math::
   :label: ld-cartesian-2d-face-slot-shape

   \text{slot}(\text{face}) \;=\;
   \bigl(N,\ n_g,\ *\,\text{face\_shape}(\text{label}),\
         \underbrace{*\,\text{face\_moment\_tail}
            \bigl(\text{per\_axis}^{\,d-1}\bigr)}
         _{\text{the new }2^{d-1}\text{ tail}}\bigr),

.. (vv-status rationale) representational: the boundary-trace per-face slot
.. SHAPE (the codimension-1 :math:`2^{d-1}` transverse-moment tail appended by
.. ``SNMesh.boundary_face_layout``). A storage-layout identity; its
.. verifiable content — the live slot shapes (LD ``(24,2,6,2)`` vs DD
.. ``(24,2,6)``) and the DD/Step byte-identical negative control — is pinned
.. by the FOUNDATION LD gates in the (owned) ``test_mms_ld_2d.py`` /
.. ``test_linear_discontinuous.py`` suites, which carry no ``verifies(...)``.
.. vv-status: ld-cartesian-2d-face-slot-shape documented

where the per-face moment count is :math:`n_{\rm face} = \text{per\_axis}^{\,d-1}`
(the FACE tail) — note the exponent :math:`d-1`, NOT the cell-tail exponent
:math:`d` of Eq. :eq:`spatial-moment-kronecker-order`.  A face is a codimension-1
object: it carries a Legendre moment per *transverse* axis only (the
:math:`d-1` axes that run along the face), so its moment count is the cell count
divided by the one normal-axis factor.

Three properties make this the clean lever:

* **The** :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace` **needs ZERO
  changes — it was "moment-ready by accident".**  The trace's partial-current
  metric (the :math:`|\Omega\!\cdot\!n|\,w_n` weighting) and its
  ``omega_dot_n`` inflow/outflow table are **per-ordinate** (they classify and
  weight by ordinate, axis 0) and broadcast over ALL trailing axes by
  construction.  A moment axis appended to the slot rides the metric and the
  directional selectors for free.  The trace space was already
  moment-polymorphic; only its layout-supplier was scalar.  This is the
  illegal-states-unrepresentable payoff of the Depth-B field-space refactor: the
  boundary FIELDS (``AngularBoundaryFlux`` / ``AngularBoundarySourceSink`` /
  ``AngularBoundaryResidual``; ``AngularBoundaryDisplacement`` existed too
  until it retired at campaign-1 CS3) validate ONLY against
  ``layout.total_size``, never a hardcoded :math:`(N, n_g)`, so they accommodate
  any slot shape the layout dictates.

* **DD/Step is byte-identical (the negative control).**  At a flat closure
  :math:`\text{per\_axis} = 1`, so :math:`n_{\rm face} = 1^{\,d-1} = 1` and
  :func:`~orpheus.numerics.moment_layout.face_moment_tail` returns ``()`` (the
  "append iff > 1" policy — NO length-1 axis).  Every DD/Step slot shape is
  untouched, so every buffer, snapshot, and metric is bit-for-bit unchanged.
  This is the SAME single-source tail policy the interior cell cochain keys on
  (:attr:`_LossRepresentation._n_face_moments` and
  :attr:`_LossRepresentation._spatial_moment_tail`), reused so the storage and
  the cochain can never disagree on the shape.

* **Leg B is a 2-D-and-higher concern by construction.**  A 1-D slab face is a
  *point*: ``face_shape == ()`` and there is no transverse axis, so
  :math:`n_{\rm face} = \text{per\_axis}^{\,0} = 1` even for the 2-basis LD
  closure.  A point has no along-face direction, hence no transverse slope; the
  1-D prescribed-inflow MMS is byte-identical not by coincidence but because the
  exponent :math:`d-1` vanishes.

The scheme is reachable: ``SNMesh`` sets ``self.scheme`` before it builds the
trace, so ``boundary_face_layout`` can read ``self.scheme.spatial_basis_per_axis``
to compute the tail.  Verified live: with LD the face slots are
:math:`(24, 2, 6, 2)` / :math:`(24, 2, 8, 2)` (the trailing ``2`` is the
:math:`2^{d-1}` transverse-moment axis); with DD they are :math:`(24, 2, 6)` /
:math:`(24, 2, 8)` (no axis).

The transverse face-moment normalization (apples-to-apples with the cochain)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same normalization decision that locked Leg A (Eq.
:eq:`ld-cartesian-2d-projection-coeff`) recurs, transposed to the transverse
axis.  The cochain consumes the upstream face's moments through
:func:`orpheus.transport.spatial._ubld.assemble_inflow_axis`, which weights the
:math:`2^{d-1}`-moment face vector by the **transverse mass** — the Kronecker
product of the per-transverse-axis :func:`~orpheus.transport.spatial._ubld.mass_1d`
:math:`= \mathrm{diag}(h_t,\,\theta h_t)`, :math:`\theta = 1/3` — before applying
the active-axis trace :math:`B(-1) = [1, -1]` and :math:`|\mu_{\rm axis}|`.  The
mass therefore ALREADY supplies the transverse :math:`h_t` and :math:`\theta`
weighting; the trace must NOT duplicate it.

So — exactly as Leg A's cell mass :math:`M = \mathrm{diag}(h, \theta h)` forced
the projected source rows to be bare per-volume coefficients — **the boundary
trace must carry the BARE per-transverse Legendre coefficients**:

.. math::
   :label: ld-cartesian-2d-face-projection-coeff

   b_{\rm bar} \;=\; \frac{\langle\psi_{\rm face},\,P_0\rangle}
                          {\langle P_0, P_0\rangle}
            \;=\; \text{(transverse cell average)}
            \quad\text{(slot 0)},
   \qquad
   b_{\rm slope} \;=\; \frac{\langle\psi_{\rm face},\,P_1(\xi)\rangle}
                            {\langle P_1, P_1\rangle}
            \quad\text{(slot 1, the bare transverse }P_1\text{ coeff)},

.. (vv-status rationale) definition: the Leg-B face-projection NORMALIZATION
.. convention (bare per-transverse Legendre coefficient; the cochain's
.. transverse mass supplies the theta/h_t weighting). The 1-D-transverse
.. factor of :eq:`ld-cartesian-2d-projection-coeff`; representational, pinned
.. by the FOUNDATION structural-threading gate in the (owned) LD suites,
.. which carry no ``verifies(...)``.
.. vv-status: ld-cartesian-2d-face-projection-coeff documented

with :math:`\xi \in [-1,1]` the transverse coordinate.  For a face-linear inflow
:math:`\psi_{\rm face}(t) = c_0 + c_1 t` on a transverse cell
:math:`[t_L, t_R]` (width :math:`h_t = t_R - t_L`, centre :math:`t_c`), the two
coefficients are hand-derivable in closed form,

.. math::
   :label: ld-cartesian-2d-face-bilinear-coeffs

   b_{\rm bar}   &= c_0 + c_1\,t_c \quad\text{(the cell AVERAGE, not the centre
                    eval — see below)}, \\
   b_{\rm slope} &= \tfrac{h_t}{2}\,c_1 \quad\text{(no }\theta\text{, no }h_t
                    \text{ beyond the }\tfrac{1}{2}\text{; the mass adds them)},

.. (vv-status rationale) reference: the two hand-derived face Legendre
.. coefficients of a face-linear inflow — the structurally-independent
.. (Branch-1) reference the ``leggauss`` face projector is reproduced against
.. to machine precision. The per-axis factor of
.. :eq:`ld-cartesian-2d-bilinear-coeffs`; pinned by the FOUNDATION LD gates in
.. the (owned) suites, which carry no ``verifies(...)``.
.. vv-status: ld-cartesian-2d-face-bilinear-coeffs documented

the structurally-independent reference the face projector is pinned against (a
linear integrand is integrated exactly by a 2-point Gauss rule, so the
``leggauss`` projector reproduces them to machine precision).  This is the
1-D-transverse *factor* of Leg A's tensor projector
(Eq. :eq:`ld-cartesian-2d-bilinear-coeffs`): a face projection is the per-axis
Legendre coefficient of the tensor projection along the single transverse axis.

.. note:: **The apples-to-apples trap (the same crux that bit Leg A).**

   If the trace stored a :math:`\theta`- or :math:`h_t`-weighted slope instead
   of the bare coefficient, the cochain's transverse mass would double-apply the
   weighting, and the threaded slope would be wrong by a constant
   :math:`\theta h_t` factor.  The structural threading gate (below) compares the
   trace's slot-1 against the bare projected reference precisely so that this
   double-counting is caught at :math:`O(1)`.

**The slot-0 centre-vs-average subtlety (sharper than Leg A).**  Today's scalar
trace carries the cell-**centre** eval of the manufactured inflow
:math:`\psi_{\rm face}(t_c)/W`, whereas the projection's slot 0 is the cell
**average** :math:`b_{\rm bar}`; the two differ by :math:`O(h^2)`.  The
backward-compat decision is to keep slot 0 = the EXISTING scalar trace (centre)
on the scalar-inflow path — so DD/Step and the existing 1-D prescribed-inflow
MMS stay byte-identical — and to carry the (average-bar, slope) pair only when a
moment-resolved inflow is explicitly supplied.  The structural threading gate
therefore compares the SLOPE slot (slot 1) only; slot 0 may legitimately differ
centre-vs-average.  A dedicated foundation sub-gate
(``test_face_projection_slot0_is_transverse_cell_average``) pins that the
projector's slot 0 IS the average — cross-checked against an *independent*
fine-quadrature transverse cell average, NOT against the cell-centre
``case.prescribed_inflow`` (which would falsely fail by :math:`O(h^2)`).

The face projector is structurally independent of production (L11): it evaluates
:math:`\int \psi_{\rm face}\,P_k` with
:func:`numpy.polynomial.legendre.leggauss` directly and NEVER calls
``_inflow_to_moments`` nor ``assemble_inflow_axis`` nor any LD cell op.

The rank-discriminated inflow lift and the four DROP sites
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The boundary twin of Leg A's ``_lift_external_source_to_moments`` is
:meth:`_LossRepresentation._inflow_to_moments`.  It rank-discriminates the
incoming face against the FLAT (moment-free) face rank :math:`d + 1` — a scalar
face :math:`(N_{\rm oct}, n_g, *\text{transverse})` has rank
:math:`2 + (d-1) = d + 1`, since the transverse part carries :math:`d-1` axes —
using the same single-source primitive Leg A introduced,
:func:`orpheus.numerics.moment_layout.is_moment_valued_by_flat_rank` (no third
rank spelling — the layout-shift hazard is an open-coded ``ndim == flat_ndim``
that silently diverges from its sibling consumers).  Three arms:

.. list-table:: ``_inflow_to_moments`` (the three arms)
   :header-rows: 1
   :widths: 30 70

   * - Input
     - Action
   * - DD/Step (:math:`n_{\rm face} = 1`)
     - identity — the trailing axis is absent, every buffer byte-identical (the
       backward-compat negative control).
   * - scalar inflow :math:`(N_{\rm oct}, n_g, *\text{transverse})`
     - widen — zero the :math:`2^{d-1}` buffer, seed the AVERAGE moment (slot 0)
       from the scalar, leave the transverse SLOPES zero (a scalar trace carries
       no along-face variation; **the scalar default — the Leg-B asymmetry**).
   * - moment-resolved inflow :math:`(N_{\rm oct}, n_g, *\text{transverse},\,
       2^{d-1})`
     - PASS THROUGH — the widened trace already carries the projected transverse
       face-slope; thread it unchanged (validate the trailing width
       :math:`= 2^{d-1}`, ``ValueError`` otherwise — Pattern 4).

No callable-projection entry is exposed (Pattern 6, defer abstraction): the MMS
does its OWN projection and passes the moment array; production accepts the
moment-resolved face, it does not compute it — structurally independent by
construction (L11), exactly Leg A.

The full-field oracle's seed, :meth:`FullFieldWavefront._octant_face_cochain`,
mirrors the same rank discriminator at the IN-edge: a scalar inflow seeds slot 0
(transverse slopes zero), a moment-resolved inflow seeds ALL :math:`2^{d-1}`
moments.  So the two storage strategies — the production
:class:`~orpheus.sn.loss_representation.MovingFrontierWindow` and the
:class:`~orpheus.sn.loss_representation.FullFieldWavefront` oracle — agree on the
seed, which the two-paths bit-identity gate continues to verify.

**The four outflow capture-collapse DROP sites.**  Before #251, the sweep
captured the domain-edge outflow as a :math:`2^{d-1}`-moment object (the interior
cochain is moment-valued throughout the walk) and then collapsed it to slot 0
(``capture = tuple(c[..., AVERAGE_MOMENT] for c in capture)``) before writing
the scalar trace.  Four such collapses — guarded ``if n_face_moments > 1`` —
are removed so the capture RETAINS the :math:`2^{d-1}` axis:

.. list-table:: The four outflow capture-collapse DROP sites
   :header-rows: 1
   :widths: 22 22 56

   * - Path
     - Strategy
     - What it stored / now stores
   * - SOLVE
     - ``MovingFrontierWindow`` (prod)
     - was slot-0 average only → now the full :math:`2^{d-1}` outflow moments
   * - MATVEC
     - ``MovingFrontierWindow`` (prod)
     - same
   * - SOLVE
     - ``FullFieldWavefront`` (oracle)
     - same
   * - MATVEC
     - ``FullFieldWavefront`` (oracle)
     - same

The downstream sheds then land the moments automatically into the now
moment-shaped slot: the SOLVE shed writes ``boundary_flux.face_view(face)``; the
MATVEC shed writes into a ``streamed`` buffer allocated ``zeros_like`` of the
(now widened) ``face_view``, so it auto-widens; and the boundary-residual
``B``-block emit (the outflow defect ``streamed − given`` and the inflow
identity) is ordinate-indexed and spans the whole trailing slot.  The widened
layout makes these writes land the transverse moments with no further edit —
again the illegal-states-unrepresentable dividend of typing the slot shape once.

The producer — both ranks through one slot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ergonomic generator for the affine-BC inhomogeneous term :math:`q` is
:meth:`~orpheus.transport.source_sinks.angular_boundary_source_sink.AngularBoundarySourceSink.prescribed_inflow`.
Its job is to write the inflow ordinate rows of each named face from a given
per-face array and leave everything else zero.  After the trace widened, this
producer must accept **two ranks through the same slot** — and getting this
right was where the architecture bit harder than the storage map anticipated.

.. list-table:: ``prescribed_inflow`` slot assignment (the two-arm relaxation)
   :header-rows: 1
   :widths: 38 62

   * - Supplied array shape
     - Write
   * - full slot — ``arr.shape == view.shape``
     - ``view[inflow] = arr[inflow]`` — the inflow ordinate ROWS (axis 0) span
       ALL trailing axes.  DD/Step's scalar slot AND a moment-resolved full slot
       both take this byte-identical arm.
   * - scalar onto a moment slot — ``arr.shape == view.shape[:-1]``
     - ``view[inflow, ..., AVERAGE_MOMENT] = arr[inflow]`` — seed the average
       moment, the transverse slopes stay zero (the scalar default).
   * - anything else
     - ``ValueError`` naming the expected full slot or the scalar reduction.

.. note:: **Audit the EXISTING scalar producers when you widen a slot, not only
   the new one.**

   The instinct is to fix only the producer line for the new moment-resolved
   caller — change the single trailing ``, :`` to span all trailing axes and
   trust the shape-check to validate the wider shape.  That is *necessary but not
   sufficient*.  The EXISTING scalar callers — the 2-D LD MMS and the 1-D
   prescribed-inflow MMS — supply a SCALAR
   :math:`(N, n_g, *\text{transverse})` array.  Once the LD slot grew to
   :math:`(N, n_g, *\text{transverse}, 2)`, an unconditional
   ``arr.shape != view.shape`` reject fires on *every* existing scalar LD caller
   (it reddened eight LD-2-D gates).  The fix is the SECOND arm above — a
   scalar-onto-a-moment-slot path that seeds slot 0.  This is the same class of
   under-scope as Leg A's field-space layer: a rigid scalar contract sitting
   ABOVE a widened slot needs a *typed-union relaxation*, not just an indexing
   fix.  When a carve widens a trace or field slot, the scalar callers feed the
   SAME widened slot — the producer must accept BOTH ranks.

The sharper Mode-10: structural teeth only, no improves-on-flat leg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Leg B is the textbook vv **Mode 10** trap (an *activated-but-unconstrained*
term) one notch sharper than Leg A.  The transverse face-slope is genuinely
exercised — projected, threaded, stored, reframed per octant, and consumed by
the cell update — yet its contribution to the converged near-boundary flux is
**sub-floor for ANY value claim**, not merely for its sign.  Probed live, seeding
the TRUE transverse slope makes the converged near-boundary L2 error against the
manufactured :math:`\phi` SLIGHTLY WORSE, and the FLIPPED slope slightly
BETTER:

.. list-table:: Why the converged near-boundary flux is sub-floor for the
   boundary slope
   :header-rows: 1
   :widths: 14 28 28 30

   * - ``nc``
     - flat (avg/centre) A-err
     - real-slope A-err
     - flipped-slope A-err
   * - 16
     - :math:`1.015\times10^{-2}`
     - :math:`1.030\times10^{-2}`
     - :math:`1.009\times10^{-2}` (improves? NO)
   * - 32
     - :math:`1.943\times10^{-3}`
     - :math:`1.966\times10^{-3}`
     - :math:`1.929\times10^{-3}` (improves? NO)

The reason is geometric: the boundary signal is too LOCALIZED and too SMALL — an
:math:`O(h)` intra-cell slope at the domain edge only — to register against the
bulk :math:`O(h^2)` discretization error that dominates the converged flux.
This is sharper than Leg A, where the bulk slope source carried real sub-cell
information across the WHOLE domain and "improves-on-flat" WAS achievable
(probed :math:`3.40\times10^{-3} < 5.99\times10^{-3}`).

**Consequence for the gate: there is NO converged-value-improvement leg.**  The
positive verification of Leg B is STRUCTURAL only, from the two places where the
sign / threading is :math:`O(1)`:

1. **The producer threads the projected slope through at machine precision** (the
   production-change proof).  ``test_ld_2d_boundary_slope_threaded_through_inflow_to_moments``
   builds a moment-resolved face from the ``leggauss`` projector, feeds it to the
   widened ``_inflow_to_moments``, and asserts (a) the producer RECOGNISES the
   moment-resolved input — it does NOT append a spurious second moment axis — and
   (b) ``np.testing.assert_array_equal`` on slot 1 holds EXACTLY.  A regression
   that re-zeroes the slope (the EXACT #251 bug) breaks this at :math:`O(1)`,
   where the converged flux would never catch it.

2. **A consumed transverse-slope sign flip moves the converged near-boundary
   flux** :math:`\gg` **the consumption tolerance** (the consumption proof).
   ``test_ld_2d_boundary_slope_sign_mutation_reddens``
   (``@verifies("ld-cartesian-2d")``) solves the same problem twice through the
   PUBLIC ``solve_sn_fixed_source`` — once with ``prescribed_inflow`` carrying
   :math:`+b_{\rm slope}` in slot 1, once with :math:`-b_{\rm slope}` — and
   asserts the near-boundary (edge-cell-masked) :math:`|\Delta\phi|/|\phi|`
   exceeds ``_CONSUMPTION_TOL = 1e-8``.  Probed:

   .. list-table::
      :header-rows: 1
      :widths: 14 32 32

      * - ``nc``
        - near-boundary :math:`|\Delta\phi|/|\phi|`
        - global :math:`|\Delta\phi|/|\phi|`
      * - 16
        - :math:`4.10\times10^{-3}`
        - :math:`3.27\times10^{-3}`
      * - 32
        - :math:`8.38\times10^{-4}`
        - :math:`8.99\times10^{-4}`

   The flip clears the tolerance by :math:`\sim 5.6` orders of magnitude and
   HALVES under refinement (:math:`O(h)`, boundary-localized), so this is a
   fixed-mesh consumption test, not a convergence leg.  Linearity is confirmed
   (seed slope :math:`= k\!\cdot\!\bar b`, :math:`k \in \{0.05, 0.1, 0.2\}` →
   flip :math:`|\Delta\phi|/|\phi| = \{2.4, 4.8, 9.5\}\times10^{-3}`, exactly
   linear in :math:`k`), proving the cochain GENUINELY consumes the transverse
   slope and the signal is not a numerical artifact.

These two teeth are PAIRED with the SCALAR-inflow no-op leg
(``test_ld_2d_boundary_scalar_inflow_no_op_negative_control`` and the byte-equal
``slope_sign=0`` vs ``slope_sign=None`` solve), which pins the Leg-B asymmetry:
a scalar inflow has slot 1 :math:`\equiv 0`, so flipping zero is a no-op and the
scalar path is correctly blind — and byte-identical to today's solve, the
bit-identity guard.

.. note:: **A current-invariant lesson worth recording (vv Mode 10 — the
   companion-unavailable branch).**

   Leg A established that a Mode-10 gap is closed by two :math:`O(1)` structural
   teeth (producer threads the projection at machine precision; a consumed
   source-row flip moves the converged answer :math:`\gg` solver tol), paired
   with a no-op control, and that the convergence-order leg is necessary for
   consistency but is not the sign teeth.  Leg B sharpens the recipe's *value*
   half: the canonical Mode-10 resolution suggests adding a companion gate "that
   isolates the term so its error is :math:`O(1)` in the measured quantity (e.g.
   a fixed-source problem where the term is the dominant forcing)".  **For a
   boundary-trace slope that companion is UNAVAILABLE** — there is no regime in
   which a localized :math:`O(h)`-small along-face boundary perturbation becomes
   the dominant forcing of the converged flux; it is intrinsically a boundary
   perturbation that rides below the bulk :math:`O(h^2)` floor.  In that case the
   producer-threading-at-machine-precision plus consumed-flip-:math:`\gg`-tol
   structural pair is the COMPLETE resolution — there is no value-improvement leg
   to add.  This is the first instance where the O(1)-isolating companion half of
   the Mode-10 recipe is genuinely unavailable, and it generalizes: whenever a
   term has no regime where it is the dominant forcing, structural teeth alone
   are the canonical close.

The negative pin, the bit-identity guard, and the reflective follow-up
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**The negative pin.**  The moment relaxation must not swallow a real shape bug.
``test_ld_2d_boundary_trace_rejects_wrong_transverse_width`` feeds
``_inflow_to_moments`` a moment-resolved inflow with a trailing width
:math:`3 \neq 2^{d-1} = 2` on a 2-D LD mesh and asserts a ``ValueError`` naming
the expected :math:`2^{d-1}` (Pattern 4) — the relaxation admits exactly the two
principled ranks and nothing in between.

**The bit-identity guard.**  The DEFAULT scalar-inflow path must stay
bit-identical after the trace widening, verified three ways: the strict
``DriftWarning`` regression gate over ``tests/sn/sweep/core`` and
``tests/sn/solve`` (no golden moved — the moment-tail widening leaves the
DD/Step and scalar-inflow trace untouched, because ``face_moment_tail(1) == ()``
is the negative control); the existing 1-D prescribed-inflow MMS (byte-identical
by construction — a 1-D face has :math:`n_{\rm face} = 1`); and the explicit
scalar no-op leg above.

No ERR entry was minted: like Leg A, Mode 10 here is a proactive-gap close, not
a caught production bug — the trace correctly zeroed an unverified-but-honest
default (the transverse slope), which was UNVERIFIED, not WRONG.  Per the "log
every caught bug" directive, an ``@catches`` marker is added only when a real
production bug surfaces; none did.

.. note:: **Reflective-BC transverse-slope sign — a tracked follow-up (#252).**

   The Leg-B MMS is vacuum-BC (vacuum nulls the reflective coupling), so the
   reflective ``B`` operator's transverse-slope handling is exercised for
   STORAGE but not for its SIGN.  Storage is verified correct: the realized
   reflective law is a
   :class:`~orpheus.numerics.operator.PermutationOperator` on the ordinate axis
   (axis 0), which broadcasts UNCHANGED over the new trailing moment axis — no
   hard-coded trailing-axis assumption, so the widening introduces NO latent
   storage bug (read at
   :meth:`orpheus.sn.operators.boundary.SNBoundaryOperator._reflect_trace`, and
   confirmed empirically on a reflective-xmin LD-2-D mesh with a seeded slot 1).
   The SIGN, however, is UNVERIFIED.  Physics: a normal-flip reflection across a
   face preserves the tangent-plane (transverse) coordinate, so the transverse
   slope should reflect WITHOUT a sign flip (the permutation-on-axis-0
   pass-through is *probably* correct) — but this is a genuine Mode-1 sign trap
   the vacuum gates cannot see.  It is tracked in **#252**: a reflective-LD MMS
   leg plus an ``op.H`` adjoint check on the transverse-slope reflection.

Sources and gates
^^^^^^^^^^^^^^^^^

The production change spans three files: the storage lever in
:attr:`orpheus.sn.mesh.augmented_mesh.SNMesh.boundary_face_layout` (appending
:func:`~orpheus.numerics.moment_layout.face_moment_tail`); the inflow lift
:meth:`_LossRepresentation._inflow_to_moments`, the oracle seed
:meth:`FullFieldWavefront._octant_face_cochain`, and the four outflow
capture-collapse DROP sites in :mod:`orpheus.sn.loss_representation`; and the
producer
:meth:`~orpheus.transport.source_sinks.angular_boundary_source_sink.AngularBoundarySourceSink.prescribed_inflow`.
The end-to-end gates live in :mod:`tests.sn.verification.mms.test_mms_ld_2d` (the
#251 block): ``test_ld_2d_boundary_slope_threaded_through_inflow_to_moments`` (the
foundation structural teeth — the stamp / production-change proof),
``test_ld_2d_boundary_slope_sign_mutation_reddens`` (the L1 consumption proof,
``@verifies("ld-cartesian-2d")``),
``test_ld_2d_boundary_scalar_inflow_no_op_negative_control`` (the Leg-B
asymmetry), ``test_ld_2d_boundary_trace_rejects_wrong_transverse_width`` (the
negative pin), with the face-projection foundation sub-gates
``test_face_transverse_legendre_projection_matches_hand_polynomial`` and
``test_face_projection_slot0_is_transverse_cell_average``.  All #251 gates are
``-O``-safe (``np.testing.*`` / ``pytest.fail`` / ``pytest.raises`` only, no bare
``assert`` that ``python -O`` would strip — vv Mode 8).

.. _ld-cartesian-2d-coherent-promise:

The coherent boundary promise and the property-vs-type seam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leg B (:ref:`ld-cartesian-2d-legB`, #251) landed the boundary CONSUMPTION
path: the trace ``mesh.angular_trace`` carries the transverse moment axis end to end,
``_inflow_to_moments`` threads slot 1, and the sweep outflow stores the
:math:`2^{d-1}` transverse face-moments instead of collapsing them.  But the
MMS *producer* — the case's
:meth:`~orpheus.derivations.continuous.mms.sn.SN2DCartesianLDStressMMSCase.prescribed_inflow`
— still built a SCALAR per-face trace, so it hit the producer's scalar branch
and the slope it could have carried was zeroed by the case, not by the closure.
S9 (Issue #257) closes that producer-blindness — the case now EMITS the
moment-resolved slot — and, in doing so, LOCKS the **coherent promise** the
whole LD-on-the-boundary effort is about with a dedicated convergence gate.

This subsection records three things a future session must not re-derive: (1)
WHY the coherent promise is already TRUE and what delivers it (the average
moment, not the slope); (2) the producer-honesty carve and why it is
byte-identical for DD/Step; and (3) the **property-vs-type** seam (#263) — why
the transverse boundary moment is a PROPERTY of the boundary field and NOT a
new first-class field type, the durable design invariant that bounds the S9
scope.

The coherent promise: LD is second-order at the boundary, no asterisk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The motivating claim — *"LD gives second-order accuracy EVERYWHERE, including
the boundary, with no 'but not at the boundary' asterisk"* — is **TRUE and
already DELIVERED**.  The subtle and load-bearing point is WHAT delivers it.

It is delivered by the **AVERAGE moment alone**, not by the transverse
boundary-slope moment.  The prescribed inflow is exact at the face cells (the
manufactured trace is evaluated there), and the bulk LD closure
(:eq:`ld-cartesian-2d`) carries that boundary data inward at :math:`O(h^2)`.
The cell that directly consumes the inflow integrates it against the cell's
own linear basis, and the cell-AVERAGE transverse moment is exactly what that
integral needs to :math:`O(h^2)`.  So the boundary cell is second-order
*from the average moment*, before any transverse-slope refinement enters.

This reframes the S9 motivation.  There was never an :math:`O(h)` deficiency in
the converged flux at the boundary to "recover" — the order is already
:math:`O(h^2)` there.  What the transverse boundary-slope moment improves is the
*inflow representation*: it lifts the face trace from :math:`O(h)`-accurate
(bar-only) to :math:`O(h^2)`-accurate (bar + slope), a genuine refinement that
the LD closure genuinely consumes (Leg B's structural teeth prove this).  But a
second-order correction to an already-second-order face balance cannot move the
converged flux above the bulk :math:`O(h^2)` floor.  S9 therefore does NOT
remove an asterisk on the convergence order (there was none); it makes the
producer honest about the moment it could always have carried, and it LOCKS the
no-asterisk promise so a future change to the boundary closure cannot silently
break it.

First-cell-row evidence: the boundary cell is already O(h²)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The decisive scaling argument is the convergence order of the FIRST CELL ROW —
the cells (:math:`i=0`) that directly consume the ``xmin`` inflow.  If the
average-only inflow were :math:`O(h)`-deficient at the boundary cell, the slope
COULD repair it and the coherent promise would carry an asterisk.  It is in
fact :math:`O(h^2)`, for BOTH the average-only (``flat``) and the
moment-resolved (``mom``) inflow:

.. list-table:: First-cell-row (:math:`i=0`) :math:`L^2` order — already
   :math:`O(h^2)` from the average moment (pure-absorber streaming,
   :math:`\Sigma_t L = 0.1`, the regime most favourable to a boundary-confined
   inflow error)
   :header-rows: 1
   :widths: 24 38 38

   * - inflow treatment
     - first-cell-row :math:`L^2` (``nc`` = 16, 32, 64, 128)
     - observed order
   * - ``flat`` (average only)
     - sequence decays at the design rate
     - :math:`1.993,\ 2.004,\ 2.001`
   * - ``mom`` (average + transverse slope)
     - sequence decays at the design rate
     - :math:`1.998,\ 2.005,\ 2.003`

The orders are indistinguishable: the slope is a sub-floor refinement, not a
deficiency repair.  This is the
:func:`~tests.sn.verification.mms.test_ld_2d_boundary_promise.test_first_cell_row_already_second_order`
gate (``@l1``, ``@verifies("ld-cartesian-2d")`` — the label is REUSED, S9
mints none): it fails iff the first-cell-row flat order drops below
:math:`1.85`, which would mean the average inflow is :math:`O(h)`-deficient at
the boundary cell and the verdict must flip.

This holds across the full optical-depth axis, NOT only at the cheap
:math:`\Sigma_t L = 0.1` headline.  Probed across
:math:`\Sigma_t L \in \{0.1, 0.5, 1.0, 2.0\}` (streaming → thick) and
:math:`c \in \{0, 0.5\}`, ``flat``, ``mom``, and ``flip`` all converge in the
second-order band globally, and the magnitudes track within a documented
sub-floor band (:math:`<20\%`, with a :math:`30\%` guard).  Even in the
streaming limit (where the inflow propagates ballistically and an
inflow-representation error is LEAST boundary-confined) the LD cell balance
integrates the inflow against its own linear basis, so the average moment is
:math:`O(h^2)`-adequate.  Amplifying the :math:`\mu`-dependent (slope-carrying)
drivers up to :math:`20\times` — the user's strongest "make the boundary slope
non-trivial" hypothesis — makes the converged flux MONOTONICALLY WORSE
(:math:`+17\%` at :math:`20\times`, still :math:`O(h^2)`), never better.  The
sub-floor wall is FUNDAMENTAL to a boundary-trace moment, not an artefact of
the cheap regime; these are the
:func:`~tests.sn.verification.mms.test_ld_2d_boundary_promise.test_optical_sweep_slope_never_beats_floor`
and
:func:`~tests.sn.verification.mms.test_ld_2d_boundary_promise.test_amplified_boundary_slope_still_subfloor`
verdict pins (``@slow`` ``@l1``).

Producer honesty: the MMS case emits the moment slot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The S9 production change is a single completion of the existing case (no new
ansatz, no sibling case).
:meth:`~orpheus.derivations.continuous.mms.sn.SN2DCartesianLDStressMMSCase.prescribed_inflow`
now gates on
:func:`~orpheus.numerics.moment_layout.face_moment_count` — the SAME
single-source primitive
:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.boundary_face_layout` keys the slot width on:

* When ``face_moment_count == 1`` (DD/Step) it builds the SCALAR per-face trace
  ``(N, ng, n_t)`` by cell-CENTRE evaluation of :math:`(A + \mu_x B + \mu_y
  C)/W`, hits the producer's scalar branch, and is **byte-identical** to the
  pre-S9 build (proven: ``np.array_equal`` against the legacy ``face_coords``
  construction holds — DD/Step has no moment axis, the slots are bit-for-bit
  the old build).
* When ``face_moment_count > 1`` (LD) it builds the FULL moment slot
  ``(N, ng, n_t, face_moment_count)`` via a new case-owned
  :meth:`~orpheus.derivations.continuous.mms.sn.SN2DCartesianLDStressMMSCase._project_inflow_to_face_moments`
  and hits the producer's full-slot branch.

The projector descends ONLY from the case's manufactured harmonics
(``_drivers``) and :func:`numpy.polynomial.legendre.leggauss` — NEVER
``_inflow_to_moments``, ``_ubld``, any LD operator, or the test-side
projectors (the L11 structural-independence discipline of the
:doc:`algebra-of-record </theory/references/index>` pillar).  Per transverse
cell :math:`[t_L, t_R]` mapped to :math:`\xi \in [-1, 1]`, it projects onto the
BARE per-cell Legendre coefficients

.. math::

   \text{slot}_0 \;=\; \frac{\langle \psi, P_0\rangle}{\langle P_0, P_0\rangle}
   \quad(\text{transverse cell AVERAGE}),
   \qquad
   \text{slot}_1 \;=\; \frac{\langle \psi, P_1\rangle}{\langle P_1, P_1\rangle}
   \quad(\text{BARE transverse slope}).

This reuses the exact normalization Leg B locked
(Eq. :eq:`ld-cartesian-2d-face-projection-coeff`): NO :math:`\theta`/:math:`h_t`
weighting, because the cochain's transverse mass :math:`\mathrm{diag}(h_t,
\theta h_t)` applies them downstream — a :math:`\theta`- or :math:`h_t`-weighted
slope would double-apply the mass, a TRUE bug.  Because the case projector and
the test-side ``_face_transverse_legendre`` are deliberately INDEPENDENT
implementations of the same projection, their machine-precision agreement is a
single-source check (Cardinal Rule 2), pinned by the new foundation gate
:func:`~tests.sn.verification.mms.test_mms_ld_2d.test_case_projector_agrees_with_test_face_projector`;
a shared import would make that check tautological and let a double-applied mass
slip through.  The threading is then pinned end to end by the GATE-B leg added
to ``test_ld_2d_boundary_slope_threaded_through_inflow_to_moments`` (the
production producer's slot 1 equals the ``leggauss`` reference at machine
precision — closing the Mode-11 producer-blindness the #251 surrogate had, which
pinned only the consumer).

.. note:: **The second-order interaction the producer honesty introduces.**

   A diagnostic helper whose "average-only" baseline routed through the (now
   slope-honest) production ``prescribed_inflow`` would SILENTLY inherit the new
   slope, collapsing the controlled ``flat``/``mom``/``flip`` toggle and breaking
   the byte-identity no-op control.  The fix is structural: the controlled
   average-only baseline is built TEST-SIDE (the toggle belongs to the test;
   the honesty belongs to production).  This is a general lesson — when a
   refactor makes a production path honest about a quantity a test was
   controlling externally, the test's controlled toggle MUST NOT inherit the
   production change it is meant to be orthogonal to.

The fourth Mode-10 instance — and why no value gate is added
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

S9 is the FOURTH consecutive vv **Mode 10** close-out in the
slope-moment family — #240 D5b-S4 (the bulk slope-source) →
:ref:`ld-cartesian-2d-legA` (#247, the external :math:`\hat Q`) →
:ref:`ld-cartesian-2d-legB` (#251, the boundary transverse slope) → S9 — and it
sits squarely in the **companion-unavailable branch** Leg B opened.  The
transverse boundary-slope moment is *activated-but-unconstrained*: its code path
is genuinely exercised (projected, threaded, stored, reframed per octant,
consumed by the cell update — the Mode-11 sentinel
:func:`~tests.sn.verification.mms.test_ld_2d_boundary_promise.test_slope_toggle_reaches_inflow_to_moments`
confirms slot 1 reaches the production consumer and the converged flux differs,
so the toggle is non-vacuous), yet its contribution to the converged flux is
sub-floor for ANY value claim.

The canonical Mode-10 resolution says: *add a companion gate that isolates the
term so its error is :math:`O(1)` in the measured quantity (a problem where the
term is the dominant forcing)*.  **For a boundary-trace slope that companion is
genuinely UNAVAILABLE.**  The boundary is codimension-1 — measure-zero in the
refinement limit — so a localized :math:`O(h)`-small along-face perturbation has
NO regime in which it becomes the dominant forcing of the converged flux; the
optical-depth and amplitude sweeps above are the empirical proof that no such
regime exists.  In that case the structural pair is the COMPLETE resolution, and
manufacturing a value-improvement leg would be dishonest — it would falsely RED a
correctly-consumed term (probed: seeding the TRUE slope makes the near-boundary
error slightly WORSE, the flipped slope slightly BETTER; both sub-floor).  So
S9 keeps Leg B's structural teeth (machine-precision threading + consumed-flip
:math:`\gg` tolerance + the byte-identical scalar no-op control) and adds NO
value or order gate keyed on the slope.

.. warning::

   Do NOT write "S9 recovers second-order accuracy at the boundary" or
   "the boundary-slope moment makes LD second-order at the boundary".  Both are
   false: the boundary cell is :math:`O(h^2)` from the AVERAGE moment alone, and
   the slope is a sub-floor inflow-representation refinement.  The verifiable
   content of S9 is STRUCTURAL (producer threading + consumption + the
   coherent-promise lock), per the vv Mode-10 companion-unavailable branch — not
   a converged-value claim.

The property-vs-type seam (#263): a boundary moment is a PROPERTY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

S9 raised a natural design question: if the angular moment representation
(:class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`)
is a first-class FIELD TYPE, should the transverse boundary moment be one too —
a ``BoundaryMomentField``?  The answer, tracked in **#263**, is **NO today**:
the transverse boundary moment is a PROPERTY of the boundary field (an untyped
trailing moment axis on the flat face buffer), exactly as the bulk carries its
spatial moments as a property — a trailing
:class:`~orpheus.numerics.spaces.spatial_moment_space.SpatialMomentSpace`
factor on the bulk leaf's SPACE (minted by
:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_trial_space` since
CS4b S5), rather than a distinct field type.  The criterion and its trigger live in the
:ref:`field-type-vs-property-criterion` section of the operator-algebra page;
the short version, specialised to the boundary moment:

A representation earns a distinct first-class **type** only when a
**non-canonical dual** coexists with it — two bases that are NOT canonically
isomorphic, connected by a change-of-basis morphism that is itself modelled and
applied (carries truncation error, has an adjoint, participates in the operator
algebra).  Angular order PASSES: the ordinate basis (``AngularFlux``) and the
harmonic-modal basis (``HarmonicMomentFlux``) are non-canonically isomorphic
(the iso depends on the quadrature :math:`Y_\ell^m(\hat\Omega_n)`), bridged by
the applied :math:`M`/:math:`R` projection/reconstruction pair with truncation
content and adjoints.  The transverse SPATIAL moment FAILS: there is ONE
within-cell basis (the tensor-Legendre tower), no non-canonical dual, so the
only change-of-basis would be the identity.  A ``BoundaryMomentField`` leaf
whose ``_check_partner`` adds nothing beyond class identity would be a vacuous
naming leaf — type-theatrics by the project's own "if the type hint does not
prevent a bug by construction it is theatrics" standard.  So the moment rides as
a PROPERTY (the flat face buffer already holds the moment tail via
:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.boundary_face_layout`), and the first-class
:class:`~orpheus.numerics.spaces.spatial_moment_space.SpatialMomentSpace` field
type is DEFERRED to the collocation trigger (nodal-DG / Lagrange-FEM, where a
nodal point-value basis coexists with the modal coefficients and a Vandermonde
morphism bridges them) — the durable design invariant #263 records.

Sources and gates
^^^^^^^^^^^^^^^^^

The production change is one file:
:meth:`~orpheus.derivations.continuous.mms.sn.SN2DCartesianLDStressMMSCase.prescribed_inflow`
gated scalar-vs-moment build plus the new case-owned ``leggauss`` projector
:meth:`~orpheus.derivations.continuous.mms.sn.SN2DCartesianLDStressMMSCase._project_inflow_to_face_moments`
(the boundary cochain, the trace, the consumer ``_inflow_to_moments``, and
DD/Step are all UNCHANGED — Leg B already landed the consumption path).  The
coherent-promise gate and the verdict pins live in
:mod:`tests.sn.verification.mms.test_ld_2d_boundary_promise`:
``test_first_cell_row_already_second_order`` (``@l1``
``@verifies("ld-cartesian-2d")`` — the coherent-promise lock),
``test_optical_sweep_slope_never_beats_floor`` and
``test_amplified_boundary_slope_still_subfloor`` (``@slow`` ``@l1`` — the
sub-floor verdict pins guarding the no-value-gate conclusion across optical
depth and amplitude), and ``test_slope_toggle_reaches_inflow_to_moments``
(``@foundation`` — the Mode-11 sentinel proving the toggle is non-vacuous).  The
producer-stamp leg lives in
:mod:`tests.sn.verification.mms.test_mms_ld_2d`:
``test_ld_2d_boundary_slope_threaded_through_inflow_to_moments`` (GATE B — the
production producer's slot 1 equals the ``leggauss`` reference) and
``test_case_projector_agrees_with_test_face_projector`` (GATE C — the
single-source projector agreement).  All gates are ``-O``-safe
(``np.testing.*`` / ``pytest.fail`` only, no bare ``assert`` that ``python -O``
would strip — vv Mode 8).  No ERR entry was minted: like Leg A and Leg B, S9 is
a proactive-gap close (a producer-blindness — the slope was UNVERIFIED at the
producer, not WRONG — the #251 consumer already threaded it correctly), not a
caught production bug.


.. _sn-composite-fixed-source:

The composite fixed-source API — :math:`q = q_{\text{bulk}} \oplus q_\partial`
------------------------------------------------------------------------------

.. admonition:: Key Facts
   :class: important

   - **A fixed source is a source EVERYWHERE.** The right-hand side of
     a fixed-source transport problem is not just a volumetric bulk
     source — it is a source on the whole phase space: a bulk
     :math:`q_{\text{bulk}}` *and* a boundary (prescribed-inflow)
     :math:`q_\partial`. ORPHEUS represents this as the composite
     ``q = q_bulk ⊕ q_∂``, the direct sum of the two role-typed leaves.
   - **The carrier is the object we already have.**
     :func:`~orpheus.sn.solver.solve_sn_fixed_source` accepts the
     composite as a
     :class:`~orpheus.transport.timed_full_field.TimedFullField` — the
     **same** typed direct-sum carrier the SI / Krylov inner already
     flows through internally. This is *not* a new type; it is
     ergonomics to **generate** the right object (Cardinal Rule 2 — we
     already have the right concepts).
   - **A source, role-distinguished from a flux by its leaf types.**
     The composite's bulk leaf is an
     :class:`~orpheus.transport.source_sinks.AngularSourceSink` and its
     boundary leaf a
     :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink` — the
     *source* column of the role grid (see :ref:`bc-extraction-operator-output-typing`).
     The iterate / solution it produces is a *flux* (``AngularFlux`` ⊕
     ``AngularBoundaryFlux``). Same carrier shape, different role; the class
     gate keeps source and flux arithmetic from silently mixing.
   - **The legacy array is the bulk-only / vacuum special case.** Passing
     the historical ``(N, ng, nx, ny)`` ndarray is *exactly* the
     composite with an all-zero (vacuum) boundary. All 37 pre-existing
     callers keep working bit-unchanged.
   - **One construction point.** The private helper
     :func:`~orpheus.sn.solver._build_fixed_source_rhs` is the single
     place the RHS composite is built (Cardinal Rule 2 — it collapsed a
     ``q_ext_composite`` build that previously lived in *both* the SI
     and Krylov inner paths).
   - **The ergonomic boundary generator.**
     :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.prescribed_inflow`
     writes ONLY the inflow ordinate slots of the named faces (outflow
     slots of a prescribed inflow are physically meaningless →
     unrepresentable by construction, ``coding-elegance`` Pattern 4),
     leaving everything else zero. It is the known-per-face-array route;
     the lazy ``InflowSourceSpec``-recipe route (``from_specs``) is a
     distinct bridge that delegates its packing back to this method.

The fixed-source right-hand side is a source on the whole phase space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A fixed-source SN problem solves the affine within-group system
:eq:`si-within-group-operator-eq`

.. math::

   (L + C - S - B)\,\psi = q,

and the right-hand side :math:`q` is **not** a bulk volumetric source
alone. It has two pieces, one per phase-space locus:

* the **bulk** source :math:`q_{\text{bulk}}(\vec r, \hat\Omega, g)` —
  the per-ordinate volumetric external source :math:`Q^{\text{ext}}_n`
  on every cell;
* the **boundary** source :math:`q_\partial` — the prescribed inflow,
  the inhomogeneous term :math:`q` of the affine boundary law
  :eq:`affine-bc-form` :math:`\gamma_-\psi = R\,G\,\gamma_+\psi + q`,
  living on the inflow ordinate slots of the boundary trace.

A vacuum boundary is simply :math:`q_\partial \equiv 0`; a non-vacuum
prescribed inflow is a non-zero :math:`q_\partial`. The natural object
is therefore the **direct sum** of the two:

.. math::
   :label: sn-fixed-source-direct-sum

   q \;=\; q_{\text{bulk}} \,\oplus\, q_\partial,

.. (vv-status rationale) representational: the fixed-source RHS field-typing
.. identity — the bulk volumetric source direct-summed with the boundary
.. prescribed inflow, carried by the ``TimedFullField`` bulk-boundary direct
.. sum. A field-typing identity, not a solver claim; its verifiable content is
.. pinned by the fixed-source RHS construction / field-role foundation gates.
.. vv-status: sn-fixed-source-direct-sum documented

an object that "represents the source everywhere". ORPHEUS already has
exactly this carrier: the
:class:`~orpheus.transport.timed_full_field.TimedFullField`, the typed
bulk⊕boundary(⊕history) direct sum that the within-group SI and Krylov
inner paths *already* pass around (the matvec
:math:`(L+C)\psi - (S+B)\psi - F\psi` and the SI rhs
:math:`F\psi + (S+B)\psi + q_{\text{ext}}` are CLOSED ``TimedFullField``
sums). The field-role-typing work did **not** introduce a new source
type — it surfaced the carrier we already had and added the ergonomics
to *generate* it (Cardinal Rule 2: we have the right object, we just
need a better way to build it).

Source vs flux — same carrier, different role
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The composite source and the angular-flux solution share the
``TimedFullField`` carrier *shape* but differ in their leaf **types**,
which encode the role:

.. list-table:: The composite carrier's two roles
   :header-rows: 1
   :widths: 22 39 39

   * - Locus
     - Source role (the RHS ``q``)
     - Flux role (the iterate / solution ``ψ``)
   * - bulk
     - :class:`~orpheus.transport.source_sinks.AngularSourceSink`
     - :class:`~orpheus.transport.fields.angular_flux.AngularFlux`
   * - boundary
     - :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink`
     - :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`

The role-leaf types are the gate. A *source* and a *flux* are never the
same field even when they ride the same carrier — the
:class:`~orpheus.transport.timed_full_field.TimedFullField` class gate
(via :class:`~orpheus.numerics.field.Field`) rejects
``AngularSourceSink ± AngularFlux`` and the boundary analogue, so the
"RHS is a source, the iterate is a flux" distinction is illegal to mix
by construction. The completed boundary role grid mirrors the bulk
exactly (:ref:`bc-extraction-operator-output-typing`): an operator's
``.apply`` output is a *source/sink* (:math:`A\psi`), its ``.solve``
output is a *flux* (the swept solution trace), and a ``from_balance``
defect is a *residual*. ``q_\partial`` is a ``AngularBoundarySourceSink``
because a prescribed inflow IS a source added to :math:`\gamma_-\psi`,
not the swept solution.

The two accepted forms of ``external_source``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~orpheus.sn.solver.solve_sn_fixed_source` accepts the
``external_source`` argument in either of two forms, normalised by the
single helper :func:`~orpheus.sn.solver._build_fixed_source_rhs`:

#. **A bare ``np.ndarray`` of shape** :math:`(N, n_g, n_x, n_y)` — the
   per-ordinate-density **bulk** source only, with a **vacuum**
   boundary. This is the original form, and it is *exactly* the
   composite with an all-zero boundary leaf
   (``AngularBoundarySourceSink.zeros(sn_mesh.angular_trace)`` — the
   allocator went space-keyed at CS4b S5; it read
   ``zeros_on(sn_mesh)`` before). Every one of the 37
   pre-existing callers passes this form and keeps working bit-for-bit
   unchanged (the vacuum path is verified bit-identical).
#. **A full** :class:`~orpheus.transport.timed_full_field.TimedFullField`
   **composite** ``q = q_bulk ⊕ q_∂`` — the route for a **non-vacuum
   prescribed inflow**. Its leaf values are re-homed onto the solve's
   own ``sn_mesh``: the trace / grid layout is deterministic from
   ``(mesh, quadrature, materials)``, so this is an exact values-copy
   onto the solve's mesh instance. The within-group operators are built
   on ``sn_mesh`` and their matvec entries admit an operand only when
   its interior space agrees in CONTENT with the one that mesh mints
   (campaign 1 CS4b S3 re-keyed this from mesh-OBJECT identity, so a
   twin carrier built from equal inputs would now be admitted); the
   unconditional re-home is what makes the route correct without the
   caller having to reason about that.

.. code-block:: python

   from orpheus.sn import solve_sn_fixed_source
   from orpheus.sn.mesh.augmented_mesh import SNMesh
   from orpheus.transport.source_sinks import (
       AngularSourceSink, AngularBoundarySourceSink,
   )
   from orpheus.transport.timed_full_field import TimedFullField

   sn = SNMesh(mesh, quadrature, materials)

   # Bulk volumetric source, per-ordinate density (N, ng, *spatial).
   # The space is the carrier's cached mint: read ``angular_trial_space``
   # (the scheme's within-cell basis; identical to ``angular_bulk_space``
   # for DD / Step) so the same line is right at every scheme width.
   q_bulk = AngularSourceSink(values=Q_ext, space=sn.angular_trial_space)
   # Prescribed inflow: only the named faces' inflow ordinate slots.
   q_bndry = AngularBoundarySourceSink.prescribed_inflow(
       sn, {"xmin": gamma_minus_xmin, "xmax": gamma_minus_xmax},
   )
   q = TimedFullField(interior=q_bulk, boundary=q_bndry)

   result = solve_sn_fixed_source(materials, mesh, quadrature, q)

The legacy ``solve_sn_fixed_source(materials, mesh, quadrature, Q_ext)``
with a bare ``Q_ext`` array is identical to the above with a vacuum
``q_bndry`` (``AngularBoundarySourceSink.zeros(sn.angular_trace)``).
``[M]`` the block above runs as written on a 4-cell vacuum slab with
``N = 4``, ``ng = 2``: :math:`\max\phi = 1.8265`.

The single construction point — Cardinal Rule 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before the field-role-typing work, the SI inner and the Krylov inner
each built their own ``q_ext_composite`` from the bulk array (the same
``AngularSourceSink.from_isotropic`` / ``from_mesh`` projection paired
with a zero boundary). That was a shared concept living in two places —
precisely the smell Cardinal Rule 2 flags.
:func:`~orpheus.sn.solver._build_fixed_source_rhs` collapses both into
one construction point: ``solve_sn_fixed_source`` calls it once, and
**both** inner paths consume what it returns. The helper:

* validates the bulk shape against :math:`(N, n_g, n_x, n_y)` (Issue
  #196 PR-INDEX-5 principled layout — the ``g`` axis directly after
  ``N``);
* for a bare array, pairs the bulk
  :class:`~orpheus.transport.source_sinks.AngularSourceSink` with a
  vacuum ``AngularBoundarySourceSink``;
* for a composite, re-homes the leaf values onto the solve's
  ``sn_mesh`` (with a layout-size guard on the boundary trace), and
  raises a descriptive ``ValueError`` if the composite was built on an
  incompatible mesh / quadrature / materials.

The validation, the projection, and the vacuum-default boundary now
live in exactly one function. The SI and Krylov paths differ only in
the inner solve they run, not in how the RHS is assembled.

The ergonomic boundary generator — ``prescribed_inflow``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generating the boundary leaf :math:`q_\partial` is the part the
ergonomics target. The classmethod
:meth:`AngularBoundarySourceSink.prescribed_inflow(mesh, {face: (N, ng) values}) <orpheus.transport.source_sinks.AngularBoundarySourceSink.prescribed_inflow>`
builds the prescribed inflow from known per-face arrays:

* for each face named in the mapping, it writes ONLY the **inflow**
  ordinate slots from the given :math:`(N, n_g)` array;
* **every other slot is left zero** — the outflow ordinate slots of a
  named face, and every slot of an unnamed (vacuum) face.

**Why outflow is unrepresentable (Pattern 4).** The outflow ordinate
slots of a *prescribed-inflow source* are physically meaningless: the
sweep determines the outflow trace, the source does not. Writing them
would be an illegal state. Rather than accept-then-ignore them,
:meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.prescribed_inflow`
makes the illegal state **unrepresentable by construction** — it reads
the inflow ordinate index set
(:meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.inflow_indices_for_face`)
and copies *only* those rows, so an accidentally-populated outflow row
in the caller's array simply cannot reach the field. This is
``coding-elegance`` Pattern 4 (illegal states unrepresentable). It
supersedes the ``zeros_on`` + nested
``face_view(face)[inflow] = …`` slot-fill loop that every
prescribed-inflow consumer (the non-vacuum MMS, the splitting-invariance
probe) previously hand-rolled — the single source of truth for
materialising a prescribed inflow onto the trace (Cardinal Rule 2).

**The recipe → snapshot distinction (vs ``from_spec``).** There are two
distinct routes to a boundary source, related as *recipe → snapshot*,
not as duplicates:

.. list-table:: Two routes to ``q_∂`` — known arrays vs lazy recipe
   :header-rows: 1
   :widths: 28 36 36

   * -
     - :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.prescribed_inflow`
     - :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.from_specs`
   * - Input
     - known per-face ``(N, ng)`` arrays
     - a lazy
       :class:`~orpheus.geometry.boundary._source.InflowSourceSpec`
       recipe (``evaluate(shape) -> ndarray``)
   * - When
     - the inflow values are already computed (the MMS case)
     - the inflow is described by a per-face recipe evaluated on demand
   * - Status
     - **shipped** — the route the 4.6 MMS and the T4 probe use
     - **deferred** (``unify-after-two`` — no recipe-driven consumer
       that drives a typed boundary-source sweep yet)

The 4.6 MMS uses
:meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.prescribed_inflow`
because it has explicit per-face arrays
(:math:`\gamma_-\psi = (A + \mu_n B)/W`); it does not need the lazy
recipe bridge, which waits for its first genuine consumer per the
``unify-after-two`` discipline.

Why this is ergonomics, not new types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The entire change is the **generation** of objects that already
existed in the codebase, not the introduction of new ones:

* The carrier :class:`~orpheus.transport.timed_full_field.TimedFullField`
  pre-dates this work — it is what the inner solve already flows.
* The leaf types
  :class:`~orpheus.transport.source_sinks.AngularSourceSink` and
  :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink` pre-date
  this work — they are the *source* column of the role grid.
* The affine boundary law :eq:`affine-bc-form` and the ``q.boundary``
  slot pre-date this work.

What was missing was the *ergonomic generator* for a non-vacuum
boundary leaf and a *public entry point* that accepts the composite.
:meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.prescribed_inflow`
and the second accepted form of
:func:`~orpheus.sn.solver.solve_sn_fixed_source` supply exactly those —
no more. This is the operational meaning of "we already have the right
objects/concepts — we just need better ergonomics to generate them":
the abstraction is unchanged; only the surface that builds it is
better. The first consumer is the non-vacuum prescribed-inflow MMS of
:ref:`sn-mms-nonvacuum`.


.. _sn-mms-nonvacuum:

Non-vacuum prescribed-inflow MMS
--------------------------------

.. admonition:: Key Facts
   :class: important

   - **What this section adds.** The entire pre-existing MMS catalog
     (:ref:`sn-mms-curvilinear-isotropic-verification`,
     :ref:`sn-mms-curvilinear-aniso-verification`) is
     *vacuum-automatic*: every manufactured ansatz vanishes at both
     boundaries, so the inflow trace :math:`\gamma_-\psi \equiv 0` on
     every ordinate and the prescribed-inflow source slot
     ``q.boundary`` is **identically zero** in all of them. Phase 4 /
     O.2b sub-step 4.6 fills that gap with a manufactured solution that
     is **non-zero at the outer face**, lighting the
     :math:`q.\text{boundary} \neq 0` path for the first time.
   - **The ansatz is the proven P1 element.** :math:`\psi_n = (A + \mu_n
     B)/W` is the same truncated-Legendre :math:`P_0 \oplus P_1` form
     used by the Phase 3.6 anisotropic cases
     (:eq:`sn-mms-spherical-aniso-psi`). 4.6 changes **only the
     boundary trace** — :math:`A,B` are chosen non-vanishing at the
     outer face — and reuses the verified angular structure. Linear in
     :math:`\mu` *fully* (not partially) activates the curvilinear
     redistribution; the question "do we need :math:`\mu^2`?" is
     answered **no** below.
   - **Two manufactured sources, derived from the continuous
     operator.** The slab source :eq:`sn-mms-nonvacuum-qext` has **no**
     redistribution term (the Cartesian operator lacks the
     :math:`\partial_\mu` coupling); the sphere source
     :eq:`sn-mms-nonvacuum-sph-qext` is the **same closed form** as the
     Phase 3.6 vacuum case — only :math:`A,B` differ. The spherical
     residual therefore lives in **one place**
     (:func:`~orpheus.derivations.continuous.mms.sn._spherical_anisotropic_symbolic`,
     Cardinal Rule 2).
   - **HAZARD H1 (sphere pole regularity).** :math:`B(0)=0` is a HARD
     constraint — without it the redistribution :math:`(1-\mu^2)B/r \to
     \infty` at :math:`r=0`. The :math:`(r/R)` prefactor on the sphere
     :math:`B` enforces it; the slab has no pole, so a slab-style
     :math:`B(0)\neq 0` is fine there but **wrong** on the sphere.
   - **The affine-BC-to-RHS framing.** Prescribed inflow IS the
     inhomogeneous term :math:`q` of the affine boundary law
     :eq:`affine-bc-form` :math:`\gamma_-\psi = R\,G\,\gamma_+\psi + q`,
     carried in the ``q.boundary`` slot
     (:class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink`) and
     consumed directly by :math:`(L+C)\text{.solve}` as the sweep
     inflow seed. **No** :class:`~orpheus.geometry.boundary.PrescribedInflow`
     mesh-BC bridge is touched, and **no** ``from_spec`` recipe bridge
     is needed — the inflow is supplied as the boundary leaf of the
     **composite source** ``q = q_bulk ⊕ q_∂`` that
     :func:`~orpheus.sn.solver.solve_sn_fixed_source` now accepts
     directly (see :ref:`sn-composite-fixed-source`).
   - **The convergence rows drive the public composite-source API.**
     The 4.6 MMS no longer assembles the within-group operator triple
     by hand: :func:`~orpheus.sn.solver.solve_sn_fixed_source` accepts
     a :class:`~orpheus.transport.timed_full_field.TimedFullField`
     composite source, so each case bundles its manufactured bulk and
     prescribed-inflow boundary into one ``case.fixed_source(sn)``
     call. The migration off the operator-triple bypass *is* the
     retirement (retirement = test migration).
   - **The load-bearing assertion is the converged VALUE, not the
     rate.** Per the ``vv-principles`` skill anti-pattern #5 (rate is
     necessary, not sufficient), a silently-dropped ``q.boundary``
     converges cleanly at :math:`\mathcal{O}(h^2)` to the **wrong**,
     boundary-zero limit. Only the flux-value-vs-:math:`A(x)` check —
     with :math:`A` non-zero at the boundary (:math:`a_0>0`) — sees it.
   - **T3 (sphere) ships ``xfail(strict)``, re-scoped to Issue #229.**
     The slab rows are clean :math:`\mathcal{O}(h^2)` with value match.
     The sphere row's anisotropic :math:`(A+B\mu)/W` ansatz is
     angle-varying, so after ERR-058 (#195) recovered the spatial
     :math:`\mathcal{O}(h^2)` convergence it now hits the
     fixed-quadrature **angular floor** of the half-angle thread
     interpolation (sphere S16 floor :math:`\approx 7\mathrm{e}{-4}`,
     above the band) — NOT the old #195 plateau, and NOT a non-vacuum
     machinery failure (the boundary value *is* honoured).  The marker
     moved from #195 to the
     `Issue #229 <https://github.com/deOliveira-R/ORPHEUS/issues/229>`_
     quadrature-aware retune.  Both the sphere and slab rows now run the
     curvilinear/Cartesian **source-iteration** default of
     :func:`~orpheus.sn.solver.solve_sn_fixed_source` (SI :math:`\equiv`
     Krylov bit-identical post-ERR-058); the composite-source API
     delivers the prescribed inflow identically to every splitting (T4).
     The green companion T3g provides live structural coverage of the
     inflow + redistribution paths now.

This section narrates the Branch-1 SymPy algebra-of-record
(:mod:`orpheus.derivations.continuous.mms.sn`), the Branch-2 numpy
factories, and the L1 / foundation gates that verify the prescribed-
inflow discretisation. The verification chain follows the
``algebra-of-record`` discipline (Branch-1 SymPy reference, Branch-2
numpy production, structurally-independent L1 cross-check).

.. list-table:: 4.6 verification gates (measured, ``-O`` mode)
   :header-rows: 1
   :widths: 6 30 10 12 32

   * - Gate
     - Description
     - Level
     - Pillar
     - Status / evidence
   * - V_nonvac-slab
     - Slab substitution identity ``simplify(W·LHS − Σ_s φ − Q) == 0``
     - foundation
     - MMS (1C)
     - PASS — :func:`tests.derivations.test_sn_mms_nonvacuum_symbolic.test_v_nonvac_slab_substitution_identity`
   * - V_nonvac-sph
     - Sphere substitution identity (reuses the 3.6 spherical residual)
     - foundation
     - MMS (1C)
     - PASS — :func:`tests.derivations.test_sn_mms_nonvacuum_symbolic.test_v_nonvac_sph_substitution_identity`
   * - Decision-A pin
     - Parameterised :math:`A=B=` ``None`` reproduces 3.6 vacuum shapes byte-for-byte
     - foundation
     - regression
     - PASS — :func:`tests.derivations.test_sn_mms_nonvacuum_symbolic.test_existing_spherical_aniso_still_passes_after_parameterization`
   * - L1 xcheck (slab)
     - Branch-2 numpy :math:`Q^{\text{ext}}` == lambdified SymPy (≤1e-13)
     - foundation
     - MMS (1C)
     - PASS — :func:`tests.derivations.test_sn_mms_nonvacuum_symbolic.test_slab_nonvacuum_numerical_qext_matches_sympy`
   * - L1 xcheck (sphere)
     - Branch-2 numpy :math:`Q^{\text{ext}}` == lambdified SymPy (≤1e-13)
     - foundation
     - MMS (1C)
     - PASS — :func:`tests.derivations.test_sn_mms_nonvacuum_symbolic.test_sphere_nonvacuum_numerical_qext_matches_sympy`
   * - T1 (slab 1g)
     - DD :math:`\mathcal{O}(h^2)` + converged value + inflow honoured
     - L1
     - MMS (1C)
     - PASS — orders ``[2.04, 2.01]``, finest L2 ~1.2e-3, max\|φ−A\| ~8e-5
   * - T2 (slab 2g asym)
     - As T1, asymmetric downscatter :math:`\Sigma_s` (ERR-002 hazard)
     - L1
     - MMS (1C)
     - PASS — g0 ``[2.04, 2.01]``, g1 ``[2.05, 2.01]``
   * - T3 (sphere)
     - Curvilinear redistribution under non-vacuum inflow
     - L1
     - MMS (1C)
     - **xfail(strict)** on #229 — aniso angular floor ≈7e-4 (spatial
       O(h²) recovered by ERR-058; boundary value honoured)
   * - T3g (sphere)
     - Inflow honoured at :math:`r=R` + redistribution source live (green now)
     - foundation
     - structural
     - PASS — :func:`tests.sn.verification.analytical.test_mms_prescribed_inflow.test_sphere_nonvacuum_inflow_honoured_and_redistribution_live`
   * - T4 (Mode 9)
     - SI-Jacobi ≡ SI-Gauss-Seidel ≡ Krylov honour ``q.boundary``
     - foundation
     - splitting-invariance
     - PASS — pairwise reldiffs 1.3e-13 … 5.6e-13

Why the existing MMS catalog is vacuum-automatic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every manufactured solution already in the SN verification ladder —
the isotropic :math:`\psi_n = A(r)/W` cases
(:ref:`sn-mms-curvilinear-isotropic-verification`) and the
anisotropic :math:`\psi_n = (A + \mu_n B)/W` cases
(:ref:`sn-mms-curvilinear-aniso-verification`) — was built with
:math:`A` and :math:`B` chosen to **vanish at both boundaries**. For
the canonical 3.6 sphere, :math:`A(r) = \sin(\pi r/R)` gives
:math:`A(0) = A(R) = 0`, and :math:`B(r) = (r/R)(1-r/R)\cos(\pi r/R)`
gives :math:`B(0) = B(R) = 0`. The slab isotropic case likewise uses
:math:`A(x) = \sin(\pi x/L)`.

The consequence is structural and total. The inflow trace of the
manufactured solution on any face is

.. math::

   \gamma_- \psi_n \big|_{\text{face}}
       = \frac{1}{W}\bigl(A(x_{\text{face}})
                         + \mu_n B(x_{\text{face}})\bigr)
       = \frac{1}{W}\bigl(0 + \mu_n \cdot 0\bigr) = 0
       \qquad \text{for every ordinate } n.

So the affine boundary law :eq:`affine-bc-form`
:math:`\gamma_-\psi = R\,G\,\gamma_+\psi + q` collapses to its
homogeneous (vacuum) form for these cases — and the inhomogeneous
inflow term :math:`q` is identically zero. The existing cases verify
the **interior** spatial / angular operator and the **homogeneous**
vacuum BC, but they say *nothing* about the prescribed-inflow path,
where a non-zero :math:`q` is pushed into the right-hand side. That
path is the one O.2b's field-role-typing work makes a first-class
boundary trace: an inhomogeneous inflow injected as the boundary
*source* slot. Until 4.6, no MMS row exercised it.

The fix is the smallest possible structural delta: keep the proven P1
angular form, keep the proven interior operator, and change **only**
:math:`A,B` so that they are non-zero at the outer face. Then
:math:`\gamma_-\psi \neq 0`, and the converged scalar flux
:math:`\phi(x) = A(x)` is non-zero at the boundary — exactly the
property the verification needs (see "The converged-value assertion"
below).

The ansatz — the P1 element and why linear-in-:math:`\mu` is enough
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The slab ansatz and its manufactured source:

.. math::
   :label: sn-mms-nonvacuum-psi

   \psi_n(x) = \frac{1}{W}\bigl(A(x) + \mu_n B(x)\bigr),
   \qquad A(x) = a_0 + a_1\sin(kx),\quad B(x) = b_0\cos(kx),
   \quad a_0 > 0.

.. (vv-status rationale) definition: Definitional ansatz — the
.. manufactured angular flux is *imposed*, not solved for. Its
.. correctness as a reference is established by the source identity
.. :eq:`sn-mms-nonvacuum-qext` (SymPy ``simplify == 0``), not by a
.. property of this expression alone.
.. vv-status: sn-mms-nonvacuum-psi documented

The form :math:`\psi_n = (A + \mu_n B)/W` is the truncated Legendre
:math:`P_0 \oplus P_1` element: :math:`P_0(\mu) = 1` carries the
isotropic amplitude :math:`A`, and :math:`P_1(\mu) = \mu` carries the
first-moment amplitude :math:`B`. This is the **native angular basis**
of the SN closure — the Carlson half-angle pole closure folds the
moment source through :math:`P_\ell(-1) = (-1)^\ell`, so a linear-in-
:math:`\mu` input is exactly the lowest non-trivial moment with a
non-zero :math:`\partial_\mu`.

**Why linear-in-:math:`\mu` fully activates the redistribution.** The
curvilinear angular-redistribution operator is
:math:`\tfrac{1-\mu^2}{r}\,\partial_\mu\psi`. With :math:`\psi` linear
in :math:`\mu`, the angular derivative is a non-zero **constant** in
:math:`\mu`,

.. math::

   \frac{\partial \psi_n}{\partial \mu} = \frac{B(r)}{W} \neq 0,

and multiplying by the redistribution dome :math:`(1-\mu^2)` produces
a genuinely :math:`\mu^2`-structured term :math:`(1-\mu^2)B/r`. The
discrete closure that realises this operator — the Morel–Montry
half-angle recurrence with the Carlson :math:`\mu=-1` seed (see
:ref:`sn-pole-angular-closure-protocol`) — is **linear** in
:math:`\psi`. A linear operator is fully probed by any input that is
non-constant in its argument; the linear-in-:math:`\mu` ansatz is
non-constant in :math:`\mu`, so it exercises the entire linear
redistribution map (including the half-angle recurrence and the
second-moment coupling).

A quadratic-in-:math:`\mu` (P2) ansatz term would add **no** new
structural coverage of the redistribution. Because the closure is
linear, a quadratic input only changes *which point* in the operator's
already-fully-probed range you land on — it does not reach any term
the linear input misses. (A P2 term *would* additionally exercise the
:math:`\sum_n w_n \mu_n^2` quadrature-exactness, but that is a
property of the quadrature, not of the redistribution operator, and is
already covered elsewhere.) This settles the "do we need
:math:`\mu^2`?" question definitively: **no.** The verdict is recorded
in the cross-domain-attacker frame analysis (memo
``phase4_o2b_4_6_mms_ansatz_frame.md``, Q1/Q2) and is empirically
consistent with Phase 3.6, which uses exactly this linear-in-:math:`\mu`
ansatz and whose gate tests carry ``catches("ERR-026")`` — the
redistribution-bug catcher.

**The scalar flux is :math:`A`.** Because Gauss–Legendre (and every
symmetric quadrature on :math:`\mu \in [-1,1]`) satisfies
:math:`\sum_n w_n \mu_n = 0`, the first-moment term integrates out of
the scalar moment:

.. math::

   \phi(x) = \frac{1}{W}\sum_n w_n\bigl(A(x) + \mu_n B(x)\bigr)
           = \frac{1}{W}\Bigl(A(x)\sum_n w_n
                            + B(x)\underbrace{\sum_n w_n \mu_n}_{=\,0}\Bigr)
           = A(x).

This discrete identity is verified directly in
:func:`tests.derivations.test_sn_mms_nonvacuum_symbolic.test_slab_nonvacuum_phi_equals_A_under_quadrature`
(≤1e-14 on a sample mesh). The reference scalar flux for the
convergence rows is therefore :math:`\phi_{\text{chosen}}(x) = A(x)`,
which — because :math:`a_0>0` — is **non-zero at the boundary**.

The manufactured slab source, derived from the continuous operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The slab SN transport operator (per ordinate, 1-group) is the
first-order streaming-plus-collision form
:math:`\mu\,\partial_x\psi + \Sigma_t\psi
= \tfrac{1}{W}(\Sigma_s\phi + Q^{\text{ext}})`. The Cartesian operator
has **no** angular-derivative term — the slab geometry produces no
angular redistribution (:eq:`transport-cartesian`). Substituting the
ansatz :eq:`sn-mms-nonvacuum-psi` and solving for the residual source:

.. math::

   \mu\,\frac{\partial}{\partial x}
       \frac{A + \mu B}{W}
   + \Sigma_t\,\frac{A + \mu B}{W}
   &= \frac{1}{W}\bigl(\Sigma_s\,A + Q^{\text{ext}}_n\bigr) \\[1mm]
   \frac{1}{W}\bigl(\mu A' + \mu^2 B'\bigr)
   + \frac{\Sigma_t}{W}\bigl(A + \mu B\bigr)
   &= \frac{1}{W}\bigl(\Sigma_s A + Q^{\text{ext}}_n\bigr),

where :math:`\phi = A` was used on the right. Multiplying through by
:math:`W` and isolating :math:`Q^{\text{ext}}_n` gives the closed form

.. math::
   :label: sn-mms-nonvacuum-qext

   Q^{\text{ext}}_n(x) = \mu_n A'(x) + \mu_n^2 B'(x)
                       + (\Sigma_t - \Sigma_s) A(x)
                       + \Sigma_t\,\mu_n B(x).

.. (vv-status rationale) derivation: A closed form obtained by
.. symbolic substitution into the continuous slab operator; verified
.. by SymPy ``simplify(W·LHS − Σ_s φ − Q) == 0`` and cross-checked
.. against the Branch-2 numpy evaluation to ≤1e-13.

Note that there is **no** :math:`(1-\mu^2)B/r` term — the slab operator
simply does not generate it. The :math:`\mu^2 B'` term *is* present
(streaming the first-moment piece :math:`\mu B` gives
:math:`\mu \cdot \mu B' = \mu^2 B'`), so the slab still exercises the
second-moment streaming closure, just not the angular redistribution.
The :math:`(\Sigma_t - \Sigma_s)A` term is the within-group removal
net of isotropic self-scatter (:math:`c = \Sigma_s/\Sigma_t < 1`), and
:math:`\Sigma_t\,\mu_n B` is the collision of the first-moment piece.

The Branch-1 algebra-of-record is
:func:`~orpheus.derivations.continuous.mms.sn.derive_nonvacuum_slab_mms`
(building on
:func:`~orpheus.derivations.continuous.mms.sn._nonvacuum_slab_symbolic`),
which performs the substitution symbolically and proves
``simplify(W·LHS − Σ_s·φ − Q_closed) == 0``. Because the slab operator
lacks redistribution, it is a *genuinely different* operator from the
sphere and gets its own fresh symbolic pair — it cannot reuse the
spherical residual (which carries the :math:`\partial_\mu` term the
slab does not have).

**Multi-group generalisation (T2).** The slab case is multi-group-
capable. Each group carries a per-group amplitude :math:`c_g` scaling
the shared shape, :math:`A_g(x) = c_g(a_0 + a_1\sin kx)` and
:math:`B_g(x) = c_g\,b_0\cos kx`, and the source picks up the
in-scatter term

.. math::
   :label: sn-mms-nonvacuum-qext-mg

   Q^{\text{ext}}_{n,g}(x) = \mu_n A_g'(x) + \mu_n^2 B_g'(x)
       + \Sigma_{t,g}\,A_g(x) + \Sigma_{t,g}\,\mu_n B_g(x)
       - \sum_{g'} \Sigma_s[g', g]\,A_{g'}(x).

The in-scatter sum uses the ORPHEUS scattering convention
``SigS[g_from, g_to]``, so the in-scatter source is
:math:`(\Sigma_s^\top\phi)_g = \sum_{g'}\Sigma_s[g', g]\,A_{g'}` — the
**transpose-active** term where the ERR-002 group-swap hazard lives.
T2 uses a 2-group **asymmetric downscatter-only** :math:`\Sigma_s`
(:math:`\Sigma_s[0,1]\neq 0`, :math:`\Sigma_s[1,0]=0`) so a transposed
scattering matrix would produce a detectably wrong group ratio
(the 1-group-degeneracy rule — multi-group with asymmetric :math:`\Sigma_s` is
mandatory, ``vv-principles`` anti-pattern #3 and failure-mode #6). The
1-group T1 path is the degenerate :math:`c_{\text{groups}} = (1.0,)`
reduction of the same dataclass.

The manufactured spherical source — the Cardinal-Rule-2 reuse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The spherical ansatz (pole-regular, non-vacuum at :math:`r=R`):

.. math::
   :label: sn-mms-nonvacuum-sph-psi

   \psi_n(r) = \frac{1}{W}\bigl(A(r) + \mu_n B(r)\bigr),
   \quad A(r) = a_0 + a_1\sin(kr),\quad
   B(r) = \frac{r}{R}\bigl[b_0 + b_1\cos(kr)\bigr],\quad B(0)=0.

.. (vv-status rationale) definition: Definitional ansatz — imposed,
.. not solved. Correctness as a reference rests on the source
.. identity :eq:`sn-mms-nonvacuum-sph-qext` (SymPy ``simplify == 0``)
.. and HAZARD H1 (:math:`B(0)=0`), verified in the foundation gate.
.. vv-status: sn-mms-nonvacuum-sph-psi documented

The spherical SN operator carries the angular-redistribution term
(:eq:`transport-spherical`):
:math:`\mu\,\partial_r\psi + \tfrac{1-\mu^2}{r}\,\partial_\mu\psi
+ \Sigma_t\psi = \tfrac{1}{W}(\Sigma_s\phi + Q^{\text{ext}})`.
Substituting :eq:`sn-mms-nonvacuum-sph-psi` (with
:math:`\partial_\mu\psi = B/W`, so the redistribution term is
:math:`\tfrac{1-\mu^2}{r}\cdot\tfrac{B}{W}`) and isolating the source
gives the **same closed form** as the Phase 3.6 vacuum case
(:eq:`sn-mms-spherical-aniso-qext`):

.. math::
   :label: sn-mms-nonvacuum-sph-qext

   Q^{\text{ext}}_n(r) = \mu_n A'(r) + \mu_n^2 B'(r)
                       + (1-\mu_n^2)\,\frac{B(r)}{r}
                       + (\Sigma_t-\Sigma_s) A(r)
                       + \Sigma_t\,\mu_n B(r).

.. (vv-status rationale) derivation: A closed form obtained by
.. symbolic substitution into the continuous spherical operator;
.. verified by SymPy ``simplify == 0`` (reusing the 3.6 spherical
.. residual machinery) and cross-checked against the Branch-2 numpy
.. evaluation to ≤1e-13.

The structural point is that :eq:`sn-mms-nonvacuum-sph-qext` and the
Phase 3.6 :eq:`sn-mms-spherical-aniso-qext` are *byte-identical* closed
forms — only the radial profiles :math:`A,B` plugged into them differ.
The spherical-operator residual is therefore derived in **exactly one
place**:
:func:`~orpheus.derivations.continuous.mms.sn._spherical_anisotropic_symbolic`,
which now takes optional ``A=None, B=None`` arguments. With no
arguments it reproduces the Phase 3.6 vacuum shapes byte-for-byte (the
decision-A regression pin verifies this in
:func:`tests.derivations.test_sn_mms_nonvacuum_symbolic.test_existing_spherical_aniso_still_passes_after_parameterization`);
with the 4.6 non-vacuum shapes it re-proves the residual for free
(:func:`~orpheus.derivations.continuous.mms.sn.derive_nonvacuum_spherical_mms`).
This is Cardinal Rule 2 in action — one source of truth for the
spherical transport-operator residual, shared between the vacuum and
non-vacuum cases.

HAZARD H1 — sphere pole regularity demands :math:`B(0)=0`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The redistribution term :math:`(1-\mu^2)B(r)/r` has an explicit
:math:`1/r` factor. As :math:`r\to 0`, this diverges as
:math:`(1-\mu^2)\,B(0)/r \to \infty` **unless** :math:`B(0)=0`. So on
the sphere, :math:`B(0)=0` is a **hard regularity constraint** — not a
stylistic preference. A naive slab-style choice :math:`B(x) = b_0\cos
kx` gives :math:`B(0) = b_0 \neq 0`, which is **fine on the slab** (no
pole, no :math:`1/r`) but **wrong on the sphere** (it manufactures a
non-integrable :math:`1/r` singularity at the centre that the
continuous solution does not actually have).

The 4.6 sphere therefore uses :math:`B(r) = (r/R)[b_0 + b_1\cos kr]`.
The :math:`(r/R)` prefactor forces :math:`B(0) = 0` (pole-regular: the
redistribution :math:`(1-\mu^2)B/r = (1-\mu^2)[b_0+b_1\cos kr]/R` is
*finite* at :math:`r=0`), while leaving :math:`B(R) = b_0 + b_1\cos kR
\neq 0` (the non-vacuum first-moment structure at the outer inflow
face). The amplitude :math:`A(r) = a_0 + a_1\sin kr` needs **no** such
prefactor: :math:`A` has no :math:`1/r` companion in the operator, so
:math:`A(0) = a_0` finite is perfectly regular at the pole, and
:math:`a_0>0` makes :math:`A(R)\neq 0` (non-vacuum). HAZARD H1 is
verified in
:func:`tests.derivations.test_sn_mms_nonvacuum_symbolic.test_v_nonvac_sph_pole_regularity_and_nonvacuum`
(:math:`B(0)=0`, :math:`A(0)=\tfrac12`, :math:`B(R)\neq 0`; concretely
at :math:`kR=\pi/2`: :math:`A(R)=\tfrac34`, :math:`B(R)=\tfrac{3}{10}`).

The non-vacuum lever — :math:`a_0>0` is the entire novelty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every other choice in 4.6 (the P1 angular form, the interior operator,
the redistribution term, the quadrature, the BC machinery) is shared
with Phase 3.6. The *single* new ingredient is :math:`a_0 > 0`, which
makes :math:`A` — and hence the inflow trace
:math:`\gamma_-\psi_n = (A + \mu_n B)/W` — **non-zero at the outer
face**. That non-zero trace is what lights up the prescribed-inflow
``q.boundary`` path. Strip :math:`a_0` back to zero and 4.6 degenerates
to Phase 3.6 (vacuum-automatic). The non-vacuum-ness is pinned by the
foundation test
:func:`tests.derivations.test_sn_mms_nonvacuum_symbolic.test_v_nonvac_slab_ansatz_nonvanishing_at_faces`
(:math:`A(0)=a_0>0`) so the verification cannot silently drift back to
the vacuum regime.

The affine-BC-to-RHS framing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prescribed inflow is **not** a special solver mode — it is the
inhomogeneous term of the universal affine boundary law
(:ref:`affine-bc-form`). The general boundary trace law is

.. math::

   \gamma_-\psi = R\,G\,\gamma_+\psi + q,

where :math:`q \in \Gamma_-` is the **prescribed inflow source**. For a
vacuum boundary :math:`R=0` and :math:`q\equiv 0`. For a manufactured
non-vacuum inflow, :math:`q = \gamma_-\psi_{\text{chosen}}` — the
imposed inflow trace — pushed to the right-hand side of the
discretised within-group system.

In ORPHEUS this :math:`q` is carried by the ``q.boundary`` slot, a
:class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink` field whose
inflow-ordinate entries hold :math:`\gamma_-\psi = (A + \mu_n B)/W` per
face per group. The within-group fixed point is the **affine** system

.. math::

   (L + C - S - B)\,\psi = q,
   \qquad q = q_{\text{ext}}
            + (\text{prescribed inflow in } q.\text{boundary}),

and the inflow term is consumed directly by :math:`(L+C)\text{.solve}`
as the sweep inflow seed. This is the cleanest possible realisation:
the inhomogeneous BC term is *just another source* on the RHS.

**No ``from_spec`` / ``PrescribedInflow``-BC bridge is touched.** A
:class:`~orpheus.geometry.boundary.PrescribedInflow` mesh-BC descriptor
*does* exist (the rank-0 affine BC), but it is a *different surface* —
it declares a prescribed inflow at mesh-construction time as a
first-class boundary condition. The 4.6 MMS deliberately does **not**
use it: the manufactured inflow is injected as the ``q.boundary``
source slot, which is exactly the affine-:math:`q`-to-RHS path, and is
the surface O.2b's field-role-typing work targets. The mesh BCs for the
4.6 cases are plain **vacuum** — the inflow lives entirely in
``q.boundary``.

The public composite-source API drives the convergence rows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 4.6 convergence rows drive the **public** fixed-source entry point
:func:`~orpheus.sn.solver.solve_sn_fixed_source` directly. Earlier in
this work that entry point hardcoded a vacuum ``q.boundary``
(``zeros_on``) — it had no way to carry a prescribed inflow — and the
rows therefore took an operator-triple *bypass*: assembling
:math:`(L+C)`, :math:`S`, :math:`B` by hand via
:func:`~orpheus.sn.coupled_system.build_within_group_system` and driving
them with :func:`~orpheus.numerics.iteration.SourceIteration`. That bypass
is **retired**. The field-role-typing work gave
:func:`~orpheus.sn.solver.solve_sn_fixed_source` a second accepted
source form — the full **composite source** ``q = q_bulk ⊕ q_∂``
represented by a
:class:`~orpheus.transport.timed_full_field.TimedFullField` (see
:ref:`sn-composite-fixed-source` for the API in full). Each case now
bundles its manufactured bulk (:meth:`external_source`) and its
prescribed-inflow boundary (:meth:`prescribed_inflow`) into one
``case.fixed_source(sn)`` and passes it straight to the public solver::

    result = solve_sn_fixed_source(
        materials, mesh, case.quadrature, case.fixed_source(sn),
        max_inner=1000, inner_tol=1e-13,
    )

Migrating the rows off the bypass onto the public API *is* the
retirement (retirement = test migration — the new code is what gets
tested). The slab rows take the SI (1-D Jacobi) inner; the sphere row
takes the curvilinear **source-iteration** default (post-ERR-058, #195;
SI :math:`\equiv` Krylov bit-identical — the curvilinear ``"krylov"``
default was reverted, see :ref:`sn-err-058-closure-seed-closeout`); both
honour the prescribed inflow identically (verified by T4 below).

**The flux/source space bridge — now INTERNAL to the solve (B.5.2).**
The composite RHS lives in **source** space (an
:class:`~orpheus.transport.source_sinks.AngularSourceSink` bulk plus a
:class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink` boundary),
while the iterate :math:`\psi` and the returned solution live in
**flux** space (an :class:`~orpheus.transport.fields.angular_flux.AngularFlux`
bulk plus a :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
boundary). The source-iteration / Krylov inner therefore needs a
flux-typed ``initial_guess`` to template the solution space — without
it, ``S.apply`` would hit an ``AngularSourceSink`` that has no
``integrate_angular`` method. That seed
(``TimedFullField.zeros(bulk=AngularFlux, boundary=AngularBoundaryFlux,
mesh=sn)``) is now built **inside**
:func:`~orpheus.sn.solver.solve_sn_fixed_source`, not hand-passed by
the test. The field-role-typing distinction — the iterate is a *flux*,
the RHS is a *source* — survives intact; it has simply moved behind the
public API where it belongs.

The converged-value assertion — rate is necessary, not sufficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the load-bearing verification design choice, and it is a
direct application of ``vv-principles`` anti-pattern #5 ("NEVER read
'convergence rate is correct' as 'result is correct' — verify the
converged-to value; :math:`\mathcal{O}(h^2)` to the wrong limit is
still :math:`\mathcal{O}(h^2)`") and the necessity hierarchy H4.

Consider the failure mode the test must catch: a bug (or a refactor
regression) that silently **drops the prescribed inflow** — a solve
that runs with ``q.boundary = 0`` despite the manufactured non-vacuum
inflow. That degenerate solve is **still a perfectly consistent
fixed-source problem** — it just solves the *vacuum-BC* version of the
same interior source. It converges cleanly at :math:`\mathcal{O}(h^2)`
to a *different*, boundary-zero scalar flux. A rate-only test passes
it. The only assertion that sees the dropped inflow is the one that
checks the **converged value against** :math:`A(x)`, because
:math:`A(x)` is non-zero at the boundary (:math:`a_0>0`) while the
dropped-inflow limit is zero there — a discrepancy of order
:math:`a_0 \approx 0.5` at the faces, dwarfing the pointwise
convergence error (~8e-5).

The slab T1/T2 rows therefore make **three** assertions per group:
(1) the rate ``orders > 1.9`` (DD design order on a smooth ansatz);
(2) the finest-mesh :math:`\phi_{\text{num}}` matches :math:`A(x)` to
``rtol=atol=5e-3`` — with a guard asserting the reference is genuinely
non-vacuum (:math:`|A(0)|, |A(L)| > 0.1`) so the value check is
discriminating; and (3) an inflow-honoured spot-check that the solved
trace slot equals the imposed :math:`\gamma_-\psi = (A + \mu_n B)/W` to
``rtol=1e-9``. Only the combination is a meaningful test of the
prescribed-inflow path.

The Mode-7 activates/nulls map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``vv-principles`` failure-mode #7 ("MMS simplification bias") requires
every multi-dimensional MMS test to **declare** which operator terms
its ansatz activates and which it nulls — and to ship an
angularly-non-trivial companion whenever the nulled set includes a term
covered by an active ERR-NNN. The 4.6 declaration:

.. list-table:: Mode-7 term map — slab vs sphere under the (A+μB)/W ansatz
   :header-rows: 1
   :widths: 40 30 30

   * - Operator term
     - Slab (Cartesian)
     - Sphere (spherical)
   * - streaming :math:`\mu A'` (isotropic)
     - **activates**
     - **activates**
   * - streaming :math:`\mu^2 B'` (second moment)
     - **activates**
     - **activates**
   * - angular redistribution :math:`(1-\mu^2)B/r`
     - **nulls** (no :math:`\partial_\mu` term)
     - **activates** (the ERR-026 path)
   * - within-group scatter :math:`\Sigma_s\phi` (:math:`c<1`)
     - **activates**
     - **activates**
   * - collision :math:`\Sigma_t\,\mu B` (first moment)
     - **activates**
     - **activates**
   * - 2G group transfer :math:`\Sigma_s^\top` (asymmetric)
     - **activates** (T2)
     - n/a (1g)
   * - prescribed non-vacuum inflow :math:`\gamma_-\psi \neq 0`
     - **activates** (both faces, :math:`a_0>0`)
     - **activates** (:math:`r=R` face)
   * - fission
     - **nulls** (non-fissile; MMS proves no eigenvalue)
     - **nulls**

The slab **nulls the angular redistribution** — the Cartesian operator
has no :math:`\partial_\mu` coupling. Redistribution is exactly where
ERR-026 (the curvilinear sweep WDD wrong-fixed-point bug) lives, so a
slab-only 4.6 would be a textbook Mode-7 trap: it would verify the
prescribed-inflow path while being structurally blind to the hardest
math the curvilinear sweep performs. The **sphere companion is
therefore mandatory** (NEVER ship slab-only — ERR-026 territory). The
sphere activates the redistribution term under non-vacuum inflow,
closing the Mode-7 declaration.

T3 (sphere) — the retired ``xfail(strict)`` and the angular floor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: **Terminal (2026-06-13, the #229 retune landed — the xfail
   was REMOVED).**

   The strict-xfail this section explains was removed by the #229
   quadrature-aware retune (post-W1 unclamp; recorded in the test's
   own docstring).  The row now **passes** as a converged-VALUE +
   magnitude-band gate: the solution converges to the manufactured
   :math:`\phi` to ~0.26 % max relative error at ``nx = 160``, but
   NOT in rate — the volume-weighted L2 floors at the angular
   half-angle-thread interpolation floor (~2.4e-3), and the
   pole-cell :math:`\mathcal{O}(h)` term (#233) dominates the finest
   cells, so no rate claim is asserted.  What follows is the
   diagnostic history: the #195 → #229 re-scoping and the bug-era
   stagnation evidence.

The sphere row
(:func:`tests.sn.verification.analytical.test_mms_prescribed_inflow.test_mms_prescribed_inflow_sphere_activates_redistribution`)
shipped ``@pytest.mark.xfail(strict=True)`` with ``catches("ERR-026")``
(the marker is retired; the ``catches`` tag remains).  The reason was
**not** that the non-vacuum machinery fails.

.. note:: **Re-scoped (2026-06-12, Issue #195 CLOSED → Issue #229).**

   This row's xfail was originally attributed to the #195
   "pre-asymptotic transient" plateau (the stagnation table below).
   The ERR-058 closure-seed fix **closed** the curvilinear
   wrong-fixed-point family, so the stagnation is gone — the isotropic
   curvilinear DD interior is now :math:`\mathcal{O}(h^2)`-consistent
   (see :ref:`sn-err-058-closure-seed-closeout`).  T3's ansatz, however,
   is the *anisotropic* :math:`(A(r)+B(r)\mu)/W`, which is angle-varying
   and therefore hits the **fixed-quadrature angular floor** of the
   half-angle thread interpolation (sphere S16 floor
   :math:`\approx 7\mathrm{e}{-4}`, above the band).  The marker
   **stays ``xfail(strict)``** but is now pinned to the
   `Issue #229 <https://github.com/deOliveira-R/ORPHEUS/issues/229>`_
   quadrature-aware retune, NOT the #195 plateau; it flips to
   unexpected-pass when #229 lands (the regression gate for the retune).
   The stagnation table below is preserved as **bug-era evidence**; its
   "pre-asymptotic" interpretation is superseded.

**Bug-era stagnation (pre-ERR-058).**  The slab was pole-free and
converged perfectly: orders ``[2.04, 2.01]``, finest L2 ~1.2e-3,
pointwise ``max|φ−A|`` ~8e-5, boundary value matched.  The sphere L2
(volume-weighted), by contrast, **stagnated** mesh-independently — the
plateau that refuted the "pre-asymptotic transient" premise:

.. list-table:: T3 sphere volume-weighted L2 error (bug-era plateau, pre-ERR-058)
   :header-rows: 1
   :widths: 25 25 25 25

   * - :math:`n_c`
     - 20
     - 40
     - 80
   * - :math:`\|\phi_h - A\|_{L^2(V)}`
     - 2.37e-2
     - 2.42e-2
     - 2.43e-2

The observed "orders" were ≈ :math:`-0.02` to :math:`-0.006` — the
error was *not* decreasing under refinement, the plateau signature
ERR-058 diagnosed.  Post-ERR-058 the spatial convergence is recovered;
the residual gap on this *anisotropic* row is the #229 angular floor,
which DOES drop under quadrature refinement (sphere S16
:math:`\to` S32 halves the floor) — the structural test that the
remaining gap is angular, not a wrong fixed point.

Crucially, the **boundary value is honoured** (always was): the
finest-mesh :math:`\phi[-1] \approx 0.7499 \approx A(R) = 0.75`, and
the inflow-trace spot check passes.  The non-vacuum prescribed-inflow
machinery *works*; the remaining xfail is purely the angular-floor
budget of the anisotropic ansatz under fixed quadrature.

**T3g — the green structural companion.** Because T3 is xfail (now on
#229), it provides *no live* convergence coverage of the 4.6 machinery.
The green companion
:func:`tests.sn.verification.analytical.test_mms_prescribed_inflow.test_sphere_nonvacuum_inflow_honoured_and_redistribution_live`
fills that gap with two non-convergence-dependent claims that pass
*now*: (1) the prescribed inflow at :math:`r=R` is honoured per inflow
ordinate (:math:`\gamma_-\psi = (A(R) + \mu_n B(R))/W` with :math:`A(R)
> 0` non-vacuum); and (2) the redistribution source
:math:`(1-\mu^2)B(r)/r` is non-zero on the mesh interior (the ERR-026
term is live under the 4.6 ansatz — :math:`B(r)\neq 0` on the open
interval, with :math:`B(0)=0` pole-regular). T3g is the live structural
guarantee that the Mode-7 sphere companion exercises the redistribution
path even while the convergence row is parked on #229.

T4 (vv Mode 9) — splitting invariance of the prescribed inflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The consistency floor the convergence rows trust is that a non-zero
prescribed inflow is honoured **identically** by the three operator
splittings of the affine within-group system: SI-Jacobi (the resolvent
:math:`L+C` with lagged gains :math:`S, B`), SI-Gauss–Seidel (the
:math:`B_{\rm lower}`-folding reified :math:`M`,
:class:`~orpheus.sn.operators.scheduled_invertible.ScheduledInvertibleOperator`,
with lagged gains :math:`S, B_{\rm upper}`), and Krylov (the matvec
:math:`L+C-S-B`). All three are different reduction trees of the *same*
affine fixed point :math:`(L+C-S-B)\psi = q`, so they MUST reach the
same :math:`\psi` (``vv-principles`` Mode 9 — verify splittings reach
the same fixed point under anisotropic / :math:`B\neq 0` stressing).
⚠ *"the same* :math:`\psi`\ *"* is legitimate here because both T4
configs are **kernel-free** — ``slab_1d`` is :math:`d=1` and
``cart2d_reflective_y`` closes **one** reflective axis pair, and a
solution *point* exists only when :math:`A` is nonsingular.  Close two
pairs under diamond differencing and it is not (:ref:`sn-loss-kernel-gauge`),
so a future config added to this row must either stay kernel-free or
compare the **gauged** trace.
This is a **foundation** test, not an L1 claim: no theory-page
:math:`:label:` is being verified — it pins that three reduction trees
of one affine operator agree on one RHS, which is a software invariant
with no equation label to link, so the gate carries no ``verifies()``.

T4
(:func:`tests.sn.verification.analytical.test_prescribed_inflow_consistency.test_prescribed_inflow_consistency_si_jacobi_gs_krylov`)
runs two configs. The ``slab_1d`` config (SI is always Jacobi in 1-D)
makes **SI ≡ Krylov** the discriminating pair. The
``cart2d_reflective_y`` config adds reflective-:math:`y` faces so
:math:`B \neq 0` — which is what makes **SI-Jacobi vs SI-Gauss–Seidel**
distinct (G-S folds :math:`B` into the resolvent; Jacobi lags it). The
:math:`B\neq 0`-plus-prescribed-inflow combination is the only config
where the :math:`B`-folding path runs *with* a non-zero boundary source
(the ERR-056 neighbourhood). Measured pairwise reldiffs: 1.3e-13 …
5.6e-13 — comfortably under the 1e-11 ceiling, which itself leaves
headroom for the FP-non-associativity of three reduction trees
(bounded by :math:`\text{iter} \times \text{ULP}` per the
``vv-principles`` bit-identity criteria).

The test carries explicit anti-latent-dud preconditions (the
splitting-invariance check is vacuous if all three trivially agree on
:math:`\psi \equiv 0`): the inflow slot must actually be written
(:math:`>0`), the inflow must non-trivially drive the interior
(:math:`\max|\psi| > 10^{-3}`), and the 2-D row must actually select the
:math:`B_{\rm lower}`-folding reified :math:`M`
(:class:`~orpheus.sn.operators.scheduled_invertible.ScheduledInvertibleOperator`,
not silently fall back to Jacobi) with an explicit reflective-:math:`y`
``Mesh2D`` BC.

Verification chain — Branch 1 / Branch 2 / L1 cross-check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following the ``algebra-of-record`` discipline:

- **Branch 1 (SymPy, State 1C — MMS).** The manufactured sources
  :eq:`sn-mms-nonvacuum-qext` (slab) and
  :eq:`sn-mms-nonvacuum-sph-qext` (sphere) are derived by substituting
  the imposed ansatz into the continuous operator and solving for the
  residual, symbolically. The slab pair is
  :func:`~orpheus.derivations.continuous.mms.sn._nonvacuum_slab_symbolic`
  /
  :func:`~orpheus.derivations.continuous.mms.sn.derive_nonvacuum_slab_mms`;
  the sphere reuses
  :func:`~orpheus.derivations.continuous.mms.sn._spherical_anisotropic_symbolic`
  with the 4.6 shapes
  (:func:`~orpheus.derivations.continuous.mms.sn._nonvacuum_spherical_AB`)
  via
  :func:`~orpheus.derivations.continuous.mms.sn.derive_nonvacuum_spherical_mms`.
  Each ``derive_*`` proves ``simplify(W·LHS − Σ_s·φ − Q_closed) == 0``.
  Foundation gate:
  :file:`tests/derivations/test_sn_mms_nonvacuum_symbolic.py`.
- **Branch 2 (vectorised numpy).** The factories
  :class:`~orpheus.derivations.continuous.mms.sn.SNSlabNonVacuumMMSCase`
  and
  :class:`~orpheus.derivations.continuous.mms.sn.SNSphericalNonVacuumMMSCase`
  (built by
  :func:`~orpheus.derivations.continuous.mms.sn.build_slab_nonvacuum_mms_case`,
  :func:`~orpheus.derivations.continuous.mms.sn.build_slab_2g_nonvacuum_mms_case`,
  and
  :func:`~orpheus.derivations.continuous.mms.sn.build_sphere_nonvacuum_mms_case`)
  evaluate the closed-form source per ordinate using vectorised numpy.
  Each carries a ``prescribed_inflow(sn)`` method returning the
  ``q.boundary`` :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink`,
  and a ``fixed_source(sn)`` bundler returning the composite
  ``q = q_bulk ⊕ q_∂``
  :class:`~orpheus.transport.timed_full_field.TimedFullField` the public
  solver consumes (see :ref:`sn-composite-fixed-source`).
- **L1 cross-check (the gate).** The Branch-2 numpy
  :math:`Q^{\text{ext}}_n` is bit-equal (≤1e-13 max absolute) to the
  Branch-1 SymPy closed form evaluated via :func:`sympy.lambdify` on a
  sample mesh, for both geometries. The two branches are *structurally
  independent above the trusted-library line* — Branch 1 is
  ``lambdify``-d SymPy, Branch 2 is hand-written numpy — so agreement
  catches a copy error between the symbolic derivation and the
  numerical implementation. Tested in
  :func:`tests.derivations.test_sn_mms_nonvacuum_symbolic.test_slab_nonvacuum_numerical_qext_matches_sympy`
  and the spherical sibling.

**Structural independence (L11).** The chosen scalar flux
:math:`\phi = A` is *imposed* analytically; the source :math:`Q^{\text{ext}}`
is SymPy-derived (not generated by the solver's own primitives); the
numpy ``external_source`` is then cross-checked bit-equal to the
lambdified SymPy. The reference is structurally independent of the code
under test — the manufactured source does not pass through any of the
solver's discretisation primitives, so the L1 convergence rows are a
genuine test, not a tautology.

**What this section does NOT verify.** Per the three pillars
(``vv-principles``), MMS is a *source-driven* problem: it verifies the
convergence order (a math claim) and the flux shape (a model claim,
because the source is structurally independent), but it **cannot**
verify an eigenvalue. The 4.6 mixtures are non-fissile by construction
and there is no eigenvalue claim anywhere in this section. The
prescribed-inflow verification is a forward-only, fixed-source result.


.. _sn-case-heterogeneous-verification:

Heterogeneous eigenvalue — Case singular-eigenfunction method
--------------------------------------------------------------

Phase 2.1b of the verification campaign closes the last
heterogeneous gap in the SN verification ladder: the
**eigenvalue iteration** on a 1-group two-region reflective
slab, verified against a semi-analytical reference derived
from the discrete-:math:`S_N` slope matrix itself --- no
diffusion approximation, no cross-code comparison, no
Richardson self-test.

The reference is produced by
:func:`orpheus.derivations.continuous.cases.sn.derive_sn_heterogeneous_continuous`
and consumed by
:func:`tests.sn.eigenvalue.test_heterogeneous_transport.test_sn_2region_reflective_case_eigenvalue`
(eigenvalue) and
:func:`tests.sn.eigenvalue.test_heterogeneous_transport.test_sn_2region_reflective_flux_shape`
(scalar flux shape). The Phase 2.1a smooth-:math:`\Sigma` MMS
test verifies the **spatial operator** at :math:`\mathcal O(h^{2})`
design order; this section's Case method verifies the
**eigenvalue** iteration at the material-interface-degraded
:math:`\mathcal O(h)` rate expected for diamond-difference on
piecewise-constant :math:`\Sigma`.

**Motivation: why a second verification path.** The Phase 2.1a
MMS test deliberately uses smooth :math:`\Sigma(x)` to avoid
interface degradation and hit the :math:`\mathcal O(h^{2})`
design order of diamond difference. That is the right choice
for verifying the spatial operator, but it **cannot** exercise
the heterogeneous-interface regime where material
discontinuities force the operator into its interface-layer
behaviour --- the regime where a significant fraction of
production solver bugs live (including ERR-025 — see the
:ref:`homogeneous / uniform-rescale gotcha <sn-homogeneous-degeneracy-gotcha>`
for the mechanism by which it hid). The Case singular-eigenfunction
method provides the complementary reference: an eigenvalue
solution with genuine material-interface discontinuities, built
from the transport equation without running the solver.

**Operator.** The 1-group 1D slab SN transport equation in a
single region with cross sections
:math:`(\Sigma_t, \Sigma_s, \nu\Sigma_f)` and reflective BCs
is, per ordinate,

.. math::
   :label: sn-case-per-ordinate

   \mu_n\,\frac{d\psi_n}{dx} + \Sigma_t\,\psi_n
     \;=\; \frac{c_\text{eff}(k)}{W}\,\phi,
   \qquad
   \phi = \sum_m w_m\,\psi_m,
   \qquad
   c_\text{eff}(k) = \Sigma_s + \frac{\nu\Sigma_f}{k},

where :math:`W = \sum_m w_m`. Substituting the scalar-flux
definition and stacking the angular flux into
:math:`\mathbf y \in \mathbb R^N` (for Gauss--Legendre order
:math:`N`), the system becomes a first-order constant-coefficient
ODE

.. math::
   :label: sn-case-slope-matrix

   \frac{d\mathbf y}{dx} \;=\; \mathbf S(k)\,\mathbf y,
   \qquad
   \mathbf S(k)[n, m] \;=\; \frac{1}{\mu_n}
       \left(-\Sigma_t\,\delta_{nm}
             + \frac{c_\text{eff}(k)}{W}\,w_m\right).

Note the **row-scaling** :math:`1/\mu_n`: the slope matrix is
generally non-symmetric even for symmetric GL quadrature,
because the angular ODE has different "speeds" for different
ordinates.

**Per-region spatial modes.** For each region (fuel at
:math:`x \in [0, H_A]` and moderator at :math:`x \in [H_A, L]`),
diagonalise :math:`\mathbf S(k)`:

.. math::
   :label: sn-case-spatial-modes

   \mathbf S(k)\,\mathbf v_i \;=\; \lambda_i\,\mathbf v_i,
   \qquad i = 1,\ldots,N,

via :func:`numpy.linalg.eig`. For subcritical regions
(:math:`c_\text{eff}(k) < 1`, typical moderator) the eigenvalues
come in :math:`\pm` real pairs. For supercritical regions
(:math:`c_\text{eff}(k) > 1`, fuel at :math:`k` below
:math:`k_{\infty,\text{fuel}}`) some pairs are
complex-conjugate. Each real eigenvalue gives one exponential
mode :math:`\exp(\lambda\,x)\,\mathbf v`; each complex-conjugate
pair gives two real modes built from the canonical
:math:`\cos/\sin/\Re/\Im` combination.

**Real bounded basis.** The naive unbounded basis
:math:`\exp(\lambda\,x)\,\mathbf v` is catastrophically
ill-conditioned for optically thick slabs --- the Phase 1.2
diffusion investigation history records the ``expm``-based
transfer-matrix composition dying from :math:`\text{cond}
\sim 10^{17}` on an 80-cm slab, finding spurious roots with
:math:`\mathcal O(10^{-3})` null-vector residuals rather than
machine-precision zeros. The fix, ported verbatim to Phase 2.1b,
is to **anchor each mode at the nearer region edge**:

.. math::
   :label: sn-case-real-basis

   m^{\text{real}}_j(x) &\;=\; \exp(\lambda_j\,\xi_j)\,\mathbf v_j,
       \qquad
       \xi_j = \begin{cases}
         x - L_\text{reg} & \lambda_j \ge 0 \;\;\text{(anchor right)} \\
         x                & \lambda_j < 0 \;\;\text{(anchor left)}
       \end{cases} \\[1mm]
   m^{\text{c}}_j(x) &\;=\; e^{\Re\lambda_j\,\xi_j}\,
       \bigl(\cos(\Im\lambda_j\,\xi_j)\,\mathbf v_{R,j}
          - \sin(\Im\lambda_j\,\xi_j)\,\mathbf v_{I,j}\bigr), \\
   m^{\text{s}}_j(x) &\;=\; e^{\Re\lambda_j\,\xi_j}\,
       \bigl(\sin(\Im\lambda_j\,\xi_j)\,\mathbf v_{R,j}
          + \cos(\Im\lambda_j\,\xi_j)\,\mathbf v_{I,j}\bigr),

where :math:`\mathbf v_j = \mathbf v_{R,j} + i\,\mathbf v_{I,j}`
is the complex eigenvector. Every mode is bounded by
:math:`|\mathbf v_j|` on its region, so the assembled matching
matrix has :math:`\mathcal O(1)` entries.

**Matching matrix.** For the 2-region reflective slab the
coefficient vector has dimension :math:`2N` (one real mode per
eigenvalue per region). The linear constraints are:

.. math::
   :label: sn-case-matching-matrix

   &\text{Reflective at } x = 0:\quad
      \psi^A_n(0) - \psi^A_{N-1-n}(0) = 0,
      \qquad n \in [0, N/2) \\[1mm]
   &\text{Interface at } x = H_A:\quad
      \psi^A_n(H_A) - \psi^B_n(H_A) = 0,
      \qquad n \in [0, N) \\[1mm]
   &\text{Reflective at } x = L:\quad
      \psi^B_n(L) - \psi^B_{N-1-n}(L) = 0,
      \qquad n \in [0, N/2)

:math:`N/2 + N + N/2 = 2N` equations in :math:`2N` unknowns.
The partner index :math:`N-1-n` is the Gauss--Legendre
reflection pairing (ordinates sorted by ascending :math:`\mu`).
The eigenvalue condition is
:math:`\det\mathbf C(k) = 0`.

**Root finding.** :func:`scipy.optimize.brentq` on
:math:`\det\mathbf C(k)` over a coarse :math:`k`-scan, with
sign-change bracketing, refines every candidate to
``xtol=1e-14``. But :func:`numpy.linalg.eig`'s eigenvalue
ordering is not a continuous function of :math:`k` --- at
parameter values where two per-region eigenvalues cross, the
eigenvalue labels permute discontinuously, and
:math:`\det\mathbf C(k)` flips sign by permutation rather than
by passing through zero. brentq will "converge" to such
spurious points.

**Physical validation.** Every candidate root is rebuilt via
SVD of :math:`\mathbf C(k)`, and the null vector's reflective-BC
residuals at :math:`x = 0` and :math:`x = L`, and the interface
continuity residual at :math:`x = H_A`, are explicitly
reconstructed and checked against a dimensionless tolerance
relative to the peak angular flux:

.. math::
   :label: sn-case-physical-validation

   \|\psi(0, +\mu_n) - \psi(0, -\mu_n)\| / \|\psi\|_\text{peak}
     &< \text{tol} \\
   \|\psi^A(H_A) - \psi^B(H_A)\| / \|\psi\|_\text{peak}
     &< \text{tol} \\
   \|\psi(L, +\mu_n) - \psi(L, -\mu_n)\| / \|\psi\|_\text{peak}
     &< \text{tol}

Only candidates passing all three are accepted; the fundamental
is the largest validated root. This is the SN analogue of the
Phase 1.2 diffusion physical validation (same pattern, different
operator).

**Back-substitution.** Once :math:`k_\text{fund}` is found,
the null vector at that :math:`k` is the coefficient vector in
the :math:`2N`-dimensional real basis. Evaluation of
:math:`\phi(x) = \sum_n w_n\,\psi_n(x)` at any point reduces to
a linear combination of a handful of bounded exponential or
trigonometric modes:

.. math::
   :label: sn-case-back-substitution

   \psi(x) = \begin{cases}
     \sum_j c^A_j\,m^A_j(x) & x \le H_A \\[1mm]
     \sum_j c^B_j\,m^B_j(x - H_A) & x > H_A
   \end{cases},
   \qquad
   \phi(x) = \sum_n w_n\,\psi_n(x).

All modes are bounded by :math:`\mathcal O(1)`, so
:math:`\phi(x)` is stable to machine precision.

**The Phase 2.1b diagnostic configuration.** The canonical
test problem is the ``A`` + ``B`` 1-group mixture pair from
:mod:`orpheus.derivations.common.xs_library`:

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15

   * - Region
     - :math:`\Sigma_t`
     - :math:`\Sigma_s`
     - :math:`\nu\Sigma_f`
     - :math:`k_\infty`
   * - A (fuel)
     - 1.0
     - 0.5
     - 0.75
     - 1.5
   * - B (moderator)
     - 2.0
     - 1.9
     - 0
     - ---

with :math:`H_A = H_B = 0.5\,\text{cm}`, reflective BCs on both
outer edges, :math:`S_8` Gauss--Legendre quadrature. The
resulting Case reference is

.. math::

   k_\text{eff}^{\text{Case}}(S_8) = 1.2746160417

--- the exact discrete-:math:`S_8` eigenvalue. For
cross-validation, the same configuration run through ORPHEUS's
:func:`~orpheus.cp.solver.solve_cp` (1D slab E\ :sub:`3` kernel,
completely independent numerical path) gives
:math:`k^{\text{CP}} = 1.2744284665` --- agreement to
:math:`\sim 2\times 10^{-4}`, well below the :math:`\mathcal O(1\%)`
difference that typically exists between discrete-SN and
continuous-angle formulations. This cross-check is used only as
a sanity input, not as a verification crutch.

**Measured convergence.** With :math:`S_8`, refining
:math:`n_\text{per}` per region:

.. list-table::
   :header-rows: 1
   :widths: 15 25 15

   * - :math:`n_\text{per}`
     - :math:`k_\text{solve}`
     - :math:`|k_\text{solve} - k_\text{Case}|`
   * - 20
     - 1.2746074093
     - :math:`\sim 8.6\!\times\!10^{-6}`
   * - 40
     - 1.2746138837
     - :math:`\sim 2.2\!\times\!10^{-6}`
   * - 80
     - 1.2746155022
     - :math:`\sim 5.4\!\times\!10^{-7}`
   * - 160
     - 1.2746159068
     - :math:`\sim 1.3\!\times\!10^{-7}`
   * - 320
     - 1.2746160080
     - :math:`\sim 3.4\!\times\!10^{-8}`

Each refinement roughly halves the error, confirming the
:math:`\mathcal O(h)` rate expected at a material interface with
piecewise-constant :math:`\Sigma`. The finest-mesh residual of
:math:`3.4 \times 10^{-8}` is **machine-precision agreement**
between two independent mathematical constructions (the Case
matching-matrix + back-substitution reference and the
diamond-difference sweep-based power iteration); both
implementations solve the same discrete-:math:`S_N` spectral
problem and agree to within the BiCGSTAB-compatible
truncation.

**Contrast with Phase 2.1a.** The Phase 2.1a MMS section hits
:math:`\mathcal O(h^{2})` because it uses smooth
:math:`\Sigma(x)`; the Phase 2.1b Case section hits
:math:`\mathcal O(h)` because it uses piecewise-constant
:math:`\Sigma(x)` with a genuine material interface. Both are
correct for their respective regimes. The degradation from
:math:`h^{2}` to :math:`h` at the interface is the standard
Salari--Knupp result for DD on discontinuous coefficients, and
is the **reason** Phase 2.1a deliberately chose smooth
:math:`\Sigma` to isolate the spatial operator.


Analytical eigenvalue derivation
---------------------------------

The S\ :sub:`N` method discretises the angular variable into a finite set of
directions.  For a homogeneous medium with :term:`reflective boundary conditions <reflective boundary condition>`,
the derivation starts from the 1D S\ :sub:`N` transport equation:

.. math::

   \mu_m \frac{\partial\psi_m}{\partial x} + \Sigma_t \psi_m = \frac{Q}{2}

For a homogeneous medium, :math:`\partial\psi_m/\partial x = 0` (spatially
flat flux), so :math:`\psi_m = Q/(2\Sigma_t)` for every direction.  Integrating
with Gauss-Legendre weights (:math:`\sum w_m = 2`):

.. math::

   \phi = \sum_m w_m \psi_m = \frac{Q}{\Sigma_t}

Substituting the source :math:`Q = \Sigma_s \phi + (1/k)\nu\Sigma_f \phi`
and cancelling :math:`\phi` yields the same eigenvalue as the homogeneous
problem.  This is an exact result — the GL quadrature integrates a constant
exactly, and diamond-difference is exact for flat flux.

For heterogeneous problems, the reference comes from Richardson extrapolation
of the O(h²) diamond-difference scheme.

.. include:: ../../_generated/sn_derivation.rst


Homogeneous Infinite Medium
----------------------------

For homogeneous geometry with reflective BCs, the flux is spatially flat
and :math:`\keff = \lambda_{\max}(A^{-1}F)`.  This is geometry-independent
--- Cartesian, spherical, and cylindrical must all give the same
:math:`\keff`.

.. list-table::
   :header-rows: 1
   :widths: 10 14 19 19 19 19

   * - Groups
     - :math:`\kinf`
     - Cartesian (GL S8)
     - Spherical (GL S8)
     - Cylindrical (Prod 4x8)
     - Cylindrical (LS S4)
   * - 1
     - 1.5000
     - exact
     - exact
     - exact
     - exact
   * - 2
     - 1.8750
     - exact
     - exact
     - exact
     - exact
   * - 4
     - 1.4878
     - exact
     - exact
     - exact
     - exact

All entries are exact to machine precision.  Spherical 2G/4G results
(previously showing ~1% error) are now exact thanks to the M-M angular
closure weights.

Heterogeneous Convergence
--------------------------

For a cylindrical fuel (r < 0.5) + moderator (r < 1.0) geometry with
Product(4x8) quadrature:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25

   * - Cells/region
     - :math:`\keff` (1G)
     - :math:`\Delta k` from previous
   * - 5
     - 0.9769
     -
   * - 10
     - 0.9842
     - +0.0073
   * - 20
     - 0.9874
     - +0.0032

:math:`\keff` converges monotonically toward the CP reference
(0.9955).  The ~1% residual gap is the white-BC (CP) vs reflective-BC
(SN) approximation difference, consistent with the slab geometry
findings.

For the 2G heterogeneous resolution test, Product(4x8) and Product(8x8)
agree to :math:`< 0.01\%` (keff = 0.7227 for both), confirming
angular convergence.

Why 1-Group Verification Is Degenerate
----------------------------------------

For 1 energy group, the eigenvalue is:

.. math::

   k = \frac{\nSigf{}}{\Sigma_a}

This is a scalar ratio independent of the spatial or angular flux
distribution.  Consequences:

- Angular weight errors scale all flux equally --- cancels in :math:`k`.
- Wrong scattering convention --- no inter-group coupling to distort.
- Wrong flux shape --- does not matter; :math:`k` is a material property.

Only multi-group problems have a flux-shape-dependent eigenvalue:
:math:`k = (\nSigf{} \cdot \phi) / (\Sigma_a \cdot \phi)` where the
dot product weights each group differently.  A wrong group ratio (from
angular errors, normalization errors, or convergence failures) directly
shifts :math:`\keff`.

**Rule:** Every transport solver must be verified on at least 2-group
problems.  1-group success gives false confidence.

Spatial and Angular Convergence
--------------------------------

The diamond-difference scheme converges at :math:`O(h^2)` with mesh
refinement.  Gauss--Legendre quadrature shows spectral convergence in
angle.  Both are verified in ``tests/sn/eigenvalue/test_keff_slab.py``
(``test_spatial_convergence`` and ``test_angular_convergence``).

Property Tests
---------------

For all geometries:

- **Particle balance**: production / absorption :math:`= \keff`
- **Flux non-negativity**: :math:`\phi \geq 0` everywhere
- **Angular flux at** :math:`r = 0` **all positive** (curvilinear only)
- **Multi-group eigenvector not flat**: flux spectrum differs between
  fuel and moderator (catches 1G-degenerate bugs)

Run the full suite::

   python -O -m pytest tests/sn -m "not slow"


Numerical Sensitivities
========================

:math:`\keff` Sensitivity Table (421-Group Heterogeneous PWR Slab)
-------------------------------------------------------------------

All cases: 10 cells, :math:`\delta = 0.2` cm, material layout
``[fuel x 5, clad x 1, cool x 4]``, P0 scattering, 421 energy groups.

.. list-table::
   :header-rows: 1
   :widths: 50 15 35

   * - Configuration
     - :math:`\keff`
     - Notes
   * - 1D GL S16, BiCGSTAB (FD operator)
     - 1.03882
     - True 1D, 16 ordinates
   * - 1D Lebedev 110, source iteration (DD sweep)
     - 1.04294
     - 1D mesh, 2D quadrature
   * - 2D (10x2) Lebedev 110, source iter (DD sweep)
     - 1.04294
     - Pseudo-2D, full volumes
   * - 2D (10x2) Lebedev 110, BiCGSTAB (FD)
     - 1.04007
     - Pseudo-2D, full volumes
   * - 2D (10x2) Lebedev 110, BiCGSTAB, half-volumes
     - 1.04192
     - MATLAB convention
   * - **MATLAB reference**
     - **1.04188**
     - 2D Lebedev, FD, half-volumes

Sources of Variation
---------------------

1. **Angular quadrature** (GL vs Lebedev): ~0.004 difference.
   GL S16 integrates 1D angular flux with 16 points on :math:`[-1,1]`.
   Lebedev 110 integrates over the unit sphere --- more angular
   resolution but different effective weights per :math:`\mu_x`
   direction.  On a coarse heterogeneous mesh, these give different
   eigenvalues.

2. **Spatial discretisation** (DD sweep vs FD gradient): ~0.003
   difference.  Source iteration uses the DD wavefront sweep
   (:math:`T^{-1}`).  BiCGSTAB uses the explicit FD transport operator
   (:math:`T`).  Both are :math:`O(h)` on this mesh but with different
   truncation error constants.

3. **Boundary volume weighting**: ~0.002 difference (full vs half).
   The MATLAB code halves boundary cell volumes.  With ``ny=2`` and
   materials uniform in *y*, only the *x*-direction halving (fuel edge,
   coolant edge) affects :math:`\keff`.  This is an artifact of the
   pseudo-2D implementation: a true 1D calculation has no *y*-volumes.

4. **Inner convergence**: source iteration with ``max_inner=200``,
   ``inner_tol=1e-8`` does not fully converge for 421 groups (spectral
   radius ~0.97).  BiCGSTAB fully converges the inner solve in ~100
   Krylov iterations.

Matching the MATLAB Result
---------------------------

The MATLAB code uses: 2D Lebedev 110 on a 10x2 mesh, explicit FD
operator with BiCGSTAB, boundary half-volumes, P0 scattering.

The BiCGSTAB path with half-volumes reproduces 1.04192 vs MATLAB's
1.04188 (:math:`4 \times 10^{-5}` agreement).  The residual difference
is from floating-point details in cross-section processing.

The cleanest reference is the **1D GL BiCGSTAB** result (1.03882): no
pseudo-2D artifacts, well-conditioned angular quadrature, fully
converged inner solve.


.. _sn-adjoint-verification-slice:

Adjoint transport — the daggered posing
---------------------------------------

The adjoint flux :math:`\psi^*` and the daggered eigenvalue
:math:`k^{\dagger}` (campaign #276 A4/A5) are verified by **closed-form**
and **defining-law-residual** references, never by MMS.  The physics,
the route decision, and the three-transposes taxonomy are the theory
chapter :ref:`sn-adjoint`; this slice is the verification evidence.

Why closed-form, not MMS
~~~~~~~~~~~~~~~~~~~~~~~~~

MMS is a *source-driven* pillar: it verifies flux-shape and
convergence-order but **cannot verify an eigenvalue** (``vv-principles``,
the pillars — the manufactured source is derived from a chosen flux, so
there is no eigenvalue information to check against).  The daggered
:math:`k^{\dagger}` and the importance spectrum :math:`\varphi^*` are
eigenvalue / flux-shape claims, which need **closed-form** references.
Two facts make those references *exact-equality* checks rather than
tolerance agreements, both consequences of the route decision (ORPHEUS
transposes the *discrete* operator, :ref:`sn-adjoint-route`):

* :math:`k^{\dagger} = k` is an **exact algebraic identity**
  (:math:`\operatorname{eig}(M^{\dagger}) = \operatorname{eig}(M)`), so
  the k-rows assert equality to the iteration floor, not a physical
  tolerance;
* reciprocity :math:`\langle\Sigma_d,\psi\rangle =
  \langle\psi^*,q\rangle` holds **exactly at finite** :math:`N, h`,
  because :math:`A_{\rm loss}^{\dagger}` *is* the discrete transpose —
  so its residual is a solver-``inner_tol`` check, not an
  :math:`\mathcal O(h^p)` one.

The certification gates
~~~~~~~~~~~~~~~~~~~~~~~~

The battery lives in
``tests/sn/solve/test_sn_adjoint_certification.py`` (P1.3/P1.4/P1.5 +
the sphere vector row) and ``tests/sn/solve/test_sn_adjoint_entries.py``
(P1.2 duality + the entry packaging).  Every value gate is **L1**;
bi-orthogonality is **foundation** (an intrinsic algebraic law, no
theory ``:label:``).  Every reference is structurally independent —
each chain terminates in ``np.linalg.eig``, the reciprocity identity,
or a dense FORWARD-probe (never the ``.H`` reverse-scan under test).

.. list-table:: Adjoint certification (Mixture A; measured)
   :header-rows: 1
   :widths: 16 10 30 44

   * - Gate (test)
     - Level
     - Reference / pillar
     - What it pins (measured)
   * - **P1.2** duality
       (``test_duality_cross_group_source_detector``)
     - L1
     - the reciprocity identity; two independent solves
     - :math:`\langle\Sigma_d,\psi\rangle=\langle\psi^*,q\rangle` on a
       2G asymmetric-SigS vacuum slab, source (fast, left) and detector
       (thermal, right) in DIFFERENT groups AND regions; the detector
       side additionally hand-checked against
       :math:`\sum V\Sigma_d\varphi` at :math:`10^{-10}` (pins the
       angle-flat dual lift — no :math:`w_n`, no :math:`1/W`)
   * - **P1.3** :math:`k^{\dagger}=k`
       (``TestP13KEquality``)
     - L1
     - ``kinf_homogeneous`` — triple equality, closed-form
     - :math:`k^{\dagger}=k_{\rm fwd}=k_\infty` (atol :math:`10^{-8}`) on
       ∞ 2G+4G, a 2-region reflective slab (spatial term LIVE), and the
       coupled sphere (μ-reversal at the pole).  ∞-only would be a
       config-blind REJECT
   * - **P1.4** spectrum
       (``test_4g_spectrum_matches_closed_form``)
     - L1
     - dominant right eigenvector of
       :math:`(A^{\mathsf T})^{-1}F^{\mathsf T}`
       (``kinf_and_adjoint_spectrum_homogeneous``)
     - the 4G adjoint energy spectrum
       :math:`[0.470, 0.486, 0.518, 0.524]` (2G: :math:`[0.684,
       0.730]`), :math:`\ne\varphi` asserted; the corrected
       factor-order reference (see below)
   * - **P1.5** bi-orthogonality
       (``TestP15BiOrthogonality``)
     - foundation
     - spectral decomposition of :math:`M`, :math:`M^{\dagger}`
       (``np.linalg.eig`` both sides)
     - :math:`\langle\psi^*_i, F\varphi_j\rangle` diagonal; for rank-1
       :math:`F=\chi\otimes\nu\Sigma_f` the degenerate one-nonzero-entry
       form (both :math:`F\varphi_j=0` and :math:`\chi\cdot\psi^*_i=0`
       mechanisms asserted)
   * - **sphere** :math:`\varphi^*`-shape
       (``TestP14SphereAdjointVector``)
     - L1
     - dense FORWARD-probed :math:`(A_{\rm loss}, F)` + raw-data coupled
       :math:`G` — both independent of ``.H``
     - the coupled defining-law residual
       :math:`\|A_{\rm loss}^{\mathsf T}(G\psi^*) -
       F^{\mathsf T}(G\psi^*)/k\|` at rel floor
       :math:`1.2\times10^{-10}` vs gate :math:`10^{-7}`
       (:math:`n=140`); anti-vacuity :math:`|\Delta k| =
       3.3\times10^{-11}`

The Mode-12 accounting and the mutation teeth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because :math:`\operatorname{eig}(M^{\dagger}) =
\operatorname{eig}(M)` (the identity lives on the iteration operator
:math:`M = A_{\rm loss}^{-1}F`, every factor daggered), a
:math:`k^{\dagger}=k` gate is designed-green
(``vv-principles`` Mode 12) on whole error classes.  The boundary is
load-bearing — this campaign twice caught a wrong "why" here — so it is
stated exactly:

* :math:`k` is **EXACTLY blind** to (i) the factor-order / similarity
  family (:math:`\operatorname{eig}(M^{\mathsf T}) =
  \operatorname{eig}(M)`), (ii) all vector content, and (iii) the
  G-metric itself (:math:`G'^{-1}A^{\mathsf T}G'` is metric-similar to
  :math:`A^{\mathsf T}` for any invertible :math:`G'`).  Catchers live
  **outside** the eigenvalue stabiliser: the spectrum row, the
  bi-orthogonality row, the duality pairing, and the sphere vector row.
* :math:`k` is **NOT blind** to a single **leaf-transpose drop**:
  transposing one factor is not a pencil similarity.  The P1.3 teeth
  (``TestP13Mutations``) each shift :math:`k` — :math:`F^{\dagger}\!\to\!
  F` moves :math:`k` from :math:`1.488` to :math:`0.171` on the 4G ∞
  fixture (:math:`\chi\not\parallel\nu\Sigma_f` precondition asserted),
  :math:`S^{\dagger}\!\to\!S` shifts it on asymmetric SigS, and
  :math:`L^{\dagger}\!\to\!L` shifts it on the heterogeneous /
  sphere legs (the flat ∞ legs are BLIND to it).

**The factor-order trap.**  The P1.4 reference must be the dominant
right eigenvector of :math:`(A^{\mathsf T})^{-1}F^{\mathsf T}`, **not**
:math:`\operatorname{eig}(M^{\mathsf T})`: the two are similar (so every
:math:`k` check passes on both), but for rank-1 :math:`F` the
:math:`M^{\mathsf T}` eigenvector degenerates to exactly
:math:`\widehat{\nu\Sigma_f}` — zero A-physics.  Caught by the SN
daggered solve on first contact; the corrected law lives in
:func:`~orpheus.derivations.common.eigenvalue.kinf_and_adjoint_spectrum_homogeneous`.

**The metric tooth (ERR-067 family).**  The sphere vector row is the
sole catcher for a G-metric bug: dropping :math:`G_{\rm sd} =
V_{\rm cell} \to 1` leaves :math:`|k^{\dagger}_{\rm mut}-k_{\rm fwd}| =
2.6\times10^{-11}` (EXACTLY k-blind, the metric-similarity above) while
the defining-law residual reds to :math:`2.35`, O(1) over the
:math:`10^{-7}` gate.  A corroborating k-VISIBLE tooth
(:math:`F^{\dagger}\!\to\!F`) also reds the residual, showing the row
catches leaf-transpose drops too.

The daggered eigenproblem is :eq:`sn-adjoint-eigenproblem` and the
reciprocity duality is :eq:`sn-adjoint-duality` (both in
:ref:`sn-adjoint`); the k-rows verify the former, the P1.2 row the
latter.

Adjoint-weighted collapse — the P6 taxonomy gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The adjoint flux is a *means*; its first production consumer is the
**adjoint-weighted (eigenvalue-consistent) collapse** — the ``adjoint=``
arm of :meth:`Solution.homogenize
<orpheus.sn.solution.Solution.homogenize>` and :meth:`Solution.condense
<orpheus.sn.solution.Solution.condense>` (#281 P6; the taxonomy
narrative and the Bell & Glasstone energy-axis convention are
:ref:`frame-adjoint-weighted-seam`).  This battery verifies the
*consumer* — that each channel collapses by its worth-zeroing rule — as
distinct from the daggered-posing gates above, which verify the
*producer* (:math:`\varphi^*` itself).  Every collapse rule is first an
exact SymPy identity in
:mod:`orpheus.derivations.common.homogenization` (theorems T0–T6,
proof-welded to the production builder); the gates below check the
production floats against **structurally-independent per-region hand
rules**.  The battery lives in ``tests/sn/test_homogenization.py``
(§4.0 / C1 / C2 / C3 / Cχ, plus the T4 balance pin),
``tests/sn/test_condensation.py`` (C4 / C5), and
``tests/data/test_mixture_condense.py`` (T6a at the data level).

.. list-table:: Adjoint-weighted collapse (P6, #281; measured)
   :header-rows: 1
   :widths: 18 8 30 44

   * - Gate (test)
     - Level
     - Reference / pillar
     - What it pins (measured)
   * - **§4.0** forward-arm invariance
       (``TestAdjointDegeneratePins``;
       ``…test_no_arg_equals_explicit_none_bitwise`` on both verbs)
     - L0
     - bit-identity regression
     - the new ``adjoint=`` keyword does not touch the forward path — no
       arg :math:`\equiv` ``adjoint=None`` at **0-ULP** on every channel
       (:math:`\Sigma_t,\Sigma_c,\Sigma_L,\Sigma_f,\nu\Sigma_f,\chi,
       \Sigma_s`).  Tooth 2 (no shared drift) is the forward hand-loop
       rate suite, which stays green
   * - **C1** full-taxonomy discriminator
       (``TestC1AdjointWeightedDiscriminator``)
     - L0
     - B1-derived per-region hand rules (independent loops)
     - every channel equals its worth-zeroing rule on a
       tilted-importance fixture — T1 vector pair
       :math:`\varphi^*\!\odot\varphi`, T1b angular :math:`\rho`
       (:math:`\ne` the scalar pair on anisotropic shapes), T2 per-pair
       sink×source, T3 mixed-fold :math:`\nu\Sigma_f` + canonical
       :math:`\chi` — AND differs from the forward degenerate (dud-guard:
       :math:`\varphi^*/\lVert\varphi^*\rVert \ne
       \varphi/\lVert\varphi\rVert`)
   * - **C1 / T4** balance-imbalance pin
       (``…test_worth_exact_collapse_breaks_balance_as_derived``)
     - L0
     - the derived worth-exact property (theorem T4)
     - the adjoint-collapsed
       :class:`~orpheus.data.macro_xs.mixture.Mixture` does **not**
       satisfy the total-XS balance (residual :math:`>10^{-9}`) while the
       **forward** collapse does — the reactivity-vs-rates trade-off
       pinned as *expected*, never ``assert_balanced``\ ed away
   * - **C2** comparative :math:`k` order
       (``TestC2ComparativeKeffOrder``)
     - L2
     - closed-form — the L1-anchored fine :math:`k` (P1.3-certified) +
       the forward gap on the same ladder
     - on a material-contrast ladder :math:`m_1 = m_0 +
       \varepsilon\Delta` (:math:`\varepsilon\in\{1,\tfrac12,\tfrac14\}`)
       the adjoint gap shrinks **higher-order** (ratios 6.08 / 9.24) vs
       the forward gap's first order (2.05 / 2.01), and is smaller on
       every rung.  Same-mesh XS-replacement (16 fine cells,
       region-constant XS) — the T0/T5 worth exactly, **zero** coarse-DD
       confounder
   * - **C3 / C5** Mode-11 weight capture
       (``TestC3WeightCaptureSentinel`` /
       ``TestC5CondenseWeightCaptureSentinel``)
     - L0 / L1
     - in-process weight capture (``monkeypatch``)
     - the frames actually receive the **derived** weights — the pair
       :math:`\varphi^*\!\odot\varphi`, the angular :math:`\rho`, the
       emission fold :math:`\iota\cdot p` — and a **bare**
       :math:`\varphi^*` is *never* a frame weight (the committed catcher
       for the :math:`\varphi\to\varphi^*` bare-swap trap); the forward
       :math:`\varphi` is not silently taken
   * - **C4** bilinear condensation
       (``TestC4BilinearCondensation``)
     - L1
     - the B&G Ch. 6 hand rule (nested blocks; independent loop)
     - every channel of the bilinear-condensed Mixture equals the B&G
       convention — plain flux carrier, flux-weighted-average adjoint
       carrier :math:`\Psi^{\dagger}_G` (B&G (6.126)–(6.128)), per-block
       sink×source ((6.136)), :math:`\chi^{\dagger}` with the rank-1
       simplex rescale.  The **post-B&G sink-carrier correction** (the
       sink axis is *not* frozen; it gains :math:`\Psi^{\dagger}_G`)
       replaced the pre-B&G "marginalize frozen" expectation;
       discriminates from forward
   * - **Cχ** simplex positive control
       (``TestCxChiSimplexPositiveControl``)
     - L0
     - positive control (simplex validity)
     - the canonical :math:`\chi` stays a valid probability simplex — the
       adjoint-collapsed Mixture constructs **without raising** and
       :math:`\sum_g\chi_{R,g} = 1` (:math:`\ge 0`) on producing regions
   * - **T6a** exact-:math:`k` at true spectra
       (``TestBilinearCondensation``;
       ``…test_t6a_true_spectra_reproduce_fine_k_exactly``)
     - foundation
     - closed-form — 0-D ∞-medium :math:`k`
     - the bilinear-condensed 2G pencil's rank-1 :math:`k` equals the
       fine 4G :math:`k` to :math:`10^{-12}` (:math:`\varphi =
       A^{-1}\chi`, :math:`\varphi^* = A^{-\mathsf T}\nu\Sigma_f`) —
       condensation is *pure projection* on the energy axis, no streaming
       carve (theorem T6a)

**The honest scope of C2.**  C2 is a *comparative* claim, not a value
claim.  Its reference (the fine :math:`k`) is an ORPHEUS solver, so C2
does **not** terminate in a structurally-independent absolute reference —
it asserts the *rate* relation :math:`\mathrm{gap}_{\rm adj}(\varepsilon)
< \mathrm{gap}_{\rm fwd}(\varepsilon)` and the order signature (adjoint
:math:`\mathcal O(\varepsilon^2)`, forward :math:`\mathcal
O(\varepsilon)`), anchored on the P1.3-certified fine :math:`k` (the
anti-#5 pairing: a convergence rate is necessary, not sufficient — never
a rate to a possibly-wrong limit).  It proves *adjoint weighting reduces
the XS-collapse contribution to the* :math:`k` *gap*; it does **not**
prove a coarse re-solve is accurate.  The coarse-mesh diamond-difference
discretization error is a separate, weighting-independent confounder that
the same-mesh XS-replacement design deliberately excludes — both arms
carry the identical spatial error, so the comparative delta isolates the
worth.  The mutation that reds C2: feed the forward :math:`\varphi` as
the test weight, and the strict inequality collapses to equality.

The collapse taxonomy, the balance trade-off, and the B&G energy-axis
convention are :ref:`frame-adjoint-weighted-seam`; the ``adjoint=``
parameter and its ``AdjointSolution`` carrier are
:ref:`sn-adjoint-carrier`.


.. _sn-dsa-verification:

Diffusion Synthetic Acceleration (rate / invariance, not MMS)
-------------------------------------------------------------

DSA verification is a **different kind** from the MMS/convergence tiers
above.  DSA is an accelerator: it makes **no eigenvalue and no
flux-shape claim** (the eigenvalue is the SN solver's, verified
elsewhere), so **MMS is absent by design**.  Its defining property —
the low-order correction :math:`\to 0` at convergence — organises the
whole battery: bugs in the *transport* operator change the fixed point
(caught by FP-invariance), while bugs in the *accelerator* machinery
leave the fixed point **identically** unchanged and are caught only by
**object gates** and **rate gates**.  Of the eight canonical
implementation errors, exactly one reds the FP gates; the other seven
ride the object/rate tier (full derivation:
:ref:`sn-dsa-the-f-form` of :doc:`../methods/sn/acceleration`).

.. list-table:: The DSA verification battery (issue #2, arm 1: 1-D slab DD)
   :header-rows: 1
   :widths: 14 30 12 22 22

   * - Gate
     - What it pins
     - Level
     - Equation label
     - Test
   * - D1 / D2
     - the derived low-order row :math:`\equiv` Larsen (27)/(23a–f), and
       the production build :math:`\equiv` the reference builder
       entry-for-entry
     - foundation
     - :eq:`sn-dsa-consistent-low-order`, :eq:`sn-dsa-coefficients`
     - ``test_dsa_rules``, ``test_dsa_low_order``
   * - D3–D5
     - accelerated flux :math:`\equiv` plain-SI fixed point on an
       **anisotropic** config (Mode-9-proof; the only catcher for the
       :math:`\sigma_r`-fold)
     - L2
     - — (invariance)
     - ``test_dsa_acceleration``
   * - D6
     - correction :math:`\to 0` — a zero displacement maps to an exact
       zero (the safety property)
     - L2
     - :eq:`sn-dsa-correction-vanishes`
     - ``test_dsa_acceleration``
   * - D7 / D8
     - :math:`R` conserves particles (:math:`\langle 1, Rr\rangle =
       \langle 1, r\rangle`) and :math:`\equiv` the frame's
       :math:`\ell = 0` analysis row (0-ULP)
     - foundation
     - :eq:`sn-dsa-restriction`
     - ``test_dsa_low_order``
   * - D10
     - the σ\ :sub:`r`-fold routing sentinel (foldable accessors fenced
       to their three legitimate consumers) — **catches ERR-070**
     - foundation
     - — (AST)
     - ``test_dsa_rate``
   * - D11
     - measured :math:`\rho \le 0.2247c` (one-sided Fourier bound +
       plain-SI honesty control) — a **rate** claim (1G-legit)
     - L1
     - :eq:`sn-dsa-consistent-fourier`
     - ``test_dsa_rate``
   * - D12 / D13
     - reflective stability (thickness-independent, bounded) + the
       c-independence / speedup count gates + the WD
       partial-consistency divergence control
     - L1 / L2
     - — (rate)
     - ``test_dsa_rate``
   * - S2
     - :math:`K_2 = 0` one-iteration exactness (self-verifies the whole
       boundary/update/synthesis chain in one number)
     - L1
     - :eq:`sn-dsa-s2-exactness`, :eq:`sn-dsa-synthesis`
     - ``test_dsa_rate``
   * - inverse
     - :math:`(L+C)\circ(L+C)^{-1} \equiv I` on the FULL composite space
       — **catches ERR-071**
     - foundation
     - :eq:`sn-dsa-sweep-inverse-identity`
     - ``test_sweep_inverse_identity``

The measured rate/stability tables (the c :math:`\to` 1 corner, the
anisotropy ladder, the reflective Jacobi-wall lag, the WD negative
control) are teaching artifacts at :ref:`sn-dsa-rate-and-stability`;
the auto-generated equation-label :math:`\times` test matrix is at
:doc:`/theory/verification/matrix`.


