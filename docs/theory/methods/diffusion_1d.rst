.. _theory-diffusion-1d:

==================
1D Diffusion (P1)
==================

The diffusion equation is the lowest-order angular approximation of
the neutron transport equation: the P1 (first spherical-harmonic)
expansion truncated to the :term:`scalar flux`. It is not a transport
solver in the strict sense — it discards angular information
beyond the current — but it is the workhorse of reactor design and
its verification is a mathematical problem in its own right.

This page documents the ORPHEUS diffusion module on two levels. It
carries the **production operator-family architecture** (#290 — how
the solver poses and inverts the multigroup diffusion criticality
problem in the shared operator algebra
:math:`(L + C - S - B)\psi = \tfrac1k F\psi`; the four-term spelling is
this solver's own, because it sums its two isotropic energy leaves into
one :math:`S` at its composition site and so has no separate
:math:`N_{2n}` term — :ref:`the two collision gains
<operator-algebra-two-gains>`) *and* the **continuous
reference solutions** with the equation labels the verification tests
point at. The operators are 1-D but coordinate-general (slab /
cylinder / sphere through the mesh's own face areas and cell volumes);
a multi-dimensional stencil is a deliberate extension seam, refused at
mesh construction. The analytical references below are plane-slab.

.. note::

   **The MATLAB-port island is gone.** Until #290 the module was a
   443-line MATLAB port — raw ``(2, n_cells)`` arrays, a scipy
   BiCGSTAB inner solver, its own ``TwoGroupXS`` / ``CoreGeometry``
   containers, string boundary keys, and a hardcoded 2-group
   ``sig_s[::-1]`` down-scatter flip. It was retired at #290 P6;
   ``orpheus.diffusion.solver`` now *is* the modern operator-algebra
   solver (family-naming parity with ``sn`` / ``cp`` / ``homogeneous``).
   The :ref:`diffusion-2rg-investigation-history` section below is the
   preserved post-mortem of the *reference-solution* dead ends — the
   island-specific solver dead end (#4) is flagged as history that no
   longer applies to the exact-LU modern path.

.. contents::
   :local:
   :depth: 2


Key Facts
=========

- **Classical form.** The 1-D multigroup diffusion equation in plane
  geometry is a second-order elliptic boundary-value problem in
  :math:`\phi_g(x)`:

  .. math::
     :label: diffusion-operator

     -\frac{d}{dx}\!\left(D_g(x)\,\frac{d\phi_g}{dx}\right)
       + \Sigma_{r,g}(x)\,\phi_g
       = \sum_{g' \ne g} \Sigma_{s,g' \to g}\,\phi_{g'}
       + \frac{1}{k}\,\chi_g\sum_{g'}\nu\Sigma_{f,g'}\,\phi_{g'}.

- **Operator-family form (production, #290).** ORPHEUS does *not*
  discretise :eq:`diffusion-operator` term-by-term. It poses the
  criticality problem in the shared operator algebra — the same
  loss composite the S\ :sub:`N` solver introduced — as

  .. math::
     :label: diffusion-operator-family

     \underbrace{(L + C - S - B)}_{A}\,\psi \;=\; \frac{1}{k}\,F\,\psi ,

  acting on the scalar composite field :math:`\psi`, with **no separate**
  :math:`N_{2n}` **term** — see the grouping note below. Here :math:`L` is
  the elliptic **leakage** leaf :math:`-\nabla\!\cdot D\nabla`,
  :math:`C` the **collision** multiplication by :math:`\Sigma_t`,
  :math:`S` the **collision-gain** kernels, :math:`B` the realized
  **boundary** :term:`albedo` block, and :math:`F` the rank-1 fission
  production :math:`\chi\otimes\nu\Sigma_f`. The removal cross section

  .. math::
     :label: diffusion-removal-xs

     \Sigma_{r,g} \;=\; \Sigma_{t,g} - \Sigma_{s,g\to g}

  is then a **theorem** — the in-group cancellation between :math:`C`
  and :math:`S`, :math:`\mathbf 1^{\mathsf T}(C - S) = \Sigma_a` — never
  an assembled input. See :ref:`diffusion-operator-family-section`.

- **Why four terms and not five — the grouping, not the member list.**
  The *general* within-group composite carries **two** collision gains,
  :math:`A = L + C - S - N_{2n} - B` (:eq:`sn-within-group-with-n2n`),
  because the :math:`(n,2n)` channel's bundling is context-dependent and
  is not decided at the operator level
  (:ref:`the two collision gains <operator-algebra-two-gains>`). This
  solver decides it **at its own composition site**, summing the two
  isotropic energy leaves
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicScattering`
  ``+``
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicN2N`
  into the single :math:`S` of :eq:`diffusion-operator-family` before
  composing the loss. So diffusion's :math:`S` *is* scattering **and**
  the :math:`(n,2n)` emission, its four-term spelling is **exact**
  rather than a :math:`\Sigma_{2n} \equiv 0` simplification, and the
  difference from S\ :sub:`N` is a
  :class:`~orpheus.numerics.operator.OperatorSum` grouping of the same
  two members — not a different member list.

- **Diffusion coefficient.** In each region :math:`D` is built from the
  transport cross section:

  .. math::
     :label: diffusion-coefficient

     D_g(x) \;=\; \frac{1}{3\,\Sigma_{\text{tr},g}(x)} ,
     \qquad
     \Sigma_{\text{tr},g} = \Sigma_{t,g}
       - \sum_{g'}\Sigma_{s1,\,g\to g'} .

  The transport correction is the *outflow* P1 approximation; when the
  mixture carries no P1 moment, :math:`\Sigma_{\text{tr}} = \Sigma_t`
  **exactly** — the correct isotropic-scattering limit, not a fallback.
  See :ref:`diffusion-data-seam`.

- **Discretisation.** Cell-centred finite difference with harmonic-mean
  face conductance (equivalent to lowest-order Raviart–Thomas with mass
  lumping); the design spatial order is :math:`\mathcal O(h^{2})` for
  smooth cross sections and :math:`\mathcal O(h)` at material interfaces
  that do not lie on cell faces.

- **Boundary conditions are the albedo family on the partial-current
  trace.** The boundary state is the per-face, per-group pair of
  half-range partial currents :math:`(J^+, J^-)` (outflow, inflow); a
  linear homogeneous diffusion boundary law collapses to a single
  scalar response per face,

  .. math::
     :label: diffusion-albedo-law

     J^- \;=\; \mathcal{A}\,J^+ .

  .. list-table:: Diffusion boundary laws (the albedo family)
     :header-rows: 1
     :widths: 20 10 70

     * - Law (``BC`` tag)
       - :math:`\mathcal{A}`
       - Physics
     * - ``"vacuum"``
       - :math:`0`
       - **Marshak**: zero incoming current :math:`J^- = 0`. In
         :math:`(\phi, J)` variables this is the Robin condition
         :math:`\phi + 2 D\,\partial_n\phi = 0` (Marshak 1947).
     * - ``"reflective"``
       - :math:`1`
       - zero net current :math:`J^+ = J^-` — a symmetry plane.
     * - ``"albedo"`` (:math:`\alpha`)
       - :math:`\alpha`
       - partial return :math:`J^- = \alpha J^+`, physical range
         :math:`\alpha \in [0, 1]`.
     * - ``"zero_flux"``
       - :math:`-1`
       - Dirichlet :math:`\phi_\Gamma = 0` (since
         :math:`\phi_\Gamma = 2(J^+ + J^-)`, this is
         :math:`J^- = -J^+`) — a mathematical idealisation, deliberately
         outside the physical :math:`[0, 1]` range.
     * - ``"white"``
       - :math:`1`
       - coincides with reflective at the P1 level (the angular
         redistribution distinguishing them integrates out of the
         half-range moments); deliberately **absent** from the diffusion
         registry — declare ``reflective``.

  .. important::

     **Vacuum means** :math:`J^- = 0` **(Marshak), not**
     :math:`\phi = 0`. The retired MATLAB island registered a
     hard-Dirichlet :math:`\phi = 0` wall under the key ``"vacuum"`` —
     unfaithful naming (a zero-flux wall returns a *negative* incoming
     current, which no vacuum does). Ruling 3 of #290 corrected this:
     vacuum IS the Marshak zero-incoming-current condition, and the
     zero-flux Dirichlet idealisation became its own honestly-named law
     ``BC("zero_flux")`` (:math:`\mathcal{A} = -1`). The analytical
     sine references on this page satisfy :math:`\phi = 0` and are
     therefore **zero-flux** references — re-attributed accordingly at
     #290 P6, with the mathematics unchanged (see
     :ref:`diffusion-1g-bare-slab`). The faithful vacuum (Marshak
     :math:`J^- = 0`) is verified today by property-level gates
     (:math:`J^- = 0` at the solution, :math:`k` strictly bracketed
     between the zero-flux and reflective values); an analytical
     Robin-face reference is the close-out follow-up **#293**.

  Boundary conditions are **declared on the mesh axes** — each endpoint
  carries a :class:`~orpheus.geometry.mesh.BC` tag — and **realized at
  construction**: promoting a mesh to a
  :class:`~orpheus.diffusion.augmented_mesh.DiffusionMesh` resolves each
  face's tag into its typed law and then the albedo operator, through
  the same shared
  :func:`~orpheus.transport.method.resolve_boundary_conditions` body the
  S\ :sub:`N` mesh uses (#290 P7). A solver never carries a boundary
  registry; a diffusion phase space with unresolved boundary conditions
  is unrepresentable. The three-layer decomposition (trace structure /
  physical law / method realisation) is documented on
  :doc:`/theory/foundations/boundary_conditions`.


.. _diffusion-operator-family-section:

Operator-family architecture (production)
=========================================

The production solver (#290) does not carry a diffusion-specific matvec.
It composes the loss :math:`A = L + C - S - B` — the general
:math:`L + C - S - N_{2n} - B` with this solver's two isotropic energy
leaves summed into the one :math:`S` — and the gain :math:`F`
from the shared operator algebra of :doc:`/theory/foundations/operator_algebra`,
inverts :math:`A` exactly, and drives the outer power iteration through
the *same* :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver`
protocol boundary the S\ :sub:`N`, CP, and homogeneous solvers plug
into. This section is the design record: what each leaf is, why the
resolvent is an explicit inverse rather than an operator splitting, how
the phase space and the boundary laws are layered onto the mesh, and the
gates that pin it.

The scalar composite
--------------------

Diffusion acts on the **scalar composite** field

.. math::
   :label: diffusion-scalar-composite

   \psi \;=\; \text{FullField}\bigl(\text{bulk}=\phi,\;
              \text{boundary}=(J^+, J^-)\bigr),

whose bulk is the group-resolved scalar flux
:class:`~orpheus.transport.fields.scalar_flux.ScalarFlux` and whose
boundary is the per-face partial-current trace
:class:`~orpheus.transport.fields.scalar_boundary_flux.ScalarBoundaryFlux`.
The trace degrees of freedom are honestly **part of the eigenvector**:
the converged mode carries its boundary partial currents, typed and
inspectable, rather than reconstructing them by a post-processing
gradient (the island's approach). Bulk and trace couple through the
half-range P1 dictionary

.. math::
   :label: diffusion-partial-current-dictionary

   J^\pm \;=\; \frac{\phi_\Gamma}{4} \pm \frac{J}{2},
   \qquad
   \phi_\Gamma = 2(J^+ + J^-),
   \qquad
   J = J^+ - J^- ,

under the same :math:`\lvert\Omega\cdot\hat n\rvert\,w` half-range
metric as the S\ :sub:`N` angular trace — degenerated here to the bare
face **area** because the currents are already angle-integrated. (This
shared metric is exactly what makes the future DSA restriction
well-posed; see :ref:`diffusion-dsa-seam`.) Within the pair
:math:`J = J^+ - J^-` is **exact** for any angular distribution, while
:math:`\phi_\Gamma = 2(J^+ + J^-)` holds only under the P1 closure —
the reason the flux accessor is fenced ``p1_boundary_scalar_flux`` while
``net_current`` is unprefixed.

.. _diffusion-leakage-boundary-leaves:

The leakage leaf L and the boundary block B
-------------------------------------------

:math:`L` (:class:`~orpheus.diffusion.operators.LeakageOperator`,
``BlockRole.FULL``) is the elliptic sibling of the S\ :sub:`N` streaming
operator: the **one** leaf that couples bulk :math:`\leftrightarrow`
boundary. Its bulk rows are the conservative finite-difference
divergence :math:`[(A_f J)_{i+1/2} - (A_f J)_{i-1/2}]/V_i` with the
current-continuous interior face current

.. math::
   :label: diffusion-interior-conductance

   J_f \;=\; -\,g_f\,(\phi_R - \phi_L),
   \qquad
   g_f \;=\; \Bigl(\frac{h_L}{2 D_L} + \frac{h_R}{2 D_R}\Bigr)^{-1} ,

the series sum of two half-cell resistances (the harmonic-mean-of-\
:math:`D` form). Interior currents stay **condensed** — never trace
degrees of freedom — and the same
:meth:`~orpheus.diffusion.operators.LeakageOperator.face_currents`
reconstruction that :meth:`~orpheus.diffusion.operators.LeakageOperator.apply`
consumes for the divergence is what serves the production current
profile (a single source, bit-identical). Written with the mesh's own
face areas :math:`A_f` and cell volumes :math:`V_i`, the *same* body is
correct on slab (:math:`A_f \equiv 1`, :math:`V = h`), cylinder, and
sphere (whose pole is not a face — the :math:`r = 0` slot simply carries
no trace flow, :math:`A(0)\,J = 0`).

Only the boundary :math:`(J^+, J^-)` pair survives as trace unknowns.
:math:`L`'s trace rows carry the **outflow-definition defect** on the
outflow row and the **inflow identity** on the inflow row,

.. math::
   :label: diffusion-boundary-closure

   \text{outflow:}\quad J^+ - c_\phi\,\phi_e - c_{J^-}\,J^- = 0,
   \qquad
   \text{inflow:}\quad J^- ,
   \qquad
   c_\phi = \frac{1}{\rho + 2},\;\;
   c_{J^-} = \frac{\rho - 2}{\rho + 2},\;\;
   \rho = \frac{h_e}{2 D_e},

from Fick's law between the edge-cell centre and the face plus the
dictionary :eq:`diffusion-partial-current-dictionary`. :math:`B`
(:class:`~orpheus.diffusion.operators.DiffusionBoundaryOperator`,
``BlockRole.BOUNDARY``) is the realized albedo block: it emits
:math:`\mathcal{A}\,J^+` on the inflow row and nothing else, so that
:math:`(L - B)` reads the boundary law :math:`J^- - \mathcal{A}\,J^+ = 0`
exactly. Eliminating :math:`(J^+, J^-)` by a Schur complement onto the
bulk recovers the classic condensed closures: zero-flux
(:math:`\mathcal{A} = -1`) gives :math:`J_{\rm net} = \phi_e/\rho` — the
island's ``φ_0/(0.5·Δz)`` "vacuum" arm — reflective (:math:`\mathcal{A}
= 1`) gives :math:`J_{\rm net} = 0`, and Marshak vacuum
(:math:`\mathcal{A} = 0`) the :math:`d = 2D` extrapolation. The
composite is the **un-condensed** spelling of the same algebra, with the
boundary law factored into its own operator.

The shared C / S / F arms
-------------------------

Everything except :math:`L` and :math:`B` is the shared transport
algebra, given scalar-composite arms at #290 P4 — *not* re-implemented:

- :math:`C` — the collision leaf, a
  :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
  over :math:`\Sigma_t`;
- :math:`S` — the isotropic scattering **K_iso** kernels
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicScattering`
  and
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicN2N`
  (the single source of :math:`\Sigma_{s0}^{\mathsf T}\phi` across every
  solver — never a diffusion-local reimplementation);
- :math:`F` — the shared rank-1 dyad
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`,
  :math:`\chi \otimes \nu\Sigma_f` (the fission **energy** binding; the
  angular binding
  :class:`~orpheus.transport.operators.fission.FissionOperator` is
  S\ :sub:`N`'s and refuses a scalar carrier — CS4c step 4,
  :ref:`sn-fission-binding-adjoint`).

The removal cross section is therefore a **theorem, not an input**: the
in-group cancellation :math:`\mathbf 1^{\mathsf T}(C - S) = \Sigma_a`
reproduces :math:`\Sigma_r = \Sigma_t - \Sigma_{s,gg}` by column-sum,
gated directly (:ref:`diffusion-operator-family-verification`). The
:math:`(n,2n)` channel is **loss-side** — :math:`S` carries the full
K_iso pair :math:`\Sigma_{s0}^{\mathsf T} + 2\Sigma_2^{\mathsf T}` while
:math:`F` and the production rate are :math:`\nu\Sigma_f` only, mirroring
the homogeneous solver (:doc:`/theory/foundations/infinite_medium`).

.. note::

   **The two solvers agree on the SIDE and differ only in the
   GROUPING.** ⛔ Until 2026-09-07 this paragraph continued *"S*\
   :sub:`N` *poses* :math:`(n,2n)` *production-side instead — both are
   consistent posings of the same balance"*. That was true when written
   and was retired by **ERR-065** (R7, 2026-07-03), which ruled the
   estimator and the posed problem may not disagree: S\ :sub:`N`'s
   :meth:`~orpheus.sn.solver.SNSolver.compute_keff` now carries a
   :math:`\nu\Sigma_f`-only numerator and subtracts the emission
   :math:`E_{2n}` from the denominator
   (:math:`R_{\nu\Sigma_f}/k = R_{\Sigma_a} + L - E_{2n}`), i.e.
   **loss-side, exactly as here**. Since CS4c step 3 (2026-08-30) the
   operator tier says the same thing structurally — S\ :sub:`N`'s
   composite is :math:`A = L + C - S - N_{2n} - B`
   (:eq:`sn-within-group-with-n2n`), with the channel subtracted into
   the loss. What still differs is only *how the two members are
   grouped*: S\ :sub:`N` keeps them as two operators, diffusion sums
   them into one :math:`S`
   (:ref:`the two collision gains <operator-algebra-two-gains>`).
   :meth:`~orpheus.sn.solver.SNSolver.compute_production_rate`
   deliberately keeps fission **plus** emission — it is the ERR-052
   renormalisation scale anchor, a different role from the k numerator.

Because the group coupling lives entirely in the
shared kernels, the solver is **ng-generic by construction**: the
island's hardcoded 2-group ``sig_s[::-1]`` down-scatter flip is
structurally dead, which unblocks arbitrary-group diffusion (#33 / #34).

The resolvent — exact inverse, not operator splitting
-----------------------------------------------------

The within-outer inner solve is the campaign-ruled **explicit dense
inverse**

.. code-block:: python

   template  = FullField.zeros(bulk=ScalarFlux, boundary=ScalarBoundaryFlux, mesh=mesh)
   resolvent = MatrixInverseOperator(FlattenedOperator(A, template))

— one eager LU factorisation of the flattened composite operator at
construction, one back-substitution per outer iteration. No scattering
inner iteration exists at all: the loss :math:`A` already carries the
full multigroup coupling (:math:`S` is subtracted *into* it), so
``solve_fixed_source`` is a single application of :math:`A^{-1}`. The
:class:`~orpheus.numerics.flat_operator.FlattenedOperator` bridges the
typed composite to the flat vector the dense
:class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
needs; the composite space's flat ``shape`` feeds ``as_matrix`` its
basis dimension for free.

.. warning::

   **Do not route the diffusion resolvent through the structure-keyed**
   ``A.inverse()``. The taxonomy's structure-keyed inverse realises
   :math:`A^{-1}` for a sum :math:`A = M - N` as a Green / Neumann
   splitting :math:`\sum_k (M^{-1}N)^k M^{-1}` — a *stationary
   iteration* that converges only when the spectral radius
   :math:`\rho(M^{-1}N) < 1`. For the triangular **sweep** inverses (S\
   :sub:`N`, MoC) that splitting is nilpotent (:math:`\rho = 0`) and
   exact in one pass. A symmetric-**elliptic** (discrete-Laplacian)
   diffusion operator admits no such convergent split: the discrete
   Laplacian is ill-conditioned (condition number :math:`\sim h^{-2}`)
   and the loss splitting is not a contraction, so the campaign found
   the Neumann series **diverges** on fine meshes. The explicit dense
   inverse is exact regardless of conditioning — the same choice, and
   the same ``MatrixInverseOperator(loss) @ F`` precedent, the 0-D
   homogeneous solver makes (:doc:`/theory/foundations/infinite_medium`).

.. _diffusion-data-seam:

The data seam — D from the transport cross section
--------------------------------------------------

:math:`L` reads its per-cell diffusion coefficient through the #290 P1
data seam: :attr:`Mixture.diffusion_coefficient
<orpheus.data.macro_xs.mixture.Mixture.diffusion_coefficient>`
:math:`= 1/(3\,\Sigma_{\text{tr}})` built on the outflow transport cross
section :attr:`Mixture.transport_xs
<orpheus.data.macro_xs.mixture.Mixture.transport_xs>`
:math:`\Sigma_{\text{tr},g} = \Sigma_{t,g} - \sum_{g'}\Sigma_{s1,g\to g'}`
(:eq:`diffusion-coefficient`). When a mixture carries no P1 moment the
out-scatter row sum is identically zero and :math:`\Sigma_{\text{tr}} =
\Sigma_t` **exactly** — the correct isotropic limit.

Legacy diffusion tables (the MATLAB ``CORE1D`` schema — per-group
``transport`` / ``absorption`` / ``fission`` / ``chi`` / ``scattering``
vectors) are mapped onto the canonical
:class:`~orpheus.data.macro_xs.mixture.Mixture` by the **one** encoder
:func:`~orpheus.derivations.common.xs_library.mixture_from_diffusion_tables`.
Ruling 4 of #290 fixed this encoding to be **bit-identical**:
``SigT := transport`` with **no** P1 moment, so
:math:`\Sigma_{\text{tr}} = \Sigma_t` and :math:`D = 1/(3\cdot\text{transport})`
reproduces the island's coefficient to the last bit, leaving every
analytical reference unmoved; in-group scatter is back-filled so removal
= absorption + down-scatter. The physical alternative
(:math:`\Sigma_{s1} = \bar\mu\,\Sigma_{s0}` with the *true* :math:`\Sigma_t`)
re-baselines :math:`D` and every downstream reference — a deliberate
close-out follow-up (**#292**), not this encoder.

Mesh and Protocol layering
--------------------------

The phase space is a
:class:`~orpheus.diffusion.augmented_mesh.DiffusionMesh` (#290 P7a): a
:class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` (method-\
agnostic mesh + materials **data**) augmented with the diffusion method's
**behaviour** — the scalar trace, the composite carrier
``full_field_space``, and the per-face boundary laws **realized at
construction**. It is the structural sibling of
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` (mesh + :term:`quadrature` +
:term:`sweep` machinery + angular trace): one method-agnostic data carrier, one
method layer per transport method. Every admission gate (1-D, bounded
geometry, supported BC tags) fires at construction — an operator built
on a bad phase space is action-at-a-distance otherwise — so a diffusion
phase space with unresolved or unrealizable boundary conditions is
**unrepresentable**.

Both method-meshes are the two witnesses of the
:class:`~orpheus.transport.method.TransportMethod` Protocol (#290 P7b),
and BC resolution flows through the **one** shared
:func:`~orpheus.transport.method.resolve_boundary_conditions` body — the
same face-loop, reflective-default, and tag-to-law parse for S\ :sub:`N`
and diffusion alike; only each mesh's ``realize_boundary_law`` arm
differs (the diffusion arm builds the albedo operator via
:class:`~orpheus.diffusion.boundary_realizer.DiffusionBoundaryRealizer`).
Conformance is structural — neither mesh imports the Protocol. The
realizer mechanics (the three-layer descriptor / law / realizer
decomposition and the rank-N composition walker) live on
:doc:`/theory/foundations/boundary_conditions`.

.. _diffusion-solver-engines:

The k-eigenvalue solver on the shared engines
---------------------------------------------

:class:`~orpheus.diffusion.solver.DiffusionSolver` implements the shared
:class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` +
``ProductionRateSolver`` protocol over the **flat composite vector** (the
trace is part of the eigenvector; conversion happens at exactly two
sites). The public driver
:func:`~orpheus.diffusion.solver.solve_diffusion_1d` takes
``materials`` + a ``Mesh1D`` — a zoned core is just a ``Mesh1D`` with
multi-valued ``mat_ids`` (the island's ``CoreGeometry`` container had no
independent content) — promotes it to a
:class:`~orpheus.diffusion.augmented_mesh.DiffusionMesh`, assembles the
family, and drives
:func:`~orpheus.numerics.eigenvalue.power_iteration`.

- **The k-update is the integrated eigenvalue relation**
  :math:`k = P(\psi)/\langle 1, (A\psi)_{\rm bulk}\rangle_V` — production
  over the volume-integrated bulk loss rate. By the column-sum theorem
  :math:`\mathbf 1^{\mathsf T}(C-S)=\Sigma_a` and the telescoping of
  :math:`L`'s conservative divergence (interior flows cancel; boundary
  flows survive as leakage), the denominator decomposes as absorption +
  leakage :math:`-` net :math:`(n,2n)` gain — the island's
  ``p_rate/(a_rate + leakage)``, but derived *through* the loss operator
  that defines the fixed point, so no term can be forgotten (the leakage
  is structural, not hand-added).
- **Normalisation** rides the typed
  :class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`
  (the #270 diffusion arm; the ERR-052 renormalisation anchor):
  :func:`~orpheus.numerics.eigenvalue.power_iteration` renormalises each
  iterate to unit production rate, so the returned mode carries
  :math:`\int_V \nu\Sigma_f\,\phi\,dV = 1`. The island's hardcoded
  ``e_per_fission`` power window and its ``fi /= max`` conditioning hack
  are both retired by this contract.
- **Cross-engine gate.** The dense
  :func:`~orpheus.numerics.eigenvalue.direct_eigenvalue`
  (:math:`k = \lambda_{\max}(A^{-1}F)`) and the iterative
  :func:`~orpheus.numerics.eigenvalue.power_iteration` are cross-checked
  at :math:`10^{-10}` on the materialised :math:`(A, F)` pair, across the
  albedo family — the committed catcher for all protocol plumbing.

The RT0 equivalence and the mixed-form seam
-------------------------------------------

The cell-centred finite-difference stencil above **is** lowest-order
Raviart–Thomas (RT0) with mass lumping (the Baliga–Patankar
equivalence): the harmonic-mean face conductance
:eq:`diffusion-interior-conductance` is the series half-cell resistance
an RT0 flux space produces, and mass-lumping the RT0 mass matrix
collapses the mixed system back to the two-point scalar stencil. The
full (un-lumped) mixed form — carrying the interior face currents as
live degrees of freedom rather than condensing them — is a documented
extension seam (**#294**); its trigger is off-face material interfaces
that need :math:`\mathcal O(h^2)` accuracy, where the mass-lumped
approximation drops to :math:`\mathcal O(h)`.

.. _diffusion-dsa-seam:

The DSA seam
------------

This operator family *is* the :math:`A_{\rm diff}` that a consistent
**diffusion-synthetic accelerator** for S\ :sub:`N` (Issue #2, the named
high-priority DSA consumer) will precondition: an in-algebra
:class:`~orpheus.numerics.operator.LinearOperator` on the scalar
composite, invertible by the explicit
:class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`,
whose low-order correction :math:`\to 0` at convergence (so DSA is
correctness-safe *by construction*, changing only the iteration rate).
The construction path is direct: an
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` promotes straight to a
:class:`~orpheus.diffusion.augmented_mesh.DiffusionMesh`
(``DiffusionMesh.from_material_mesh(sn_mesh)`` — an SNMesh *is a*
MaterialMesh), so :math:`A_{\rm diff}` assembles over the **same** axes,
materials, and BC declarations as the SN sweep it accelerates. The
SN\ :math:`\to`\ diffusion boundary restriction is the :math:`\ell = 0`
half-range moment of the angular trace under the shared
:math:`\lvert\Omega\cdot\hat n\rvert\,w` metric — the reason ruling 2
posed the trace in partial-current variables. See
:doc:`/theory/methods/sn/acceleration` for the DSA-consumer discussion
and the seam contract — and, in particular, why the consistent
accelerator is **not** :math:`A_{\rm diff}` but the derived
edge-centered SN-side system (ruling R4: :math:`A_{\rm diff}` is the
right standalone discretisation and a measured-divergent *accelerator*).

.. _diffusion-operator-family-verification:

Numerical evidence (the operator-family gates)
----------------------------------------------

The architecture is pinned by object-level and cross-engine gates in
``tests/diffusion/`` (the continuous-reference convergence study and the
MMS gate are in :ref:`diffusion-2rg-verification` and
:ref:`diffusion-mms-section`):

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Gate
     - What it pins
   * - **Object-level stencil gate** (Mode-12 companion)
     - :math:`A`\ ``.as_matrix()`` :math:`\equiv` an independently
       hand-posed finite-difference matrix on a heterogeneous,
       non-uniform, 2-group slab — mutation-verified **RED** under a
       D-face pairing swap, :math:`\Sigma_a`-for-:math:`\Sigma_t`
       confusion, a scatter transpose, and a closure-coefficient sign
       flip. Pins the *object*, not just its spectrum.
   * - **Cross-engine consistency**
     - :math:`\lvert\Delta k\rvert < 10^{-10}` between
       ``direct_eigenvalue`` and ``power_iteration`` (and the full
       composite eigenvector), across the albedo family.
   * - **L2 infinite medium**
     - reflective diffusion :math:`k \equiv` the homogeneous
       :math:`k_\infty` — itself the closed-form multigroup
       infinite-medium eigenvalue :math:`\lambda_{\max}(\Sigma_a^{-1}F)`,
       not merely another solver — at 2G **and** 3G asymmetric; the
       3-group case is the flip-trick discriminator the island
       structurally could not represent (ng-generic multigroup coupling).
   * - **CORE1D legacy bridge**
     - modern :math:`k \equiv` island :math:`k` at :math:`10^{-8}` under
       the ruling-4 bit-identical encoding (the exact-LU solver drove the
       retired island *past* its 200-outer cap to a converged
       comparison; the island's own driver never converged the PWR
       dominance ratio).
   * - **Per-law trace semantics**
     - at the solution, vacuum :math:`J^- = 0`, albedo
       :math:`J^- = \alpha J^+`, reflective :math:`J_{\rm net} = 0`,
       zero-flux :math:`\phi_\Gamma = 0` — all LU-exact; :math:`k`
       strictly monotone in :math:`\mathcal{A}`; the integrated balance
       identity :math:`P/k =` absorption + leakage.
   * - **Demo**
     - :func:`~orpheus.diffusion.solver.solve_diffusion_1d` reproduces
       the MATLAB reference :math:`k = 1.022173` (at print precision)
       under ``BC("zero_flux")``.

.. (vv-status rationale) The #290 operator-family labels are
   representational / definitional, NOT solver claims: the posing
   identity (diffusion-operator-family), the removal-XS in-group
   cancellation theorem :math:`\mathbf 1^{\mathsf T}(C-S)=\Sigma_a`
   (diffusion-removal-xs), the scalar-composite field-typing identity
   (diffusion-scalar-composite), the P1 half-range partial-current
   dictionary (diffusion-partial-current-dictionary), the albedo
   boundary-family law :math:`J^-=\mathcal{A}J^+` (diffusion-albedo-law),
   and the two discretization formulas (diffusion-interior-conductance /
   diffusion-boundary-closure). Their verifiable content is pinned by the
   object-level stencil gate (which monkeypatches the module-level
   ``_interior_conductance`` / ``_boundary_closure`` kernels), the
   cross-engine consistency gate, the field-construction foundation gates,
   and the per-law trace-semantics gates in the table above (the vacuum
   :math:`J^-=0` realization is asserted by
   ``tests/diffusion/test_properties.py::test_vacuum_means_zero_incoming_current``);
   the k-value carries no new claim here (it is verified by those gates
   plus the L1 / L2 anchors below). These sentinels are co-located here,
   with their #290-family siblings, rather than at each label's Key-Facts
   point of use (same-file, per the audit contract).
.. vv-status: diffusion-operator-family documented
.. vv-status: diffusion-removal-xs documented
.. vv-status: diffusion-scalar-composite documented
.. vv-status: diffusion-partial-current-dictionary documented
.. vv-status: diffusion-albedo-law documented
.. vv-status: diffusion-interior-conductance documented
.. vv-status: diffusion-boundary-closure documented


.. _diffusion-1g-bare-slab:

1-group bare slab
=================

The simplest verification configuration: a homogeneous slab of
thickness :math:`L` with **zero-flux** boundaries
(``BC("zero_flux")``, :math:`\phi = 0` — the honestly-named Dirichlet
law of ruling 3, *not* the Marshak vacuum) and a single energy group.
The diffusion equation collapses to

.. math::
   :label: bare-slab-diffusion-equation

   -D\,\phi''(x) + \Sigma_r\,\phi(x)
      = \frac{1}{k}\,\nu\Sigma_f\,\phi(x)

with :math:`\phi(0) = \phi(L) = 0`. Separation of variables
gives the eigenfunction

.. math::
   :label: bare-slab-eigenfunction

   \phi(x) \;=\; \sin\!\left(\frac{\pi x}{L}\right)


.. implements:: bare-slab-eigenfunction
   :by: orpheus.derivations.continuous.cases.diffusion._bare_slab_spectrum

   **Implemented by** 3 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: bare-slab-eigenfunction
   :by: orpheus.derivations.continuous.cases.diffusion.derive_1rg

.. implements:: bare-slab-eigenfunction
   :by: orpheus.derivations.continuous.cases.diffusion.derive_1rg_continuous

and the **geometric buckling**

.. math::
   :label: bare-slab-buckling

   B^{2} \;=\; \left(\frac{\pi}{L}\right)^{2}.


.. implements:: bare-slab-buckling
   :by: orpheus.derivations.continuous.cases.diffusion._bare_slab_spectrum

   **Implemented by** 3 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: bare-slab-buckling
   :by: orpheus.derivations.continuous.cases.diffusion.derive_1rg

.. implements:: bare-slab-buckling
   :by: orpheus.derivations.continuous.cases.diffusion.derive_1rg_continuous

Substituting :eq:`bare-slab-eigenfunction` into the diffusion
equation yields the eigenvalue condition

.. math::
   :label: bare-slab-critical-equation

   D\,B^{2} + \Sigma_r \;=\; \frac{1}{k}\,\nu\Sigma_f,


.. implements:: bare-slab-critical-equation
   :by: orpheus.derivations.continuous.cases.diffusion._bare_slab_spectrum

   **Implemented by** 4 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: bare-slab-critical-equation
   :by: orpheus.derivations.continuous.cases.diffusion._diffusion_coeffs

.. implements:: bare-slab-critical-equation
   :by: orpheus.derivations.continuous.cases.diffusion.derive_1rg

.. implements:: bare-slab-critical-equation
   :by: orpheus.derivations.continuous.cases.diffusion.derive_1rg_continuous

which solves to

.. math::
   :label: bare-slab-keff

   k \;=\; \frac{\nu\Sigma_f}{D\,B^{2} + \Sigma_r}.


.. implements:: bare-slab-keff
   :by: orpheus.derivations.continuous.cases.diffusion._bare_slab_spectrum

   **Implemented by** 3 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: bare-slab-keff
   :by: orpheus.derivations.continuous.cases.diffusion.derive_1rg

.. implements:: bare-slab-keff
   :by: orpheus.derivations.continuous.cases.diffusion.derive_1rg_continuous

Because the eigenfunction is independent of group in the
multigroup generalisation (all groups share the same spatial
:math:`\sin(\pi x/L)` shape), multigroup reduces to a
:math:`ng \times ng` matrix eigenvalue problem in the spectrum
vector — exactly what
:func:`orpheus.derivations.common.eigenvalue.kinf_and_spectrum_homogeneous`
solves, plus an extra ``D B²`` removal term on the diagonal of
:math:`\mathbf{A}`.

This is a **T1 analytical reference**: no integration, no
quadrature, no iteration. See
:func:`orpheus.derivations.continuous.cases.diffusion.derive_1rg_continuous` for the
Phase-0 :class:`~orpheus.derivations.ContinuousReferenceSolution`
that carries :math:`k_{\text{eff}}` and the continuous
multigroup eigenfunction callable.


.. _diffusion-2rg-fuel-reflector:

2-group fuel + reflector slab
=============================

A more demanding verification problem: fuel surrounded by a
reflector, both treated with 2-group diffusion, with **zero-flux**
boundaries (:math:`\phi = 0`, ``BC("zero_flux")``) on the outer faces.
The eigenfunction is no longer
a single sine — it is a linear combination of region-local
exponential and/or trigonometric modes, matched across the
fuel/reflector interface.

The rest of this section describes the implementation that
actually works. :ref:`diffusion-2rg-investigation-history` at
the end of the section records the two earlier approaches that
were tried and abandoned, with the numerical evidence of their
failure modes, so no future session reinvents them.

Region ODE and spatial modes
----------------------------

In each homogeneous region the multigroup diffusion equation
:eq:`diffusion-operator` reduces to

.. math::
   :label: diffusion-region-ode

   -\mathbf D\,\boldsymbol\phi''(x) + \mathbf M(k)\,\boldsymbol\phi(x)
     \;=\; \mathbf 0,

where the net removal matrix

.. math::
   :label: diffusion-M-matrix

   \mathbf M(k) \;=\; \text{diag}(\Sigma_{a,g} + \Sigma_{s,g,\text{out}})
                    \;-\; (\text{downscatter coupling})
                    \;-\; \frac{1}{k}\,\chi \otimes (\nu\Sigma_f)

absorbs the in-scatter source, the fission source, and the
removal losses into a single :math:`ng \times ng` operator.

Look for solutions of the form
:math:`\boldsymbol\phi(x) = e^{\lambda(x-x_0)}\,\mathbf u`:
substituting into :eq:`diffusion-region-ode` gives the
generalised eigenvalue problem

.. math::
   :label: diffusion-mode-decomposition

   \mathbf D^{-1}\mathbf M(k)\,\mathbf u_i \;=\; \mu_i\,\mathbf u_i,
   \qquad \mu_i \;=\; \lambda_i^{2}.

For a 2-group problem, :math:`\mathbf D^{-1}\mathbf M` is
:math:`2 \times 2`, so there are two eigenvalues
:math:`\mu_1, \mu_2` and two eigenvectors
:math:`\mathbf u_1, \mathbf u_2`. Each eigenvalue gives a
**pair** of spatial modes — the sign of :math:`\mu_i`
determines whether they are exponentials or trigonometrics:

.. math::
   :label: diffusion-exponential-branch

   \mu_i > 0 \;\Rightarrow\;
     \phi(x) \;=\; c_{i}^{+}\,e^{-\sqrt{\mu_i}\,(L_{\text{reg}} - (x - x_0))}\,\mathbf u_i
     \;+\; c_{i}^{-}\,e^{-\sqrt{\mu_i}\,(x - x_0)}\,\mathbf u_i

(subcritical region — pure decay modes, anchored to opposite
edges so both exponentials are bounded by :math:`1` within the
region), and

.. math::
   :label: diffusion-trigonometric-branch

   \mu_i < 0 \;\Rightarrow\;
     \phi(x) \;=\; c_{i}^{c}\,\cos\!\bigl(\sqrt{-\mu_i}(x - x_{\text{mid}})\bigr)\,\mathbf u_i
     \;+\; c_{i}^{s}\,\sin\!\bigl(\sqrt{-\mu_i}(x - x_{\text{mid}})\bigr)\,\mathbf u_i

(supercritical region — bounded oscillations centred at the
region midpoint :math:`x_{\text{mid}} = x_0 + L_{\text{reg}}/2`).

**Why this sign-branched basis matters.** The fuel region at
the fundamental mode :math:`k \approx 0.87` has one
:math:`\mu_i > 0` and one :math:`\mu_i < 0` — i.e. the fast
thermal couple includes *both* an exponential and a
trigonometric mode. The reflector at the same :math:`k` has
two exponential modes (both subcritical, as expected).
**Every basis mode in every region is bounded by 1** on its
domain. This bound is load-bearing: it is what makes the
assembled matching matrix :math:`\mathbf C(k)` below have
entries of :math:`\mathcal{O}(1)` and hence a determinant that
does not suffer catastrophic cancellation. See
:ref:`diffusion-2rg-investigation-history` for the earlier
approach that got this wrong.

Interface matching and zero-flux boundary conditions
-----------------------------------------------------

With the mode basis above, the solution in each region is a
linear combination of 4 basis functions (2 eigenvalues × 2
modes per eigenvalue). For the fuel + reflector slab we have
8 unknown mode coefficients total — 4 in fuel, 4 in reflector.

The 8 constraints that close the system (the outer faces carry the
zero-flux Dirichlet law :math:`\phi = 0`, ruling 3):

.. math::

   \boldsymbol\phi_{\text{fuel}}(0) \;=\; \mathbf 0
   \quad (\text{zero-flux left, 2 equations}),

.. math::
   :label: diffusion-interface-matching

   \boldsymbol\phi_{\text{fuel}}(H_f) \;=\;
     \boldsymbol\phi_{\text{refl}}(H_f),
   \qquad
   \mathbf J_{\text{fuel}}(H_f) \;=\; \mathbf J_{\text{refl}}(H_f)
   \quad (\text{interface, 4 equations}),

.. math::

   \boldsymbol\phi_{\text{refl}}(H_f + H_r) \;=\; \mathbf 0
   \quad (\text{zero-flux right, 2 equations}),

where the group current
:math:`\mathbf J_g(x) = -D_g\,\phi_g'(x)` is derived from the
mode basis analytically (the derivative of each basis mode has
a known closed form).

Collecting the 8 constraints into a matrix equation on the
8 mode coefficients
:math:`\mathbf c = [\mathbf c_{\text{fuel}};\;
\mathbf c_{\text{refl}}]` gives

.. math::
   :label: diffusion-matching-matrix

   \mathbf C(k)\,\mathbf c \;=\; \mathbf 0,
   \qquad \mathbf C(k) \in \mathbb R^{8 \times 8}.

Because every basis mode is bounded by 1,
:math:`\mathbf C(k)` has :math:`\mathcal{O}(1)` entries
(typical condition number :math:`\sim 30` at non-root values
of :math:`k`). A non-trivial mode coefficient vector exists
iff :math:`\mathbf C(k)` is singular, and the transcendental
eigenvalue condition is

.. math::
   :label: diffusion-transcendental

   \det\!\bigl(\mathbf C(k)\bigr) \;=\; 0.

This is bracketed by a coarse scan over :math:`k` and refined
via :func:`scipy.optimize.brentq` to xtol :math:`= 10^{-14}`.

Physical validation of candidate roots
--------------------------------------

The bracketing-and-refine pipeline above finds **more** sign
changes in :math:`\det(\mathbf C(k))` than there are actual
eigenvalues. These extra sign changes are an artefact of how
:func:`numpy.linalg.eig` orders the eigenvalues of
:math:`\mathbf D^{-1}\mathbf M(k)`: the order is not
continuous in :math:`k` across critical values where two
eigenvalues cross. When the order permutes, the columns of
:math:`\mathbf C(k)` permute discontinuously, and
:math:`\det(\mathbf C)` flips sign **by permutation** rather
than by passing through zero in a physically meaningful way.
``brentq`` then "converges" to a :math:`k` where
:math:`\mathbf C` is numerically singular by accident of the
eigenvalue labelling, not because the boundary-value problem
has a genuine solution there.

Each candidate root is therefore **physically validated**:

.. math::
   :label: diffusion-spurious-root-validation

   \boldsymbol\phi_{\text{fuel}}(0) \;\overset{?}{\approx}\; \mathbf 0,
   \quad
   \boldsymbol\phi_{\text{fuel}}(H_f) - \boldsymbol\phi_{\text{refl}}(H_f)
     \;\overset{?}{\approx}\; \mathbf 0,
   \quad
   \boldsymbol\phi_{\text{refl}}(H_f + H_r) \;\overset{?}{\approx}\; \mathbf 0.

The null vector of :math:`\mathbf C(k_{\text{cand}})` is
extracted via SVD, substituted back into the mode basis, and
the three residuals above are evaluated. A candidate passes
validation only when all three are below :math:`10^{-7}`
relative to the peak flux; otherwise it is rejected as a
spurious sign change. On the default 50 + 30 cm geometry the
scan between :math:`k \in [0.1, 3.0]` returns **six**
candidates, of which **three** are physical eigenvalues (0.370,
0.590, 0.870 — the fundamental plus two harmonics) and the
other three are eigenvalue-crossing artefacts. The fundamental
mode is the largest validated root.

Back-substitution for continuous :math:`\phi(x)`
-------------------------------------------------

Once the fundamental :math:`k_{\text{fund}}` is known and
validated, the null vector of
:math:`\mathbf C(k_{\text{fund}})` — extracted one last time
by SVD — gives the 8 mode coefficients
:math:`\mathbf c = [\mathbf c_{\text{fuel}};\;
\mathbf c_{\text{refl}}]`. The continuous flux at any
:math:`x` in the slab is then evaluated **pointwise** from
the region-local mode basis:

.. math::
   :label: diffusion-back-substitution

   \boldsymbol\phi(x) \;=\;
   \begin{cases}
     \displaystyle\sum_{j=1}^{4} c_{\text{fuel},j}\,m_j^{\text{fuel}}(x)\,\mathbf u_j^{\text{fuel}}
       & 0 \le x \le H_f, \\[0.3em]
     \displaystyle\sum_{j=1}^{4} c_{\text{refl},j}\,m_j^{\text{refl}}(x - H_f)\,\mathbf u_j^{\text{refl}}
       & H_f \le x \le H_f + H_r,
   \end{cases}

where :math:`m_j` is the :math:`j`-th basis mode
(:eq:`diffusion-exponential-branch` or
:eq:`diffusion-trigonometric-branch`) of the enclosing
region. **No matrix exponentials, no composition, no
condition-number explosion.** Every evaluation is a handful
of real multiplications plus at most two calls to
:func:`numpy.exp` / :func:`numpy.cos` / :func:`numpy.sin`.

The back-substituted :math:`\boldsymbol\phi(x)` is
**mesh-independent**: the test chooses its own cell centres,
calls
:meth:`~orpheus.derivations.ContinuousReferenceSolution.phi_on_mesh`,
and compares the diffusion solver's output to the continuous
reference at exactly those points. See
:func:`orpheus.derivations.continuous.cases.diffusion.derive_2rg_continuous`.

This is a **T2 semi-analytical reference**: the eigenvalue
:math:`k` is found to ``xtol=1e-14`` via brentq on a
well-conditioned determinant, the null vector is SVD-accurate
to machine precision on a matrix with condition number
:math:`\sim 10^{15}` *only at the eigenvalue itself*, and
:math:`\phi(x)` evaluation is pure O(1) algebra.

Numerical evidence at the default 50 + 30 cm geometry
-----------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Quantity
     - Expected
     - Observed
   * - :math:`k_{\text{eff}}` (transcendental)
     - —
     - :math:`0.8705068089`
   * - :math:`k_{\text{eff}}` vs legacy Richardson cache
     - :math:`\lesssim 10^{-5}` (Richardson :math:`\mathcal O(h^{4})`)
     - :math:`7 \times 10^{-8}`
   * - :math:`\phi_g(0)` residual (both groups)
     - machine :math:`\epsilon`
     - :math:`\sim 10^{-16}`
   * - :math:`\phi_g(L)` residual (both groups)
     - machine :math:`\epsilon`
     - :math:`\sim 10^{-16}`
   * - Interface continuity of :math:`\phi_g`
     - machine :math:`\epsilon`
     - :math:`\sim 10^{-11}`
   * - :math:`\mathbf C(k_{\text{fund}})` condition number
     - :math:`\sim 10^{15}` (singular)
     - :math:`\sim 10^{15}`
   * - :math:`\mathbf C(k_{\text{off-root}})` condition number
     - :math:`\mathcal{O}(1\text{–}100)`
     - :math:`\sim 30`

The finite-difference diffusion solver is then verified
against this reference by running a mesh refinement study;
see :ref:`diffusion-2rg-verification` in the verification part.


.. _diffusion-2rg-investigation-history:

Investigation history — abandoned approaches
--------------------------------------------

This section exists **on purpose** (Cardinal Rule 3: Sphinx is the
LLM's brain). It records two distinct pieces of machinery. **Dead ends
#1–#3 are the continuous reference solver** — the transcendental /
mode-basis eigenvalue reference in
:func:`~orpheus.derivations.continuous.cases.diffusion.derive_2rg_continuous`,
which is **live**. These failure modes are exactly why that reference
is built the way it is; any future session extending it (e.g. adding
Robin faces for the analytic Marshak reference, #293) must not repeat
them. **Dead end #4 is the retired MATLAB-port island** finite-\
difference solver: it is preserved because it teaches a real lesson,
but the specific bug is **moot** for the modern exact-LU path (which
has no inner iteration to mis-tolerance) — flagged in place below.

**Dead end #1 — First-order ODE state-vector composition
with** :func:`scipy.linalg.expm`.

The textbook approach
(Duderstadt & Hamilton 1976 §7-6; Stammler-Abbate §4.2) is
to carry the state vector
:math:`\mathbf y = [\boldsymbol\phi;\,\mathbf J]` and
propagate it through each region by the matrix exponential
:math:`\mathbf T(t) = \exp(\mathbf S\,t)` where

.. math::
   :label: diffusion-expm-state-matrix

   \mathbf S(k) \;=\;
   \begin{pmatrix} \mathbf 0 & -\mathbf D^{-1} \\
                    -\mathbf M(k) & \mathbf 0 \end{pmatrix}.

.. (vv-status rationale) documents a RETIRED dead-end (#1): the first-order
.. ODE state-vector / ``expm`` composition approach was implemented, found
.. ill-conditioned (5e-4 residual vs the machine-zero the zero-flux BC
.. demands), and abandoned. No production code implements this state matrix,
.. so there is no test and none should exist; the surviving reference is the
.. transcendental / mode-basis solver ``derive_2rg_continuous``.
.. vv-status: diffusion-expm-state-matrix documented

Continuity of :math:`\phi` and :math:`J` at the fuel/reflector
interface is then automatic because the state vector is the
continuous quantity. Zero-flux BCs
:math:`\phi(0) = \phi(L) = 0` pick out the upper-right
:math:`ng \times ng` block of the composed transfer matrix,
and the eigenvalue condition is
:math:`\det\bigl(\mathbf T_{\text{total}}(k)_{[0{:}ng,\,ng{:}2ng]}\bigr) = 0`.

This was implemented, the determinant brentq converged, and
the resulting "null vector" gave
:math:`|\phi_g(L)| \approx 5 \times 10^{-4}` relative to peak
flux — far from the machine-precision zero the zero-flux BC
demands. Investigation of the intermediate quantities
revealed:

- Condition number of the composed transfer matrix:
  :math:`\sim 10^{17}` over the 80 cm slab.
- Norm of the upper-right block: :math:`\sim 10^{15}`.
- Entries of the block: magnitudes up to :math:`\sim 10^{15}`.
- Pairwise products forming the determinant: :math:`\sim 10^{26}`.
- The returned determinant: :math:`\sim 10^{9}`.

That is a 17-decade cancellation of two 26-decade numbers,
leaving a "determinant" whose last 17 digits are numerical
noise. brentq dutifully finds a :math:`k` where this noisy
residual changes sign, but the resulting matrix is *not*
actually singular at that :math:`k` — its smallest singular
value is :math:`\sim 10^{-3}`, not the :math:`\sim 10^{-16}`
it would be at a genuine eigenvalue.

The root cause is that
:math:`\mathbf T_{\text{total}}(k) = \exp(\mathbf S\,t)` for
:math:`t = 80` cm contains exponentials
:math:`e^{\pm\lambda t}` with
:math:`\lambda \sim 0.5 \text{ cm}^{-1}` (the fast diffusion
length reciprocal), so
:math:`e^{\lambda t} \sim e^{40} \sim 10^{17}`. The growing
mode dominates the matrix; the physically meaningful decaying
mode lives in the last 17 bits of double precision. **No
condition-number improvement to the block, no tighter brentq
tolerance, no alternative root-finding criterion rescues
this formulation.** The error is baked into :math:`\mathbf T`
before any downstream step sees it.

**Dead end #2 — Complex eigenvalues and null-vector phase
ambiguity.**

The fix for dead end #1 is to diagonalise
:math:`\mathbf S` analytically per region and propagate in
the eigenbasis, so the growing and decaying modes are
separable and each mode is a bounded exponential. The first
attempt at this used complex eigenvalues throughout (because
:math:`\mathbf D^{-1}\mathbf M` can have negative eigenvalues
for supercritical regions, and "one size fits all" complex
arithmetic seemed simpler than sign-branching).

This nearly worked. The eigenvalue condition converged,
:math:`\phi(0)` and :math:`\phi(L)` were both machine precision
at each candidate root — but the SVD null vector was complex,
and the reconstructed :math:`\phi(x)` at interior points had
imaginary components comparable to its real part. The
imaginary components were artefacts of the absolute phase of
the null vector (which is arbitrary for SVD on a complex
matrix), not genuine complex physics. A phase-fix that made
one component of :math:`\mathbf c` real-positive did not
eliminate the imaginary parts everywhere, because different
:math:`k`-branches of :math:`\sqrt{\mu_i}` (the square roots
of complex :math:`\mu`) give different complex-conjugate
pairs, and the null vector cannot lie simultaneously on all
real-conjugate pairs without further projection.

The fix — used in the implementation above — is to treat the
two sign branches of :math:`\mu_i` separately from the start:
**real** exponentials when :math:`\mu_i > 0` and **real**
:math:`\cos\,/\,\sin` pairs when :math:`\mu_i < 0`. The
matching matrix :math:`\mathbf C(k)` is then real by
construction, the null vector is real, and no phase
corrections are needed anywhere.

**Dead end #3 — Spurious sign changes from eigenvalue
reordering.**

After the real-basis rewrite the null vector gave
:math:`\phi(0), \phi(L) \sim 10^{-16}` — machine precision,
as intended. But the first full smoke test returned
:math:`k = 1.0275` as the fundamental mode, contradicting
both the Richardson cache (:math:`\sim 0.87051`) and the
earlier complex-basis prototype. Diagnostic output showed six
sign-change candidates in :math:`\det(\mathbf C(k))`:

.. code-block:: text

   k=0.1466  s[-1]/s[0]=1.19e-16  ← validated by SVD
   k=0.2302  s[-1]/s[0]=9.13e-17  ← validated by SVD
   k=0.3700  s[-1]/s[0]=1.46e-17  ← validated by SVD
   k=0.5901  s[-1]/s[0]=1.79e-17  ← validated by SVD
   k=0.8705  s[-1]/s[0]=1.25e-16  ← validated by SVD
   k=1.0275  s[-1]/s[0]=5.75e-18  ← validated by SVD

All six had smallest singular value at machine precision.
Naïvely taking the maximum gave :math:`k = 1.0275`. But
when the null vector was substituted back into the mode
basis, :math:`\phi(0)`, interface continuity, and
:math:`\phi(L)` *all* held at :math:`\sim 10^{-7}` for the
three "extra" candidates — nothing like the :math:`\sim 10^{-16}`
seen at the three genuine roots. The extras were spurious.

Root cause: :func:`numpy.linalg.eig` returns eigenvalues in
an order set by the underlying LAPACK routine
(``DGEEV``/``ZGEEV``), and the order is **not a continuous
function of** :math:`k`. When two eigenvalues cross, their
labels can swap — and because the columns of
:math:`\mathbf C(k)` are built from the labelled eigenvectors,
the column ordering permutes discontinuously at the crossing
point. Permuting columns of a square matrix flips the sign
of its determinant, so :math:`\det(\mathbf C(k))` flips
sign at every crossing — and brentq happily "finds a root"
there even though the matrix is continuously non-singular on
both sides of the crossing.

The smallest-singular-value check does not catch these:
**at** the crossing, the swapped-column matrix is
instantaneously rank-deficient in the label-permutation
sense, so :math:`s_{\min}/s_{\max}` does drop to machine
precision. But when the null vector is substituted back
into the mode basis, it does not solve the boundary-value
problem, because the labels do not correspond to a physical
mode decomposition at the crossing.

The only reliable discriminator is the **physical
validation** in :eq:`diffusion-spurious-root-validation`:
reconstruct :math:`\phi(0)`, the interface continuity, and
:math:`\phi(L)` from the null vector and check whether they
actually vanish. Genuine eigenvalues pass to machine
precision; crossings fail to :math:`\sim 10^{-7}`. This is
the load-bearing filter in
:func:`~orpheus.derivations.continuous.cases.diffusion._solve_2region_zero_flux_eigenvalue`
(renamed from ``_solve_2region_vacuum`` at #290 P6, when the
:math:`\phi = 0` references were re-attributed to the zero-flux law).

**Dead end #4 (retired island, solver-side) — Hardcoded outer-iteration
tolerance masked quadratic convergence.**

.. note::

   This dead end belonged to the **retired MATLAB-port island**'s
   finite-difference solver, not to the continuous reference. It is
   preserved for its lesson, but the modern exact-LU solver (#290 P5)
   **cannot** exhibit it: there is no inner iteration to mis-tolerance
   and no hardcoded outer floor. The ``DiffusionSolver`` /
   ``solve_diffusion_1d`` names in this dead end refer to the island
   implementation, now deleted; the paragraphs are past-tense history.

Once the reference was correct, the island finite-difference solver
``solve_diffusion_1d`` was run at four mesh refinements to measure
convergence order. The expected :math:`\mathcal{O}(h^{2})` order of
central finite differences produced this embarrassing error sequence on
the bare slab:

.. code-block:: text

   dz=5.0    shape_err = 1.21e-04    order = -
   dz=2.5    shape_err = 3.19e-05    order = 1.93  ← quadratic
   dz=1.25   shape_err = 1.31e-05    order = 1.28  ← pre-plateau
   dz=0.625  shape_err = 1.01e-05    order = 0.37  ← plateau

The finest-mesh error plateaus at :math:`\sim 10^{-5}`, not
the :math:`\sim 10^{-6}` that would extend the quadratic
trend. Initial misdiagnosis: BiCGSTAB inner solver tolerance
(``errtol=1e-6``). Raising ``errtol=1e-12`` did not move the
plateau. Correct diagnosis: the outer power iteration in the island's
``DiffusionSolver`` had a **hardcoded** convergence criterion
``rel_change < 1e-5`` on the flux relative change between outer
iterations. That threshold was the floor — the outer solve stopped as
soon as the flux was within :math:`10^{-5}` of its own previous
iterate, which is exactly where the convergence tests were plateauing.
The finite-difference discretisation error was *below* the
outer-iteration noise at the finest meshes.

At the time the fix was a two-line change to the island's
``DiffusionSolver``: add an ``outer_tol`` keyword and replace the
hardcoded ``< 1e-5``. **The modern solver retired the question
entirely.** :class:`~orpheus.diffusion.solver.DiffusionSolver` (#290
P5) has no BiCGSTAB inner solver and no hardcoded outer floor: the
inner solve is the *exact* resolvent :math:`A^{-1}` (one LU
back-substitution, converged by construction), and the outer power
iteration converges on ``keff_tol`` (default :math:`10^{-10}`) and the
relative flux change. There is no inner-iteration noise to sit below
the discretisation error, so the :math:`\mathcal{O}(h^2)` order is
recovered at the default tolerances with no knob to tune.

⭐ **And the failure this bug taught is now self-reporting** (#340 N4/N4.7,
2026-08-11). :func:`~orpheus.diffusion.solver.solve_diffusion_1d` returns an
:class:`~orpheus.numerics.convergence.IterationRecord` and calls
:func:`~orpheus.numerics.convergence.warn_if_unconverged` before it returns,
so an outer that exhausts ``max_outer`` announces itself once, naming
``max_outer`` and the count its observed rate projects. **Diffusion's tree can
only fail at the outer**, and that is a structural statement rather than an
untested one: the inner is the exact LU resolvent, recorded with ``budget =
0`` — a ``DIRECT`` level, which by construction can never be TRUNCATED. So
the warning here is always about the power iteration, and ``max_outer`` is
always the knob it names.

⚠ Which makes diffusion the awkward family to *starve on purpose*, and the
awkwardness is itself the finding. `[M]` 2026-08-11: with
``max_outer = 3`` the solve **converges** —
:data:`~orpheus.numerics.eigenvalue.MINIMUM_OUTER_ITERATIONS` is also 3 — so
no budget can starve it, and ``keff_tol = 1e-15`` does not
either, because an exact resolvent drives :math:`|\Delta k|` to
:math:`\sim10^{-16}` immediately. The only reliable way to hold this level
open is an unsatisfiable tolerance (``keff_tol = 0.0``, which
:class:`~orpheus.numerics.convergence.StoppingCriterion` documents as its
never-clears input). A method whose failure mode is *hard to provoke* is a
good method and a badly-covered one; the fixture is spelled out in
``tests/numerics/test_family_convergence_contract.py`` so the next reader
does not rediscover all three dead ends.

The general lesson **transfers** to any iterative solver even though
this specific bug is now moot: when a convergence-order verification
test plateaus, check both the inner and outer solver tolerances before
blaming the reference solution. If the measured order is pathological
for a well-understood discretisation, the solver's own convergence
machinery is the first suspect, not the reference.


.. _diffusion-verification-pins:

Verification — what pins this chapter
=====================================

The verification evidence for the diffusion solver — the bare-slab
buckling anchor, the 2-region continuous-reference cases with their
measured convergence orders, and the retired-Richardson record —
lives in the verification part: :doc:`/theory/verification/diffusion`
(anchor :ref:`diffusion-2rg-verification`).  The MMS operator gate
below stays with this chapter's derivation.  The auto-generated
:doc:`/theory/verification/matrix` reports per-equation test
coverage; :ref:`theory-verification` carries the part-wide
principles and harness contracts.


.. _diffusion-mms-section:

Method of Manufactured Solutions (fixed-source operator gate)
=============================================================

The eigenvalue anchors above run **piecewise-constant** cross
sections: the conductance interpolation is exercised only at the
single fuel/reflector interface, and the group coupling only through
the eigen-spectrum. Issue #93 originally proposed an MMS gate with a
*single-group sine and constant* :math:`D` — an ansatz that **nulls**
(vv-principles Mode 7) exactly the two hardest terms of the operator:
with :math:`D' \equiv 0` the :math:`(D\phi')'` product-rule content
and the per-face conductance interpolation never differ from the
constant-:math:`D` stencil, and with one group there is no scatter
coupling at all. A green gate on that ansatz would be blind to the
face-interpolation and scatter-transpose bug classes (AI failure
modes 2/3/6). The landed gate (#290 P6) therefore overrides the
proposal with a **heterogeneous-D, multigroup-coupled** manufactured
problem on :math:`x \in [0, L]`, :math:`L = 10` cm:

.. math::
   :label: diffusion-mms

   \begin{aligned}
   D_1(x) &= 1.2\,\bigl(1 + 0.35\sin(\pi x/L)\bigr), &
   \phi_1(x) &= \sin(\pi x/L),\\
   D_2(x) &= 0.45\,\bigl(1 - 0.25\cos(2\pi x/L)\bigr), &
   \phi_2(x) &= \sin(2\pi x/L) + 0.6\sin(\pi x/L),\\
   q_1 &= -(D_1\phi_1')' + (\Sigma_{a,1} + \Sigma_{1\to2})\,\phi_1, &
   q_2 &= -(D_2\phi_2')' + \Sigma_{a,2}\,\phi_2 - \Sigma_{1\to2}\,\phi_1,
   \end{aligned}

with :math:`\Sigma_a = (0.010, 0.080)`,
:math:`\Sigma_{1\to2} = 0.015`, zero fission, and the zero-flux law
on both faces (:math:`\phi_g` vanishes there by construction). The
in-group scatter cancels against the collision term by the column-sum
theorem, so the forcing carries removal = absorption + out-scatter —
the same cancellation the assembled :math:`A = L + C - S - B`
realizes (this fixture is :math:`\Sigma_{2n} \equiv 0`, so the summed
:math:`S` carries the scattering leaf alone). The forcing is SymPy-differentiated from the same symbolic
:math:`(D_g, \phi_g)` (structurally independent of the
finite-difference assembly under test), sampled at cell centres, and
pushed through the solver's exact resolvent as a fixed-source solve
:math:`\psi_h = A^{-1} q`; the gate asserts
:math:`\lVert\psi_h - \phi\rVert_{\ell^2(h)} = \mathcal O(h^2)`.

**Discrete posing.** Every cell is its own material with
:math:`\sigma_{t,g}(x_i) = 1/(3 D_g(x_i))` (the P1 data seam
inverted), so EVERY interior face is a material interface for the
current-continuous conductance
:math:`g_f = (h_L/2D_L + h_R/2D_R)^{-1}` — the term the constant-D
sine can never probe.

**Ansatz activation declaration (Mode 7).** Activated: the
D-gradient / face-interpolation content (:math:`D_g' \neq 0` across
the whole domain, distinct per group), the down-scatter in-scatter
row (:math:`\phi_2 \not\propto \phi_1` — an :math:`\mathcal O(1)`
mismatch against any transpose or sign confusion), and removal.
Nulled, each with its covering gate elsewhere: fission (the L1
eigenvalue anchors + the L2 infinite-medium gate), the
non-zero-flux boundary laws (the per-law trace-semantics gates), and
curvilinear area/volume factors (slab — pinned by the P4 hand-posed
stencil gate).

**Constrained, not merely activated (Mode 10).** The suite COMMITS
two controls proving the error functional responds
:math:`\mathcal O(1)` to a corruption of each activated term (at
:math:`n = 40`, clean error :math:`3.4 \times 10^{-3}`): flattening
:math:`D` to its midslab value while the forcing keeps the
heterogeneous one gives :math:`4.9 \times 10^{-1}` (×144), and
flipping the in-scatter forcing sign gives
:math:`5.7 \times 10^{-1}` (×166). A production-side mutation probe
(one-sided left-cell conductance in place of the harmonic mean)
collapses the measured order from :math:`(2.004, 2.001, 2.000)` to
:math:`(1.121, 1.034, 1.009)` with the finest-mesh error
:math:`3.4 \times 10^{-3}` — RED under both gate assertions.

Measured evidence (``tests/diffusion/test_mms.py``, cells
:math:`n = 20, 40, 80, 160`): errors
:math:`1.37 \times 10^{-2}`, :math:`3.41 \times 10^{-3}`,
:math:`8.53 \times 10^{-4}`, :math:`2.13 \times 10^{-4}`;
orders :math:`2.004, 2.001, 2.000`.


.. _diffusion-development-history:

Development history
===================

Reverse-chronological changelog of the **#290 diffusion-integration
campaign** (branch ``feature/diffusion-integration``, 2026-07-03),
which replaced the MATLAB-port island with the operator-algebra solver
documented above. Each phase is a gated commit; the branch is unmerged
at the time of writing (P8, this documentation pass). See the campaign
plan ``.claude/plans/archive/diffusion_integration_290.md`` and the GitHub
issues for finer granularity.

.. list-table::
   :header-rows: 1
   :widths: 9 67 24

   * - Phase
     - Architectural milestone (what + why)
     - Commit
   * - **P7b**
     - **The** :class:`~orpheus.transport.method.TransportMethod`
       **Protocol + shared BC-resolution body + registry dissolution**
       — minted the structural Protocol over the two method-meshes and
       collapsed the twin ``_resolve_bcs`` loops into one
       :func:`~orpheus.transport.method.resolve_boundary_conditions`
       body; deleted the string-keyed realizer registry (you hold the
       method-mesh, therefore you hold its realizer). Discharges the
       second ``TransportMethod`` witness.
     - ``44d583e``
   * - **P7a**
     - **The** :class:`~orpheus.diffusion.augmented_mesh.DiffusionMesh`
       **method-mesh** — reclaimed the scalar trace and composite
       carrier off ``MaterialMesh`` onto a diffusion method-mesh (the
       ``SNMesh`` sibling), realizing boundary laws at construction and
       restoring the data/behaviour axis (a ``MaterialMesh`` does not
       know what a trace is).
     - ``738e355``
   * - **P6**
     - **Island retirement + reference re-attribution + MMS (#93)** —
       deleted the MATLAB-port island (``CoreGeometry`` / ``TwoGroupXS``
       / BiCGSTAB); ``k_eigenvalue.py`` became ``solver.py``; the
       analytic :math:`\phi = 0` references were re-attributed to
       ``BC("zero_flux")`` at unchanged tolerances (ruling 3); the
       heterogeneous-\ :math:`D` multigroup MMS gate landed (closes #93).
     - ``9104233``
   * - **P5**
     - **The modern k-eigenvalue solver on the shared engines** —
       :class:`~orpheus.diffusion.solver.DiffusionSolver` on the
       ``EigenvalueSolver`` protocol over the flat composite; the
       exact-LU resolvent; ``power_iteration`` :math:`\equiv`
       ``direct_eigenvalue`` at :math:`10^{-10}`; #270 production
       normalisation; the 3-group discriminator that kills the flip
       trick.
     - ``9470266``
   * - **P4**
     - **The operator family** — the two new leaves
       :class:`~orpheus.diffusion.operators.LeakageOperator` (``L``) and
       :class:`~orpheus.diffusion.operators.DiffusionBoundaryOperator`
       (``B``) on the scalar composite; ``C`` / ``S`` / ``F`` gained
       scalar-composite arms on the shared kernels; the object-level
       stencil gate with four RED mutations.
     - ``db14643``
   * - **P3**
     - **Boundary laws + functional realizer (closes #182)** — the
       ``ZeroFluxBoundary`` law and the
       :class:`~orpheus.diffusion.boundary_realizer.DiffusionBoundaryRealizer`
       (law :math:`\to \mathcal{A} \to` operator), realizing every
       diffusion BC as an albedo-family scalar :math:`J^- = \mathcal{A}
       J^+`.
     - ``6672e7a``
   * - **P2.5**
     - **Trace naming coherence** — the angular / scalar trace family
       split (``AngularTraceSpace`` / ``ScalarTraceSpace``,
       ``ScalarBoundaryFlux``), so the composite's bulk :math:`\times`
       boundary vocabulary reads coherently across both methods.
     - ``1cd8d32``
   * - **P2**
     - **Scalar trace substrate** — the ``ScalarTraceSpace`` and the
       ``(J^+, J^-)`` partial-current trace leaf; ``FullField`` widened
       to admit the scalar family (the anticipated second consumer of
       scalar-flux composites, now arrived).
     - ``78d1431``
   * - **P1**
     - **The data seam** — ``Mixture.transport_xs`` +
       :attr:`Mixture.diffusion_coefficient
       <orpheus.data.macro_xs.mixture.Mixture.diffusion_coefficient>`,
       so :math:`D = 1/(3\Sigma_{\text{tr}})` reads through the
       canonical XS type; the legacy tables encode bit-identically
       (ruling 4).
     - ``836f424``
   * - pre-#290
     - **MATLAB-port island (retired)** — the original 443-line port:
       raw ``(2, n_cells)`` arrays, scipy BiCGSTAB, ``TwoGroupXS`` /
       ``CoreGeometry``, string BC keys, and a hardcoded 2-group
       ``sig_s[::-1]`` down-scatter flip. Superseded entirely by #290.
     - (deleted at P6)


References
==========

- Bell, G. I. and Glasstone, S., *Nuclear Reactor Theory*,
  Van Nostrand Reinhold, 1970. Chapter 7 covers multigroup
  diffusion theory; §7.4 specifically treats the slab
  eigenvalue problem and the two-region interface matching.
- Stacey, W. M., *Nuclear Reactor Physics*, 3rd ed., Wiley,
  2018. Ch. 3 on diffusion theory and Ch. 8 on multigroup
  formulation.
- Duderstadt, J. J. and Hamilton, L. J., *Nuclear Reactor
  Analysis*, Wiley, 1976. Ch. 5 (one-group) and Ch. 7
  (multigroup) — the transfer matrix formulation is
  spelled out explicitly. §5.2 gives the Marshak / extrapolation
  boundary algebra.
- Marshak, R. E., "Note on the spherical harmonic method as applied to
  the Milne problem for a sphere", *Phys. Rev.* **71**, 443 (1947) —
  the zero-incoming-partial-current (Marshak) :term:`vacuum boundary condition`
  :math:`J^- = 0`.
- Baliga, B. R. and Patankar, S. V., "A control-volume finite-element
  method for two-dimensional fluid flow and heat transfer",
  *Numer. Heat Transfer* **6**, 245 (1983) — the finite-difference /
  lowest-order Raviart–Thomas (mass-lumped) equivalence for the
  diffusion stencil.
