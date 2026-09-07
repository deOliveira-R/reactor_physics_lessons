.. _theory-foundations:

===========
Foundations
===========

The mathematics **every** method shares. A concept lives here once; each
method's chapter carries only *its realization* plus a link back.

This part exists because the shared content is genuinely shared **in code,
not by analogy**: :class:`~orpheus.transport.operators.MultiplicationOperator`
and the fission energy binding
:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`
are the *same Python classes* instantiated by S\ :sub:`N`, diffusion and the
infinite-medium solver, and all three draw their scattering from the same
:mod:`orpheus.transport.operators` package
(:class:`~orpheus.transport.operators.IsotropicScattering` /
:class:`~orpheus.transport.operators.IsotropicN2N` for the isotropic
consumers, the same package's
:class:`~orpheus.transport.operators.ScatteringOperator` kernel for
S\ :sub:`N`). Each reaction channel carries **two bindings of one datum** —
an energy binding on the scalar flux and, where a method resolves angle, the
harmonic frame's conjugation of that same binding — so the two faces of a
channel cannot drift. What varies between methods is how streaming is
represented — not what collision, scattering, and fission *are*.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - What it settles
   * - :doc:`/theory/foundations/path_integral`
     - **The root of the corpus** — the one object all methods discretize
       (the sum over neutron histories), where it comes from and why it is
       linear, what is invariant (:math:`C, S, F`), what varies, the three
       independent axes on which methods differ, where each lands, and the
       eigenvalue posing every method inherits. The parent of
       :doc:`/theory/methods/index`.
   * - :doc:`/theory/foundations/operator_algebra`
     - The operator algebra itself: :math:`A = L + C - S - N_{2n} - B`, posed
       :math:`A\psi = \tfrac{1}{k}F\psi` (eigenvalue) or :math:`A\psi = q`
       (fixed source). :math:`B` is a first-class **sibling**, not folded
       into :math:`L`; :math:`(L+C)` is the sub-composite whose inverse
       **is** the transport sweep.
   * - :doc:`/theory/foundations/operator_inverse_family`
     - The **inverse family** of that algebra: the four realizations of
       "apply :math:`A^{-1}` / materialize :math:`A`" the #226 taxonomy
       separates — the driver-applied sweep, the Green preconditioned
       inverse, the dense materialising inverse, and the sparse assembly
       axis.
   * - :doc:`/theory/foundations/operator_tensor_network`
     - The **tensor-network shape decomposition** of that algebra: which
       factored shape each S\ :sub:`N` operator leaf takes, the **MA-Q1**
       admissibility condition, and why streaming's in-sweep recurrence
       resists a clean tensor product — **five distinct shapes, not one**.
   * - :doc:`/theory/foundations/operator_adjoint`
     - The **composite metric adjoint** of that algebra: the Hilbert
       adjoint :math:`A^{\dagger} = G^{-1} A^{\mathsf T} G` (``op.H``) over
       the block-diagonal ``FullFieldSpace`` metric, its singular-trace
       Moore–Penrose pseudo-inverse — the **operator's** adjoint, distinct
       from the **frame's** Petrov–Galerkin test-space adjoint.
   * - :doc:`/theory/foundations/field_algebra`
     - The **field algebra** those operators act on: flux in the
       **positive cone** :math:`K` of an ordered vector space :math:`V`,
       cone membership as an element predicate and cone preservation as a
       realization property, the **flux / source / residual** role grid,
       and the iterate diagnostics (contraction ratio, true-error
       estimate, per-entry map) the iteration record carries. Includes the
       six-argument adjudication that overturned the 2026-06 affine
       ontology, with the retired design kept as dated history.
   * - :doc:`/theory/foundations/spaces`
     - The **space layer** those fields live in: a function space as the
       ordered product of its **axes** (index shape, factor measure,
       basis kind, and the **generator** that minted it — provenance
       deliberately excluded from the axis's structural identity), the
       **forgetful-map doctrine** that makes an axis able to hand back
       the nodes it dropped, the **counting-measure theorem** that makes
       the energy metric the identity, the **metric-as-object doctrine**
       (three sources with one resolution, the Moore–Penrose face on a
       matrix, and why an axis keeps a *measure* while a space may hold
       a *form*), and the **collapse doctrine** —
       the two one-line tests that decide which axes survive a
       degeneracy (the homogeneous quotient point persists; the angular
       axis of a scalar space does not), with both refuted earlier
       doctrines kept beside the questions that refuted them.
   * - :doc:`/theory/foundations/wavefront_cochain`
     - The **interior face-flux cochain** :math:`C^1_{\rm int}` — the
       sweep-internal cochain that carries flux across cell faces during a
       sweep, its biproduct :math:`C^1 = C^1_{\rm int} \oplus C^1_\partial`
       and trace algebra, and why the typed ``WavefrontFlux`` carrier
       retired (the concept survives in its two native realizations).
   * - :doc:`/theory/foundations/coupled_block_operator`
     - The **2×2 coupled block operator** — the curvilinear
       starting-direction flux :math:`\psi_{1/2}` (the ψ½ ray) as a
       first-class **System B**, its four named blocks (:math:`A_{AA}`
       the within-group loss composite, plus the seed / emission / march
       couplings), the N-general block machinery, and the structure-keyed
       block solve.
   * - :doc:`/theory/foundations/discretization`
     - How a continuous conservation law becomes a finite algebraic system:
       the **cell-balance** invariant (sinks = sources) and its **closures** —
       Step (upwind), Diamond Difference (central), Linear Discontinuous
       (linear upwind), derived once and dimension-agnostic (the same closure
       in space **and** angle).
   * - :doc:`/theory/foundations/frame`
     - Frames, and why projection is **Petrov-Galerkin**: the trial/test
       split, the adjoint, and the realizations (spherical-harmonics
       Galerkin; homogenization and energy condensation as Petrov-Galerkin).
   * - :doc:`/theory/foundations/boundary_conditions`
     - The boundary law :math:`B` as an operator: trace realization,
       reflective / vacuum / white, and the extraction criterion.
   * - :doc:`/theory/foundations/cross_section_data`
     - The cross-section pipeline: mixtures, multigroup data, condensation.
   * - :doc:`/theory/foundations/manifolds`
     - The **point-set layer** beneath
       :doc:`/theory/foundations/discrete_measures` and
       :doc:`/theory/foundations/spaces`: the manifold
       :math:`M` a measure is supported *on* and a basis function is
       defined *over*, its algebra (product, orbit space, membership),
       the invariant-theoretic **Procesi–Schwarz** derivation that
       produces an orbit space from a symmetry group, and the
       three-level separation (manifold / fields on it / coefficients)
       that keeps a ``FunctionSpace`` from being mistaken for a domain.
       ✅ The migration off ``Space = str`` **landed 2026-09-01**
       (tracker 2.0c): a measure's ``support`` IS a manifold, and its
       phase, its induced space's name and its orbit space are all
       derived from it.
   * - :doc:`/theory/foundations/discrete_measures`
     - Quadrature and measure: axes, weights, and integration.
   * - :doc:`/theory/foundations/spherical_harmonics`
     - The angular basis and the addition theorem.
   * - :doc:`/theory/foundations/structured_geometry`
     - Meshes and structured geometry.
   * - :doc:`/theory/foundations/infinite_medium`
     - The 0-D infinite-medium (:math:`k_\infty`) baseline — the analytical
       anchor every method must reproduce. **Not** spatial homogenization;
       for that see :doc:`/theory/foundations/frame`.

.. toctree::
   :maxdepth: 2

   path_integral
   operator_algebra
   operator_inverse_family
   operator_tensor_network
   operator_adjoint
   field_algebra
   spaces
   wavefront_cochain
   coupled_block_operator
   discretization
   frame
   boundary_conditions
   cross_section_data
   manifolds
   discrete_measures
   spherical_harmonics
   structured_geometry
   infinite_medium
