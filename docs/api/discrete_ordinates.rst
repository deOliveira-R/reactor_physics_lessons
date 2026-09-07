Discrete Ordinates Solvers
==========================

Reference for the :mod:`orpheus.sn` package — the discrete-ordinates
(S\ :sub:`N`) transport solvers. Two execution paths share the same
quadrature and geometry layer:

* **Source iteration** via diamond-difference sweeps
  (:mod:`~orpheus.sn.loss_representation` — the loss representations +
  the orchestration that historically lived in ``sweep.py``, dissolved
  at S6.4(f)) — the default path used by
  :class:`~orpheus.sn.solver.SNSolver`.
* **Krylov** via an explicit transport operator
  (:mod:`~orpheus.sn.operators.streaming`) — forms ``T: ψ → T·ψ`` as a
  :class:`scipy.sparse.linalg.LinearOperator` so scipy's BiCGSTAB / GMRES
  can drive the inner solve directly.

The theory pages cover the diamond-difference discretisation, the
angular-redistribution term for curvilinear geometry (the
:math:`\alpha` dome recursion — Lathrop & Carlson 1966, implemented
form Hébert 2009 §3.9.3/§3.9.4; the Morel--Montry closure weight is
Morel & Montry 1984, in the Bailey--Morel--Chang 2010 Eqs. (42)/(43)
form), and the source-iteration / Krylov trade-offs.  ⛔ This sentence
credited "Bailey et al. 2009" until 2026-08-27 — the wrong-paper
citation retracted at Issue #168 Phase B; see
:ref:`sn-citation-corrections`.

Solver
------

.. automodule:: orpheus.sn.solver
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Geometry
--------

.. automodule:: orpheus.sn.mesh.augmented_mesh
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Angular Quadrature
------------------

The angular quadrature now lives in the method-agnostic
:mod:`orpheus.numerics.quadrature` package (re-exported as
:class:`orpheus.sn.Quadrature` for the SN solver's convenience). The
five legacy SN-only subclasses (``AngularQuadrature``,
``GaussLegendre1D``, ``LebedevSphere``, ``LevelSymmetricSN``,
``ProductQuadrature``) collapsed into the single
:class:`~orpheus.numerics.quadrature.Quadrature` value type with
``classmethod`` factories:

* :meth:`~orpheus.numerics.quadrature.Quadrature.gauss_legendre`
  — 1-D Gauss-Legendre on :math:`\mu \in [-1, 1]` (slab / curvilinear
  radial).
* :meth:`~orpheus.numerics.quadrature.Quadrature.level_symmetric`
  — :math:`O_h`-invariant level-symmetric :math:`S_N` (Carlson &
  Lathrop 1968).
* :meth:`~orpheus.numerics.quadrature.Quadrature.lebedev`
  — :math:`O_h`-invariant Lebedev sphere quadrature.
* :meth:`~orpheus.numerics.quadrature.Quadrature.product`
  — Gauss-Legendre :math:`(\mu)` :math:`\times` equispaced
  :math:`(\phi)` product rule.

The per-ordinate angular data is exposed through the cached
:attr:`~orpheus.numerics.quadrature.Quadrature.octants` partition and
the :meth:`~orpheus.numerics.quadrature.Quadrature.angular_frame`
/ :meth:`~orpheus.numerics.quadrature.Quadrature.ordinate_permutation`
methods. The selection driver is
:func:`~orpheus.numerics.quadrature.select_quadrature`, backed by the
:data:`~orpheus.numerics.quadrature.quadrature_registry`. The full
mathematical narrative — the level-symmetric construction, the
selection criterion, and the product-rule cosine layout — lives at
:ref:`discrete-measures` and in the per-module docstrings, accessible
via the standard ``orpheus.numerics.quadrature`` import path. (The
package carries rich ``.. math:: :label:`` docstrings, so it is
cross-referenced here rather than ``automodule``-rendered, to avoid
duplicate-label collisions with the theory pages.)

Transport Sweep — the loss representations
------------------------------------------

.. automodule:: orpheus.sn.loss_representation
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Direct Transport Operator
-------------------------

The :mod:`~orpheus.sn.operators.streaming` module carries the streaming
leaves of the within-group algebra
:math:`A = L + C - S - N_{2n} - B` (:eq:`sn-within-group-with-n2n`):
:class:`~orpheus.sn.operators.streaming.StreamingOperator` is the pure
:math:`\sigma`-free :math:`L = \Omega\cdot\nabla` (plus the curvilinear
angular redistribution), and
:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` is
the sweep-invertible specialisation :math:`(L + C)` returned by
``L + C``. The collision multiplier :math:`C = M[\sigma_t]` is not
defined here — it is a plain
:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`.

Both consume and emit the typed composite carrier
:class:`~orpheus.transport.timed_full_field.TimedFullField` (bulk
:class:`~orpheus.transport.fields.angular_flux.AngularFlux` + boundary
:class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`).
There is **no packed-vector codec**: the ``EquationMap`` /
``build_equation_map`` / ``solution_to_angular_flux`` slot-map family
that once enumerated which ``(ordinate, cell)`` pairs are unknowns was
retired in 2026-05 once the typed contract landed at every operator
leaf. The equivalent information is now read straight off the
quadrature and the mesh (the ordinate sign mask), and the flat view
scipy's Krylov drivers need is produced by the carrier's inherited
:meth:`~orpheus.transport.full_field.Composite.to_flat` /
:meth:`~orpheus.transport.full_field.Composite.from_flat` pair.

.. automodule:: orpheus.sn.operators.streaming
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:


Realized Boundary Law Operator (``B``)
--------------------------------------

The :mod:`~orpheus.sn.operators.boundary` module assembles the realized
per-face boundary laws into the whole-trace operator
:class:`~orpheus.sn.operators.boundary.SNBoundaryOperator` — the
:math:`A_{ss}` boundary block of the canonical SN loss
:math:`(L_{\rm full} + C - S - F - B)`. It is the first-class sibling
:math:`-B` introduced by Wave O step O.4a.2 (Issue #208). See
:ref:`bc-extraction` for the block-matrix derivation and design
rationale.

.. automodule:: orpheus.sn.operators.boundary
   :members:
   :show-inheritance:


ψ½ Coupled-Block Operators (A_BB, A_AB)
---------------------------------------------------

The :mod:`~orpheus.sn.operators.radial_characteristic` module hosts the two
System-B blocks of the ψ½ coupled block operator (the augmented within-group
system re-partitioned as a 2×2 block operator over the transport bulk⊕trace and
the radial-characteristic ray). :class:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator`
(``A_BB``) is the radial straight-characteristic transport self-block (the
two-point radial BVP whose direct Carlson march IS the resolvent :math:`A_{BB}^{-1}`);
:class:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicSeeding`
(``A_AB``) is the cell-local angular ray→bulk seed injection (the Morel–Montry
ψ½ seed folded into the bulk angular recurrence). See the class docstrings for
the operator-algebra posing and the campaign plan
``coupled_block_operator_campaign.md`` for the full 2×2 assembly (step 4).

.. automodule:: orpheus.sn.operators.radial_characteristic
   :members:
   :show-inheritance:
   :noindex:


The Loss-Kernel Gauge (``Pi``)
-------------------------------

The :mod:`~orpheus.sn.operators.loss_kernel_gauge` module is not a leaf of
:math:`A` but a statement *about* :math:`A`: on an all-reflective Cartesian box
closed by diamond differencing, :math:`A = L + C - S - N_{2n} - B` is **exactly
singular**, so the boundary trace a solve returns is a function of the cold
start rather than of the problem (#344 — ``[M]`` up to 27.3 % apart, with both
convergence functionals blind to the difference).

:class:`~orpheus.sn.operators.loss_kernel_gauge.LossKernelGauge` is the
:math:`G`-orthogonal projector onto :math:`\ker A`, built in **closed form** —
no eigensolve and no SVD of :math:`A`. Its applicability is *derived*, never
tabulated: :func:`~orpheus.sn.operators.loss_kernel_gauge.gauge_freedom` asks
the spatial closure whether it leaves a face mode undamped
(:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.face_transmission_spectrum`)
and asks the mesh how many reflective axis pairs close
(:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.reflective_axis_pairs`), so a
discretization added tomorrow answers for itself — and "switch to a closure
without the undamped mode" is a real remedy at the root rather than a
coincidence.

The module docstring carries the full derivation: the diamond face involution
that makes every null vector a bulk-zero face sawtooth, the substitution that
empties the cell balance of all physics and leaves a purely combinatorial
identity, the character/ANOVA solution and its pair generators, both counting
laws as theorems, and the blocked :math:`G`-orthonormal representation that
makes the projector shippable.

.. automodule:: orpheus.sn.operators.loss_kernel_gauge
   :members:
   :show-inheritance:
