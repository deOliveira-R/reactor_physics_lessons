Numerical Methods (``numerics``)
================================

The :mod:`orpheus.numerics` package holds the algorithm-agnostic
numerical primitives that every deterministic solver shares. Its
job is to keep one copy of "how to converge an eigenvalue
problem" in the codebase rather than replicating the loop in each
of the SN, CP, MOC, diffusion, and homogeneous drivers.

.. contents::
   :local:
   :depth: 2


Power Iteration
---------------

The criticality eigenvalue problem

.. math::

   A\,\phi \;=\; \frac{1}{k}\,F\,\phi

has a spectrum of eigenvalues
:math:`k_0 > k_1 > k_2 > \dots`. Only the **dominant eigenvalue**
:math:`k_0 = k_{\rm eff}` and its eigenvector :math:`\phi_0` are
physically meaningful: by the Perron–Frobenius theorem
:math:`\phi_0` is the unique non-negative eigenvector, while all
higher harmonics change sign in space.

:func:`~orpheus.numerics.eigenvalue.power_iteration` converges to
:math:`(k_0, \phi_0)` by repeatedly applying the transport
operator to an estimate of :math:`\phi`:

.. math::

   \phi^{(n+1)} \;=\; A^{-1}\,\frac{1}{k^{(n)}}\,F\,\phi^{(n)},
   \qquad
   k^{(n+1)} \;=\; \frac{\lVert F\,\phi^{(n+1)}\rVert}
                         {\lVert L\,\phi^{(n+1)}\rVert}.

The convergence rate is governed by the **dominance ratio**
:math:`|k_1/k_0|`; problems with a narrow spectral gap (large
lattices, near-critical systems with weakly coupled regions)
converge slowly and may benefit from Chebyshev or Wielandt
acceleration — not currently implemented in ORPHEUS.

**Normalisation.**
The returned eigenvector has *arbitrary* absolute scale. Power
iteration preserves shape but not magnitude — callers that need
absolute flux (e.g. for power calibration, dose calculations) must
post-normalise, typically by fixing the total integral fission
source or the total power deposition.


The EigenvalueSolver Protocol
-----------------------------

Every deterministic solver plugs into the same power iteration
loop by implementing the
:class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` protocol.
The protocol has five methods and one structural contract:

* ``initial_flux_distribution`` — return a flux guess. Most
  solvers use a flat unit array; MOC uses a cell-averaged flat
  angular flux.
* ``compute_fission_source`` — build
  :math:`Q_f = \chi\,(\nu\Sigma_f\,\phi)/k`. Pure function of the
  current flux and eigenvalue.
* ``solve_fixed_source`` — apply :math:`A^{-1}` to the fission
  source. **Scattering and (n,2n) sources are assembled *inside*
  this method** because they need to be updated between inner
  iterations (source iteration in SN, Gauss–Seidel in CP, etc.).
  This is the single most important structural decision in the
  protocol: it lets each solver manage its own inner iteration
  strategy without leaking through to the outer loop.
* ``compute_keff`` — update the eigenvalue from the current
  :math:`\phi`. For reflective lattices the leakage term is zero;
  for whole-core diffusion it is not.
* ``converged`` — stopping test. Typical tolerance
  :math:`10^{-6}` on :math:`|\Delta k|`; richer tests on flux
  L2 norm are also used.

**Reference implementations** (each satisfies the protocol and is
tested against the power-iteration loop without any solver-specific
glue):

* :class:`orpheus.cp.solver.CPSolver` — collision probability.
* :class:`orpheus.sn.solver.SNSolver` — discrete ordinates.
* :class:`orpheus.moc.solver.MOCSolver` — method of characteristics.
* :class:`orpheus.diffusion.solver.DiffusionSolver` — 1-D two-group
  diffusion.
* :class:`orpheus.homogeneous.solver.HomogeneousSolver` — infinite
  homogeneous medium.


API Reference
-------------

.. automodule:: orpheus.numerics.eigenvalue
   :members:
   :undoc-members:
   :show-inheritance:
