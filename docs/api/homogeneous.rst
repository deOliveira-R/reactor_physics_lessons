Homogeneous Infinite-Medium Solver
====================================

The :mod:`orpheus.homogeneous` package solves the multi-group
eigenvalue problem in an infinite homogeneous medium — the
simplest reactor physics configuration and the foundation for
every spatial solver in ORPHEUS. Because the flux is uniform and
all streaming terms vanish, the transport equation collapses to a
single linear algebra problem per power iteration, with an exact
analytical structure that makes it the go-to harness for L0 /
L1 verification of cross-section libraries and scattering-matrix
conventions.

.. contents::
   :local:
   :depth: 2

.. seealso::

   :ref:`theory-homogeneous` — full derivation, scattering
   convention, and worked examples.


Eigenvalue Problem
------------------

With no spatial dependence, the multi-group transport equation
reduces to

.. math::

   \bigl(\operatorname{diag}(\Sigma_t)
   - \Sigma_{s0}^{\mathsf T}
   - 2\,\Sigma_{2}^{\mathsf T}\bigr)\,\phi
   \;=\; \frac{1}{k_\infty}\,\chi\,
   \bigl(\Sigma_p + 2\,\operatorname{colsum}(\Sigma_2)\bigr)^{\mathsf T}\phi,

where :math:`\Sigma_{s0}` is the :math:`P_0` (isotropic) scattering
matrix, :math:`\Sigma_2` is the (n,2n) cross-section matrix (stored
separately because each collision produces two neutrons), and
:math:`\chi` is the prompt fission spectrum. The
:math:`2\,\Sigma_2^{\mathsf T}` term on the left moves the (n,2n)
multiplication into the removal operator and the
:math:`2\,\mathrm{colsum}(\Sigma_2)` term on the right adds it to
the production rate.

**Scattering convention.**
:attr:`~orpheus.data.macro_xs.mixture.Mixture.SigS` stores matrices
in ``SigS[g_from, g_to]`` order — **the source uses the transpose**,
:math:`Q_{\rm scatter} = \Sigma_{s}^{\mathsf T}\phi`. The same
transpose appears in the removal matrix
:math:`\Sigma_{s0}^{\mathsf T}` above. This is the single convention
every ORPHEUS solver follows; mis-transposing is the most common
bug when porting from other codes and is caught by L0 spectrum
tests on asymmetric scattering matrices.


Implementation
--------------

:class:`~orpheus.homogeneous.solver.HomogeneousSolver` satisfies the
:class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` protocol and
plugs into the generic
:func:`~orpheus.numerics.eigenvalue.power_iteration` loop:

* ``initial_flux_distribution`` — flat flux of ones, length
  ``ng``.
* ``compute_fission_source`` — evaluates
  :math:`\chi\,(\Sigma_p + 2\,\mathrm{colsum}(\Sigma_2))\,\phi /
  k`.
* ``solve_fixed_source`` — one sparse direct solve
  (:func:`scipy.sparse.linalg.spsolve`) of the pre-assembled
  removal matrix :math:`A`. Because :math:`A` is constant across
  iterations, it is factored once in ``__init__``.
* ``compute_keff`` — Rayleigh quotient of production over
  absorption.
* ``converged`` — tolerance :math:`10^{-10}` on :math:`|\Delta k|`
  after a three-iteration warm-up.

The convenience wrapper
:func:`~orpheus.homogeneous.solver.solve_homogeneous_infinite`
runs power iteration, normalises the flux so that the total
production rate equals :math:`100\ {\rm n/cm^3/s}`, computes the
one-group collapsed production and absorption cross sections,
and packages everything into a
:class:`~orpheus.homogeneous.solver.HomogeneousResult`.

**Why five iterations is enough.**
In an infinite homogeneous medium there is no spatial eigenmode
spectrum to filter — the only mode is the fundamental spectral
shape, and power iteration reaches it after a handful of sweeps
regardless of the initial guess. Deterministic convergence on
realistic PWR mixtures typically locks :math:`k_\infty` to
:math:`10^{-10}` within five iterations.


API Reference
-------------

.. automodule:: orpheus.homogeneous.solver
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
