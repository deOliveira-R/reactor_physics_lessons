1-D Multigroup Diffusion Solver
================================

Reference for the :mod:`orpheus.diffusion` package — the #290
operator-algebra diffusion solver: the scalar-composite operator family
:math:`A = L + C - S - B` (four-term by construction: the diffusion
solver sums scattering and :math:`(n,2n)` emission into one :math:`S` at
its own composition site, so it has no separate :math:`N_{2n}` — see
:ref:`sn-n2n-adjoint`), the albedo-family boundary realizer
(:math:`J^- = \mathcal A\,J^+`), and the k-eigenvalue solver on the
shared :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` engines.
The underlying theory is covered in :doc:`/theory/methods/diffusion_1d` and
verified against the analytical / semi-analytical references built by
:mod:`orpheus.derivations.continuous.cases.diffusion`.

See :ref:`theory-verification` for the verification-case philosophy.

Method mesh
-----------

.. automodule:: orpheus.diffusion.augmented_mesh
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Solver
------

.. automodule:: orpheus.diffusion.solver
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Operators
---------

.. automodule:: orpheus.diffusion.operators
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Boundary realization
--------------------

.. automodule:: orpheus.diffusion.boundary_realizer
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: orpheus.diffusion.method_space
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
