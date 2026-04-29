Analytical Derivations (``derivations``)
========================================

Reference for the :mod:`orpheus.derivations` package — the single
source of truth for analytical reference eigenvalues used across the
V&V ladder. Each derivation module is a SymPy- or closed-form
calculation that emits :class:`~orpheus.derivations._types.VerificationCase`
objects carrying the analytical :math:`k_\infty` (or :math:`k_\text{eff}`),
the material definitions, the geometry parameters, a LaTeX trace of the
derivation, and the V&V level.

Tests pull these cases through
:func:`orpheus.derivations.reference_values.get` (exposed at the
package top level as ``orpheus.derivations.get``), and the conftest hook
propagates ``vv_level`` / ``equation_labels`` from the case onto the
parametrized test node so that a single ``@pytest.mark.verifies(...)``
covers every consumer automatically.

Submodules
----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Purpose
   * - :mod:`~orpheus.derivations.homogeneous`
     - Infinite-medium :math:`k_\infty` for 1/2/4-group synthetic XS.
   * - :mod:`~orpheus.derivations.cp_slab`
     - Closed-form slab collision-probability eigenvalues.
   * - :mod:`~orpheus.derivations.cp_cylinder`
     - Closed-form cylindrical-annulus collision-probability eigenvalues.
   * - :mod:`~orpheus.derivations.diffusion`
     - 1-D / 2-region two-group diffusion eigenvalues matching the
       ``diffusion`` solver's core geometry.
   * - :mod:`~orpheus.derivations._kernels`
     - Exponential-integral (``E_3``) and Bickley–Naylor
       (``Ki_3``/``Ki_4``) kernels shared by the slab and cylinder
       derivations, plus the :func:`chord_half_lengths` chord-segment
       primitive consumed by chord-impact-parameter integrals.
   * - :mod:`~orpheus.derivations._quadrature`
     - Unified 1-D quadrature contract
       (:class:`~orpheus.derivations._quadrature.Quadrature1D` value
       object) and its primitive constructors:
       :func:`gauss_legendre`,
       :func:`gauss_legendre_visibility_cone`,
       :func:`composite_gauss_legendre`, :func:`gauss_laguerre`.
       Plus the sibling
       :class:`~orpheus.derivations._quadrature.AdaptiveQuadrature1D`
       (no-fixed-nodes adaptive rule built via
       :func:`~orpheus.derivations._quadrature.adaptive_mpmath`).
   * - :mod:`~orpheus.derivations._quadrature_recipes`
     - Geometry-aware quadrature recipes:
       :func:`chord_quadrature` (impact-parameter integrals on
       concentric annular geometries) and
       :func:`observer_angular_quadrature` (kink-aware ω-sweeps from
       an internal observer).
   * - :mod:`~orpheus.derivations._xs_library`
     - Synthetic cross-section library (``_FUEL_XS``, ``_REFL_XS``, …)
       that guarantees derivation cases and solver tests use the exact
       same numbers.
   * - :mod:`~orpheus.derivations._types`
     - :class:`VerificationCase` dataclass + ``VVLevel`` literal.
   * - :mod:`~orpheus.derivations.reference_values`
     - Lazy registry and lookup helpers (``get``, ``all_names``,
       ``by_geometry``, ``by_groups``, ``by_method``).

Reference-value registry
------------------------

.. automodule:: orpheus.derivations.reference_values
   :members:

Verification case type
----------------------

.. automodule:: orpheus.derivations._types
   :members:

Homogeneous
-----------

.. automodule:: orpheus.derivations.homogeneous
   :members:

Slab Collision Probability
--------------------------

.. automodule:: orpheus.derivations.cp_slab
   :members:
   :exclude-members: _XS_A, _XS_B

Cylindrical Collision Probability
---------------------------------

.. automodule:: orpheus.derivations.cp_cylinder
   :members:
   :exclude-members: _XS_A, _XS_B

Diffusion
---------

.. automodule:: orpheus.derivations.diffusion
   :members:

Kernels
-------

.. automodule:: orpheus.derivations._kernels
   :members:

Quadrature contract
-------------------

.. automodule:: orpheus.derivations._quadrature
   :members:

Quadrature recipes (geometry-aware)
-----------------------------------

.. automodule:: orpheus.derivations._quadrature_recipes
   :members:

Cross-section library
---------------------

.. automodule:: orpheus.derivations._xs_library
   :members:
