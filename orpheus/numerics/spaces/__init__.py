r"""Function-space subclasses, organised by geometric / algebraic role.

The base :class:`~orpheus.numerics.space.FunctionSpace` lives in
``numerics/space.py``. This sub-package houses the specialised
subclasses that carry domain-specific metadata beyond
``(name, shape, inner_product_weights)``:

* :class:`SphericalHarmonicSpace` (P1.2) — moment-space carrier for SH
  coefficients with the ``MomentMassMatrix`` diagonal already broadcast
  to the storage layout.

* :class:`AngularTraceSpace` (#205 / #201 unification) — the single
  whole-boundary trace function space. Carries the :class:`FaceLayout`
  + the signed :math:`\Omega\cdot\hat n` per face; inflow / outflow are
  *selectors* over it (no longer separate ``Inflow`` / ``Outflow``
  subclasses).
* :class:`AngularFaceTraceSpace` (G6.1, #330) — one directional tier of
  that trace at ONE face: the whole ordinate slot :math:`\Gamma(f)`, or
  the half-traces :math:`\Gamma_+(f)` / :math:`\Gamma_-(f)`. These are
  the spaces the boundary operators BIND to, so that
  :math:`\gamma_\pm`, the deck transformation :math:`G` and the
  constitutive response :math:`R` carry a domain and a codomain and
  their adjoints fall out of :math:`A^\dagger = G_V^{-1}A^{\mathsf T}G_W`
  instead of being hand-rolled. Built by the parent trace space
  (:meth:`~AngularTraceSpace.face_space` /
  :meth:`~AngularTraceSpace.outflow_space` /
  :meth:`~AngularTraceSpace.inflow_space`), never directly. Note the
  three tiers do NOT partition — tangential ordinates belong to
  :math:`\Gamma(f)` alone.
* :class:`ScalarTraceSpace` (#290 P2) — the quadrature-free scalar
  sibling of :class:`AngularTraceSpace`: per-face ``(J⁺, J⁻)``
  partial-current pairs under the face-AREA metric, for methods whose
  boundary state is already angle-integrated (diffusion; CP / MoC
  scalar traces to follow).
* :class:`FullFieldSpace` (Wave O / O.2b R5) — the composite direct sum
  :math:`V_{\rm bulk} \oplus V_{\rm trace}` carrying the block-diagonal
  G-adjoint metric (bulk :math:`V\,w_n` :math:`\oplus` trace
  :math:`|\Omega\cdot\hat n|\,w_n`); the carrier of the FULL streaming
  operator and every bulk :math:`\oplus` boundary composite.
* Future: ``MeshFunctionSpace``, ``EnergyGroupSpace``,
  ``DiscreteAngularSpace`` per Grand Report v3 §5.3.

References
----------

* Grand Report v3 §5.3 — Space hierarchy.
* :mod:`orpheus.numerics.space` — the :class:`FunctionSpace` base.
"""

from __future__ import annotations

from orpheus.numerics.spaces.full_field_space import FullFieldSpace
from orpheus.numerics.spaces.legendre_space import LegendreSpace
from orpheus.numerics.spaces.moment_head import MomentHead
from orpheus.numerics.spaces.spherical_harmonic_space import (
    SphericalHarmonicSpace,
)
from orpheus.numerics.spaces.scalar_trace_space import ScalarTraceSpace
from orpheus.numerics.spaces.angular_trace_space import (
    AngularFaceTraceSpace,
    AngularTraceSpace,
    TraceRole,
)

__all__ = [
    "AngularFaceTraceSpace",
    "AngularTraceSpace",
    "FullFieldSpace",
    "LegendreSpace",
    "MomentHead",
    "ScalarTraceSpace",
    "SphericalHarmonicSpace",
    "TraceRole",
]
