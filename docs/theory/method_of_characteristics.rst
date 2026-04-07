.. _theory-method-of-characteristics:

====================================
Method of Characteristics (MOC)
====================================

.. contents:: Contents
   :local:
   :depth: 3


Key Facts
=========

**Read this before modifying the MOC solver.**

- Characteristic ODE: :math:`\frac{d\psi}{ds} + \Sigma_t \psi = Q / 4\pi`
- Flat-source solution: :math:`\bar\psi = \psi_{\text{in}} \frac{1 - e^{-\tau}}{\tau} + \frac{Q}{4\pi\Sigma_t}(1 - \frac{1-e^{-\tau}}{\tau})`
- Scalar flux update (Boyd Eq. 45): :math:`\phi = (4\pi Q + \Delta\phi / A) / \Sigma_t`
- Weight formula: :math:`4\pi \cdot \omega_a \cdot \omega_p \cdot t_s \cdot \sin(\theta_p)` — the :math:`4\pi` and :math:`\sin\theta_p` are invisible to homogeneous tests (ERR-019)
- Inverse Wigner-Seitz: pitch = ``mesh.edges[-1] * sqrt(pi)``, outermost annulus = square border
- Tabuchi-Yamamoto polar quadrature (TY-1/2/3) × uniform azimuthal
- Reflective BC via track linking: vertical → same direction, horizontal → reversed
- **Gotcha**: homogeneous flat-source is exact regardless of weight errors — only heterogeneous multi-region exposes bugs
- Ray-circle intersection is exact (analytical); ray-box uses parametric clipping
- Convergence: O(ray_spacing²) for spatial, spectral for angular
- Verification uses :ref:`synthetic cross sections <synthetic-xs-library>`, not real nuclear data

.. admonition:: Conventions

   - Scattering matrix: :ref:`scattering-matrix-convention` — ``SigS[g_from, g_to]``, source uses transpose
   - Multi-group balance: :eq:`mg-balance` in :ref:`theory-homogeneous`
   - Cross sections: :ref:`theory-cross-section-data`
   - Verification: :ref:`synthetic-xs-library` — regions A/B/C/D
   - Eigenvalue: :ref:`power-iteration-algorithm` shared with all deterministic solvers


Overview
========

The method of characteristics (MOC) solves the
:ref:`multi-group eigenvalue problem <mg-eigenvalue-problem>` in
integro-differential form by tracing **characteristic rays** across the
geometry and integrating the transport equation analytically along each
ray.  Unlike :ref:`discrete ordinates
<theory-discrete-ordinates>` (which discretises both angle and space on
a grid) or :ref:`collision probabilities <theory-collision-probability>`
(which integrates out the angular variable entirely), MOC retains the
angular flux :math:`\psi(\mathbf{r}, \hat{\Omega}, E)` along each
characteristic while using a **flat-source approximation** within
spatial sub-regions.

For a 2-D pin cell with concentric annuli inside a square lattice
cell, MOC offers several advantages over the other ORPHEUS solvers:

- **Exact circular geometry** --- rays intersect circles analytically,
  avoiding the staircase approximation of Cartesian SN or the white-BC
  approximation of CP.
- **Arbitrary angular resolution** --- configurable azimuthal angles and
  ray spacing (unlike the fixed 8 directions of the pedagogical solver).
- **Reflective boundary conditions** --- exact, without the isotropic
  re-entry assumption of the CP white BC.

The implementation follows [Boyd2014]_ (the OpenMOC formulation) for
the transport sweep and scalar flux update, and [Yamamoto2007]_ for
the Tabuchi-Yamamoto (TY) polar quadrature.  The original MOC
formulation for reactor physics is due to [Askew1972]_; the TY
quadrature tables are also documented in [KnottYamamoto2010]_.

**Derivation sources.**  The analytical eigenvalues used for
verification are computed by:

- ``derivations/_eigenvalue.py`` ---
  :func:`~derivations._eigenvalue.kinf_homogeneous` for infinite-medium
  eigenvalue (all solvers share this)
- ``derivations/moc.py`` --- homogeneous cases (analytical) and
  heterogeneous cases (:ref:`Richardson extrapolation <richardson-extrapolation>`
  of the MOC solver with ray-spacing refinement)

Every equation in this chapter can be verified against these scripts
and the cited references.

The key equations (segment-averaged angular flux, scalar flux update
weight, homogeneous consistency) are independently verified by SymPy
in ``derivations/moc_equations.py``.


Architecture
============

Two-Layer Mesh Pattern
----------------------

The MOC solver follows the same two-layer pattern as all ORPHEUS
deterministic solvers:

1. **Base geometry** --- :class:`~geometry.mesh.Mesh1D` with
   ``CoordSystem.CYLINDRICAL``, typically constructed by
   :func:`~geometry.factories.pwr_pin_equivalent`.  Stores radial cell
   edges, material IDs, and volumes.

2. **Augmented geometry** --- :class:`MOCMesh` wraps the ``Mesh1D`` and
   an :class:`MOCQuadrature`, precomputing all ray-tracing data:
   tracks, segments through flat-source regions, effective ray spacings,
   and reflective boundary condition links.

3. **Solver** --- :class:`MOCSolver` satisfies the
   :class:`~numerics.eigenvalue.EigenvalueSolver` protocol.
   :func:`solve_moc` orchestrates the calculation and returns an
   :class:`MoCResult`.

.. code-block:: text

   Mesh1D (cylindrical, Wigner-Seitz)
       |
       v
   MOCMesh (tracks + segments + reflective BC links)
       |
       v
   MOCSolver (EigenvalueSolver protocol)
       |
       v
   solve_moc() --> MoCResult


Inverse Wigner-Seitz Geometry
-----------------------------

The base ``Mesh1D`` is constructed by :func:`~geometry.factories.pwr_pin_equivalent`, which
approximates the square unit cell (side = pitch) by a cylinder of equal
area:

.. math::
   :label: moc-wigner-seitz

   r_{\text{cell}} = \frac{\text{pitch}}{\sqrt{\pi}}

All annular regions, including the outermost (coolant), are defined
within this equivalent cylinder.  Material IDs are assigned via
``mesh.mat_ids``.

``MOCMesh`` **inverts** this approximation: it recovers the pitch from
the outer Wigner-Seitz radius and reinterprets the outermost annular
region as the **square border** --- the physical region between the last
circular boundary and the square cell walls:

.. math::
   :label: pitch-recovery

   \text{pitch} = r_{\text{cell}} \cdot \sqrt{\pi} = \texttt{mesh.edges[-1]} \cdot \sqrt{\pi}

The circular boundaries used for ray tracing are ``mesh.edges[1:-1]``
(all radial edges except the origin and the Wigner-Seitz outer radius).
The outermost region is bounded by the last circle and the cell walls.

**Region areas** are exact:

.. math::

   A_k = \begin{cases}
     \pi(r_{k+1}^2 - r_k^2) & k < N-1 \text{ (annular)} \\
     \text{pitch}^2 - \pi r_k^2 & k = N-1 \text{ (square border)}
   \end{cases}

By construction of the Wigner-Seitz radius, the total area is
:math:`\sum A_k = \text{pitch}^2`, and the square-border area equals
the Wigner-Seitz annular area:
:math:`\text{pitch}^2 - \pi r_{N-2}^2 = \pi(r_\text{cell}^2 - r_{N-2}^2)`.

This convention means material IDs flow directly from ``mesh.mat_ids``
with no special treatment of the outer region.  The same ``Mesh1D``
used by the CP solver can be reused by MOC.


The Transport Equation Along a Characteristic
=============================================

Starting Point
--------------

The steady-state multi-group Boltzmann transport equation with
isotropic scattering:

.. math::
   :label: transport-equation

   \hat{\Omega} \cdot \nabla \psi_g(\mathbf{r}, \hat{\Omega})
   + \Sigt{g}(\mathbf{r}) \, \psi_g(\mathbf{r}, \hat{\Omega})
   = Q_g(\mathbf{r})

where :math:`Q_g` is the isotropic source (fission + scattering + (n,2n)),
independent of :math:`\hat{\Omega}`.

A **characteristic** (ray) is parametrised by arc length :math:`s` along
the direction :math:`\hat{\Omega}`:

.. math::

   \mathbf{r}(s) = \mathbf{r}_0 + s \, \hat{\Omega}, \quad s \geq 0

Along this ray, the PDE reduces to a first-order ODE:

.. math::
   :label: characteristic-ode

   \frac{d\psi_g}{ds} + \Sigt{g}(s) \, \psi_g(s) = Q_g(s)


.. _flat-source-approximation-moc:

Flat-Source Approximation
-------------------------

Within each flat-source region (FSR) :math:`i`, both :math:`\Sigt{i,g}`
and :math:`Q_{i,g}` are assumed spatially constant.  The CP method uses
the same approximation; see :ref:`flat-source-approximation-cp`.
For a ray segment
of 2-D length :math:`\ell` crossing region :math:`i` at polar angle
:math:`\theta_p`, the 3-D path length is :math:`\ell / \sin\theta_p`
and the **optical thickness** is:

.. math::
   :label: optical-thickness

   \tau_{g,p} = \Sigt{i,g} \cdot \frac{\ell}{\sin\theta_p}

The ODE :eq:`characteristic-ode` has the exact analytical solution:

.. math::
   :label: attenuation

   \psi_g^{\text{out}} = \psi_g^{\text{in}} \, e^{-\tau_{g,p}}
   + \frac{Q_{i,g}}{\Sigt{i,g}} \bigl(1 - e^{-\tau_{g,p}}\bigr)

This is the MOC **attenuation formula**.  The first term is the
uncollided flux from the incoming angular flux; the second term is the
contribution from the in-region source.

The **angular flux change** across the segment is:

.. math::
   :label: delta-psi

   \Delta\psi_{g,p} = \psi_g^{\text{in}} - \psi_g^{\text{out}}
   = \Bigl(\psi_g^{\text{in}} - \frac{Q_{i,g}}{\Sigt{i,g}}\Bigr)
     \bigl(1 - e^{-\tau_{g,p}}\bigr)

This quantity is the central building block of the MOC scalar flux
update.

.. note::

   **Numerical stability.**  For small :math:`\tau < 10^{-10}`, the
   exponential is computed via Taylor expansion
   :math:`1 - e^{-\tau} \approx \tau(1 - \tau/2)` to avoid
   catastrophic cancellation.  This is implemented in the inner loop
   of :meth:`MOCSolver.solve_fixed_source`.


The Isotropic Source
--------------------

For isotropic scattering and isotropic fission emission:

.. math::
   :label: isotropic-source

   Q_{i,g} = \frac{1}{4\pi} \left[
     \sum_{g'=1}^{G} \Sigs{g' \to g} \, \phi_{i,g'}
     + 2 \sum_{g'=1}^{G} \Sigma_{2,g' \to g} \, \phi_{i,g'}
     + F_{i,g}
   \right]

where :math:`F_{i,g}` is the fission source (from the outer iteration):

.. math::

   F_{i,g} = \frac{\chi_{i,g}}{k_{\text{eff}}} \sum_{g'=1}^{G} \nu\Sigf{i,g'} \, \phi_{i,g'}

The :math:`1/(4\pi)` normalisation converts from volumetric source
density to isotropic angular source density.

**Scattering convention.**  ``SigS[0]`` is indexed as
``SigS[g_from, g_to]``.  The in-scatter source uses the transpose:
``Q += SigS.T @ phi`` (scattering *into* group :math:`g` from all
:math:`g'`).  This is the same convention used by all ORPHEUS solvers
(CP, SN, MC).  The :math:`(n,2n)` matrix follows the same convention
with a factor of 2.


Angular Quadrature
==================

MOC uses a **product quadrature**: azimuthal angles in the 2-D plane
times polar angles from the z-axis.  Contrast with the
:ref:`SN quadrature types <quadrature-types>`, which use Gauss-Legendre
(1D) and Lebedev (2D).

Azimuthal Quadrature
--------------------

:math:`N_\varphi` uniformly spaced angles in :math:`[0, \pi)`:

.. math::
   :label: azimuthal-angles

   \varphi_m = \frac{\pi}{2 N_\varphi} + (m-1) \frac{\pi}{N_\varphi},
   \quad m = 1, \ldots, N_\varphi

with weights :math:`\omega_m^a = 1 / N_\varphi` (summing to 1).

The supplementary angle :math:`\varphi_m + \pi` corresponds to the
**same physical track** traversed in the opposite direction (the
backward sweep).  Thus, only :math:`[0, \pi)` needs to be traced
explicitly; the backward direction is obtained by reversing the
segment order.

Typical values: :math:`N_\varphi = 16, 32, 64` for increasing accuracy.

Tabuchi-Yamamoto Polar Quadrature
----------------------------------

The polar angle :math:`\theta_p` enters the MOC only through the
effective 3-D optical thickness :math:`\tau / \sin\theta_p`.  The
angular integration involves the Bickley function:

.. math::
   :label: bickley-integral

   \text{Ki}_3(\tau) = \int_0^{\pi/2} e^{-\tau/\sin\theta} \sin\theta \, d\theta

Yamamoto et al. [Yamamoto2007]_ derived optimal polar angles and
weights that minimise the maximum approximation error of the
Bickley function.  The TY quadrature with :math:`N_p = 3` polar
angles per half-space achieves accuracy comparable to Gauss-Legendre
with 12--16 points.

The TY weights include the :math:`\sin\theta` Jacobian from the
angular measure :math:`d\Omega = \sin\theta \, d\theta \, d\varphi`.
They sum to 0.5 for one hemisphere (upper or lower); the full sphere
gives :math:`2 \times 0.5 = 1.0`.

.. list-table:: Tabuchi-Yamamoto quadrature tables [Yamamoto2007]_
   :header-rows: 1
   :widths: 10 10 20 20

   * - :math:`N_p`
     - :math:`p`
     - :math:`\sin\theta_p`
     - :math:`\omega_p`
   * - 1
     - 1
     - 0.798184
     - 0.500000
   * - 2
     - 1
     - 0.363900
     - 0.106427
   * - 2
     - 2
     - 0.899900
     - 0.393573
   * - 3
     - 1
     - 0.166648
     - 0.023117
   * - 3
     - 2
     - 0.537707
     - 0.141810
   * - 3
     - 3
     - 0.932954
     - 0.335074

These tables are hardcoded in :class:`MOCQuadrature` from [Yamamoto2007]_
Table 2.  Verified by ``test_moc_quadrature.py::test_ty3_values_match_published``.

The combined angular weight normalisation satisfies:

.. math::

   2 \sum_{m=1}^{N_\varphi} \omega_m^a \sum_{p=1}^{N_p} \omega_p
   = 2 \cdot 1 \cdot 0.5 = 1

The factor of 2 accounts for the two hemispheres (upper = forward sweep,
lower = backward sweep).  This is verified by
``test_moc_quadrature.py::test_combined_weight_normalisation``.


Ray Tracing
===========

Track Generation
----------------

For each azimuthal angle :math:`\varphi_m`, a family of parallel rays is
laid across the square cell :math:`[0, \text{pitch}]^2` with
perpendicular spacing :math:`t_s`.

The perpendicular coordinate of a ray is:

.. math::

   t = -x \sin\varphi_m + y \cos\varphi_m

The range of :math:`t` values that intersect the cell is determined by
projecting the four cell corners onto the perpendicular axis.  Rays are
placed at :math:`t_k = t_{\min} + (k + \tfrac{1}{2}) t_s^{\text{eff}}`,
where:

.. math::
   :label: effective-spacing

   t_s^{\text{eff}} = \frac{t_{\max} - t_{\min}}{n_{\text{rays}}},
   \quad
   n_{\text{rays}} = \left\lceil \frac{t_{\max} - t_{\min}}{t_s} \right\rceil

The effective spacing is chosen so that rays exactly cover the cell
width at each angle, with no gaps.

Ray-Circle Intersection
-----------------------

A ray starting at :math:`(x_0, y_0)` with direction
:math:`(\cos\varphi, \sin\varphi)` is parametrised as:

.. math::

   x(s) = x_0 + s \cos\varphi, \quad
   y(s) = y_0 + s \sin\varphi

The intersection with a circle of radius :math:`R_k` centred at
:math:`(c_x, c_y)` satisfies:

.. math::
   :label: ray-circle

   s^2 + 2bs + c = 0

where:

.. math::

   b = (x_0 - c_x)\cos\varphi + (y_0 - c_y)\sin\varphi, \quad
   c = (x_0 - c_x)^2 + (y_0 - c_y)^2 - R_k^2

The discriminant :math:`\Delta = b^2 - c` determines whether the ray
intersects the circle:

- :math:`\Delta < 0` --- ray misses the circle (no intersection)
- :math:`\Delta \geq 0` --- two intersections at
  :math:`s_{1,2} = -b \mp \sqrt{\Delta}`

Only intersections within the cell (between entry and exit parameters)
are retained.

Segment Construction
--------------------

For a given ray through the cell:

1. Compute all intersection parameters: cell-wall entry/exit (from
   ``_ray_box_intersections()``) and circle crossings (from
   ``_ray_circle_intersections()``).
2. Sort all parameters and remove near-duplicates.
3. For each consecutive pair :math:`(s_a, s_b)`, the segment length is
   :math:`\ell = s_b - s_a`.  The **midpoint** determines which FSR the
   segment belongs to (by computing distance from pin centre).
4. Build the segment list: ``(region_id, length)`` tuples.

**Region assignment.**  A point at distance :math:`d` from the pin
centre belongs to region :math:`k` where
:math:`r_{k-1} < d \leq r_k` for :math:`k < N-1`, or region
:math:`N-1` (square border) if :math:`d > r_{N-2}`.

**Verified by** ``test_moc_ray_tracing.py``:

- ``test_ray_circle_chord_length``: chord length =
  :math:`2\sqrt{R^2 - d^2}` for impact parameter :math:`d`
- ``test_trace_three_regions``: correct segment count (5) and region
  ordering for 3 concentric annuli
- ``test_volume_conservation``: sum of (segment length × ray spacing)
  approximates region areas to < 5% at ``ray_spacing=0.02``


Reflective Boundary Conditions
==============================

For a pin cell with reflective BCs, each ray that exits through a cell
wall re-enters as a reflected ray.  The outgoing angular flux from one
track becomes the incoming flux for the reflected track.

Reflection Rules
----------------

For azimuthal angles :math:`\varphi \in [0, \pi)`, all forward rays
have :math:`\sin\varphi \geq 0` (upward or horizontal).  Two types of
reflection occur:

.. list-table:: Reflection rules for angles in :math:`[0, \pi)`
   :header-rows: 1
   :widths: 15 20 20 25

   * - Sweep
     - Exit wall
     - Reflected angle
     - Target direction
   * - Forward
     - Vertical (L/R)
     - :math:`\pi - \varphi`
     - Forward
   * - Forward
     - Horizontal (T)
     - :math:`\pi - \varphi`
     - **Backward**
   * - Backward
     - Horizontal (B)
     - :math:`\pi - \varphi`
     - **Forward**
   * - Backward
     - Vertical (L/R)
     - :math:`\pi - \varphi`
     - Backward

**Key distinction:**  Both reflection types map the azimuthal angle to
:math:`\pi - \varphi`.  The difference is whether the reflected ray
continues in the **same traversal direction** (forward → forward at the
reflected angle) or **reverses** (forward → backward at the reflected
angle).

- **Vertical wall** (right :math:`x = P`, left :math:`x = 0`):
  reverses the :math:`x`-component of :math:`\hat{\Omega}`.  Since
  :math:`(-\cos\varphi, \sin\varphi) = (\cos(\pi-\varphi), \sin(\pi-\varphi))`,
  this is the **forward direction** at the reflected angle.

- **Horizontal wall** (top :math:`y = P`, bottom :math:`y = 0`):
  reverses the :math:`y`-component.  Since
  :math:`(\cos\varphi, -\sin\varphi) = -(\cos(\pi-\varphi), \sin(\pi-\varphi))`,
  this is the **backward direction** at the reflected angle.

**Summary rule:** vertical reflections preserve forward/backward;
horizontal reflections flip it.  In the code,
``_is_vertical()`` tests this and
:meth:`MOCMesh._link_tracks` uses it to set ``fwd_link_fwd`` and
``bwd_link_fwd``.

Track Linking
-------------

After generating all tracks, :meth:`MOCMesh._link_tracks` assigns four
link fields per track:

- ``fwd_link``, ``fwd_link_fwd``: where the forward sweep's outgoing
  flux goes (target track index, and whether it feeds into the target's
  forward or backward entry)
- ``bwd_link``, ``bwd_link_fwd``: same for the backward sweep

The target track is found by matching the exit point of one track to the
entry point (for forward targets) or exit point (for backward targets)
of a track at the reflected azimuthal angle.  The closest match is used.

**Verified by** ``test_moc_verification.py::TestL0GeometricInvariants::
test_reflective_links_form_cycles``: following forward links from any
track must return to the starting track after a finite number of
reflections (closed cycle).


Scalar Flux Update
==================

Derivation from First Principles
---------------------------------

The scalar flux in FSR :math:`i` for energy group :math:`g` is the
angular integral of the segment-averaged angular flux over all
directions:

.. math::
   :label: scalar-flux-integral

   \phi_{i,g} = \frac{1}{A_i} \int_{4\pi}
   \int_{A_i} \bar{\psi}_g(\mathbf{r}, \hat{\Omega}) \, dA \, d\Omega

where :math:`\bar{\psi}` is the angular flux averaged along the
characteristic segment through region :math:`i`.

**Segment-averaged angular flux.**  The average angular flux along a
segment of 3-D path length :math:`L = \ell / \sin\theta_p` is obtained
by integrating the ODE solution :eq:`attenuation` over the segment:

.. math::

   \bar{\psi}_{k,g,p}
   = \frac{1}{L} \int_0^{L} \psi_g(s) \, ds
   = \frac{1}{L} \int_0^{L} \left[
       \psi_g^{\text{in}} \, e^{-\Sigt{} s}
       + \frac{Q}{\Sigt{}} (1 - e^{-\Sigt{} s})
     \right] ds

Evaluating the integral term by term:

.. math::

   \int_0^L \psi^{\text{in}} e^{-\Sigt{} s} \, ds
   = \frac{\psi^{\text{in}}}{\Sigt{}} (1 - e^{-\tau})

.. math::

   \int_0^L \frac{Q}{\Sigt{}} (1 - e^{-\Sigt{} s}) \, ds
   = \frac{Q}{\Sigt{}} \left[ L - \frac{1-e^{-\tau}}{\Sigt{}} \right]

where :math:`\tau = \Sigt{} L = \Sigt{i,g} \ell_k / \sin\theta_p`.
Combining and dividing by :math:`L`:

.. math::

   \bar{\psi} = \frac{Q}{\Sigt{}}
   + \frac{1}{\tau} \left( \psi^{\text{in}} - \frac{Q}{\Sigt{}} \right)
     (1 - e^{-\tau})

Recognising :math:`\Delta\psi = (\psi^{\text{in}} - Q/\Sigt{})(1 - e^{-\tau})`
from :eq:`delta-psi`:

.. math::
   :label: bar-psi

   \boxed{
   \bar{\psi}_{k,g,p} = \frac{Q_{i,g}}{\Sigt{i,g}}
   + \frac{\Delta\psi_{k,g,p}}{\tau_{k,g,p}}
   }

This result is exact under the flat-source approximation.  The first
term is the asymptotic angular flux in the region; the second term is
the correction from non-equilibrium boundary flux.

**Physical interpretation.**  In an infinite homogeneous medium,
:math:`\psi^{\text{in}} = Q/\Sigt{}` everywhere, so
:math:`\Delta\psi = 0` and :math:`\bar\psi = Q/\Sigt{}`.  The
correction term carries the spatial coupling between heterogeneous
regions --- it is the only term that distinguishes fuel from coolant.
This is why the angular integration weight (which multiplies
:math:`\Delta\psi`) is invisible to homogeneous tests: the quantity
it multiplies is identically zero.

Substituting :eq:`bar-psi` into :eq:`scalar-flux-integral`:

.. math::

   \phi_{i,g} = \frac{1}{A_i} \int_{4\pi} \sum_{k \in i}
   t_s \, \ell_k \left[
     \frac{Q_{i,g}}{\Sigt{i,g}}
     + \frac{\Delta\psi_{k,g,p} \, \sin\theta_p}{\Sigt{i,g} \ell_k}
   \right] d\Omega

where the :math:`\sin\theta_p` factor arises from
:math:`\tau = \Sigt{} \ell / \sin\theta_p` in the denominator of
:eq:`bar-psi`: :math:`\Delta\psi / \tau = \Delta\psi \sin\theta_p / (\Sigt{} \ell)`.

The first term uses :math:`\sum_k t_s \ell_k \approx A_i` (the tracks
at each azimuthal angle tile the region area), and
:math:`\int_{4\pi} d\Omega = 4\pi`:

.. math::

   \phi_{i,g} = \frac{4\pi Q_{i,g}}{\Sigt{i,g}}
   + \frac{1}{A_i \, \Sigt{i,g}}
     \int_{4\pi} \sin\theta_p
     \sum_{k \in i} t_s \, \Delta\psi_{k,g,p} \, d\Omega

Discretising the angular integral with the product quadrature (azimuthal
:math:`\omega_m^a`, polar :math:`\omega_p`, factor of
:math:`4\pi` for the full sphere, and factor of 2 for forward + backward sweeps
absorbed into the summation over both sweep directions):

.. math::
   :label: boyd-eq-45

   \boxed{
   \phi_{i,g} = \frac{1}{\Sigt{i,g}} \left[
     4\pi \, Q_{i,g}
     + \frac{1}{A_i} \sum_{m,p,k \in i}
       4\pi \, \omega_m^a \, \omega_p \, t_s \, \sin\theta_p \,
       \Delta\psi_{k,g,p}
   \right]
   }

This is [Boyd2014]_ Equation 45 (with our weight normalisation
convention).

**Implementation.**  In :meth:`MOCSolver.solve_fixed_source`, the
accumulator ``delta_phi[i, g]`` collects the weighted
:math:`\Delta\psi` contributions during the sweep.  The weight per
segment is:

.. code-block:: python

   weight = 4.0 * np.pi * omega_a * omega_p * ts * sin_p

After the sweep, the scalar flux is updated as::

   phi[i, g] = (4*pi*Q[i,g] + delta_phi[i,g] / area[i]) / sig_t[i,g]


.. admonition:: ERR-019 — Missing weight factor (caught during development)

   **Bug:** The initial implementation used ``weight = omega_a * omega_p * ts``,
   missing both the :math:`4\pi` full-sphere factor and the :math:`\sin\theta_p`
   factor from :eq:`boyd-eq-45`.

   **Impact:** Heterogeneous keff was completely wrong: MOC gave 1.344 vs
   CP reference of 0.902 for a 2-region pin cell.  All three homogeneous
   tests passed to machine precision because :math:`\Delta\psi = 0` when
   the angular flux is spatially uniform.

   **How it hid:** For homogeneous material, :math:`\psi_{\text{in}} = Q / \Sigt{}`
   everywhere, so :math:`\Delta\psi = 0`.  The scalar flux reduces to
   :math:`4\pi Q / \Sigt{}` regardless of the weight.  This is the
   fundamental degeneracy: **weights are invisible to homogeneous tests.**

   **What caught it:** The first heterogeneous cross-verification against
   the CP solver revealed a 0.44 discrepancy.  Re-deriving the weight
   formula from :eq:`scalar-flux-integral` identified the two missing
   factors.

   **Lesson:** Always verify the transport sweep with a heterogeneous
   problem before trusting the homogeneous result.  The weight formula
   should be derived from first principles, not guessed by analogy.
   See ``tests/l0_error_catalog.md`` ERR-019.


Eigenvalue Update
-----------------

For reflective boundary conditions (zero leakage), the eigenvalue is:

.. math::
   :label: moc-keff-update

   k_{\text{eff}} = \frac{\text{production}}{\text{absorption}}
   = \frac{\sum_i \sum_g (\nSigf{i,g} + 2 \Sigma_{2,i,g}^{\text{out}})
     \phi_{i,g} \, A_i}
     {\sum_i \sum_g \Siga{i,g} \, \phi_{i,g} \, A_i}

where :math:`\Sigma_{2,i,g}^{\text{out}} = \sum_{g'} \Sigma_{2,g \to g'}`
is the total (n,2n) transfer out of group :math:`g`.

**Derivation.**  Each (n,2n) reaction absorbs one neutron and produces
two.  In the eigenvalue balance:

- **Production** (numerator): :math:`\nSigf{}` (fission) + :math:`2\Sigma_2^{\text{out}}`
  (two neutrons from each (n,2n))
- **Absorption** (denominator): :math:`\Siga{} = \Sigma_c + \Sigma_f + \Sigma_L +
  \Sigma_2^{\text{out}}` (capture + fission + leakage + (n,2n) removal)

The (n,2n) appears in BOTH numerator (2×, production) and denominator
(1×, removal).  The net contribution per (n,2n) reaction is +1 neutron.

When :math:`\Sigma_2 = 0`, :eq:`moc-keff-update` reduces to the standard
:math:`k = \nSigf{}\phi A / \Siga{}\phi A`.

Implemented in :meth:`MOCSolver.compute_keff`.  Verified by
``test_moc_verification.py::TestL0N2nReaction::test_n2n_1g_analytical_keff``.


Power Iteration
===============

The MOC solver plugs into the generic :func:`~numerics.eigenvalue.power_iteration`
loop via the :class:`~numerics.eigenvalue.EigenvalueSolver` protocol:

1. **Fission source** (:meth:`MOCSolver.compute_fission_source`):
   :math:`F_{i,g} = \chi_{i,g} \cdot (\nSigf{i} \cdot \phi_i) / k`

2. **Transport sweep** (:meth:`MOCSolver.solve_fixed_source`):
   Build :math:`Q` from :eq:`isotropic-source`, sweep all tracks
   (forward + backward), accumulate :math:`\Delta\psi` into
   ``delta_phi``, update :math:`\phi` from :eq:`boyd-eq-45`.

   The method performs ``n_inner_sweeps`` transport sweeps per outer
   iteration to converge the boundary angular fluxes.  Within each
   sweep, the scattering source is updated from the latest flux.

   **Boundary flux persistence:** The angular fluxes at track
   entry/exit points (``_fwd_bflux``, ``_bwd_bflux``) persist between
   outer iterations.  This allows the reflective BCs to converge
   progressively without explicit cyclic tracking.

3. **Eigenvalue update** (:meth:`MOCSolver.compute_keff`):
   :eq:`moc-keff-update`.

4. **Convergence** (:meth:`MOCSolver.converged`):
   :math:`|\Delta k| < \texttt{keff\_tol}` and
   :math:`\|\Delta\phi\| / \|\phi\| < \texttt{flux\_tol}`.

The power iteration algorithm is shared with all ORPHEUS eigenvalue
solvers; see :ref:`power-iteration-algorithm` in the homogeneous theory
for the general formulation.


Cross-Section Data Layout
=========================

Cross sections flow through two paths:

1. :class:`~data.macro_xs.mixture.Mixture` --- per-material arrays for
   :math:`\Sigt{}`, :math:`\Siga{}`, :math:`\nSigf{}`, :math:`\chi`;
   sparse matrices for :math:`\Sigma_s` (``SigS[0]``) and
   :math:`\Sigma_2` (``Sig2``).

2. Per-region assembly in :meth:`MOCSolver.__init__` --- loops over
   ``MOCMesh.region_mat_ids`` to build ``(n_regions, ng)`` arrays.

**Indexing:** ``sig_t[i, g]`` is the total cross section in FSR
:math:`i`, group :math:`g`.  Group 0 = fastest; group :math:`G-1` =
thermal.  Region 0 = innermost annulus; region :math:`N-1` = square
border.

**Scattering convention:** ``SigS[g_from, g_to]``.  The in-scatter
source uses the transpose: ``Q += SigS.T @ phi`` (same as CP, SN, MC).


Verification
============

102 Tests Across Four Levels
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 20 15 35

   * - Level
     - File(s)
     - Tests
     - Coverage
   * - L0
     - ``test_moc_quadrature.py``
     - 24
     - Weight sums, TY values, shapes, validation
   * - L0
     - ``test_moc_ray_tracing.py``
     - 20
     - Ray-circle intersection, region ID, segments, volume, links
   * - L0
     - ``test_moc_verification.py``
     - 27
     - Single-track attenuation, equilibrium flux, fission-only, (n,2n),
       scatter isolation, geometric invariants, protocol compliance, volume
       tracking, boundary conditions
   * - L1
     - ``test_moc.py``
     - 6
     - Homogeneous eigenvalue (1G/2G/4G), heterogeneous (slow)
   * - L1
     - ``test_moc_properties.py``
     - 4
     - Particle balance, positivity, flux consistency, thermal depression
   * - L1
     - ``test_moc_verification.py``
     - 13
     - Eigenvalue + flux ratio, heterogeneous + monotonicity, particle
       balance (2G), flux positivity (all groups), material sensitivity
   * - L2
     - ``test_moc_verification.py``
     - 4
     - Ray spacing convergence, azimuthal convergence, polar convergence
   * - XV
     - ``test_moc_verification.py``
     - 2
     - MOC vs CP cross-verification (slow)

Run the full suite (excluding slow tests)::

   pytest tests/test_moc_quadrature.py tests/test_moc_ray_tracing.py \
          tests/test_moc.py tests/test_moc_properties.py \
          tests/test_moc_verification.py -v -k "not slow"


Homogeneous Infinite Medium
----------------------------

For homogeneous geometry with reflective BCs, the flux is spatially flat
and :math:`k_{\text{eff}} = \lambda_{\max}(A^{-1}F)`.

.. list-table::
   :header-rows: 1
   :widths: 10 14 20 20

   * - Groups
     - :math:`k_\infty`
     - Error (8 azi, TY-3)
     - Tolerance
   * - 1
     - 1.5000
     - :math:`< 10^{-15}` (exact)
     - :math:`< 10^{-4}`
   * - 2
     - 1.8750
     - :math:`\sim 10^{-6}`
     - :math:`< 10^{-4}`
   * - 4
     - 1.4878
     - :math:`\sim 10^{-6}`
     - :math:`< 10^{-4}`

The 1-group case is exact because :math:`k = \nu\Sigma_f / \Sigma_a`
is independent of the angular flux distribution.  The multi-group cases
converge through the power iteration.


Why Homogeneous Verification Is Necessary But Insufficient
-----------------------------------------------------------

For a homogeneous medium, the flat-source approximation is **exact**:
the source is genuinely spatially flat.  Consequences:

1. :math:`\Delta\psi = 0` everywhere (all boundary fluxes equal
   :math:`Q / \Sigt{}`).
2. Angular integration weights cancel in the eigenvalue ratio.
3. The transport sweep contributes nothing --- :math:`\phi = 4\pi Q / \Sigt{}`.

This means homogeneous tests **cannot detect**:

- Wrong angular integration weights (ERR-019)
- Wrong scattering convention (SigS vs SigS.T)
- Wrong boundary condition linking
- Wrong ray spacing / volume conservation

**Rule:** Every transport solver must be verified on heterogeneous
multi-group problems.  Homogeneous success gives false confidence.


Heterogeneous Cross-Verification
---------------------------------

For a 2-region cylindrical pin cell (fuel + coolant), the MOC
eigenvalue is compared against the CP solver on the same geometry:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20

   * - Groups
     - MOC :math:`k_{\text{eff}}`
     - CP :math:`k_{\text{eff}}`
     - Gap
   * - 2
     - 0.6078
     - 0.6072
     - 0.001
   * - 4
     - 0.5399
     - 0.5394
     - 0.001

The ~0.1% gap is consistent with the white-BC (CP) vs reflective-BC
(MOC) approximation difference.  This is verified by
``test_moc_verification.py::TestXVCrossVerification``.


Convergence Properties
-----------------------

All convergence studies use a 2-region 2G pin cell (fuel + coolant,
:math:`r_{\text{fuel}} = 0.5` cm, pitch = 2.0 cm) with materials A
(fissile) and B (scatterer) from the verification library.

**Ray spacing convergence** (:math:`N_\varphi = 32`, TY-3):

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15

   * - :math:`t_s` (cm)
     - Tracks
     - :math:`k_{\text{eff}}`
     - :math:`|\Delta k|`
     - Ratio
   * - 0.16
     - 528
     - 0.608256
     -
     -
   * - 0.08
     - 1036
     - 0.608021
     - :math:`2.36 \times 10^{-4}`
     -
   * - 0.04
     - 2048
     - 0.608129
     - :math:`1.09 \times 10^{-4}`
     - 2.2
   * - 0.02
     - 4092
     - 0.608038
     - :math:`9.10 \times 10^{-5}`
     - 1.2
   * - 0.01
     - 8168
     - 0.608029
     - :math:`9.23 \times 10^{-6}`
     - 9.9

The eigenvalue converges with decreasing ray spacing.  The convergence
is not perfectly monotonic (non-monotone behaviour between 0.08 and 0.04
reflects the discrete nature of track placement on the annular boundary),
but the refinement from 0.02 to 0.01 shows a clear order-of-magnitude
improvement.

**Azimuthal convergence** (:math:`t_s = 0.02` cm, TY-3):

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15

   * - :math:`N_\varphi`
     - Tracks
     - :math:`k_{\text{eff}}`
     - :math:`|\Delta k|`
   * - 4
     - 524
     - 0.605797
     -
   * - 8
     - 1028
     - 0.607311
     - :math:`1.51 \times 10^{-3}`
   * - 16
     - 2048
     - 0.607959
     - :math:`6.47 \times 10^{-4}`
   * - 32
     - 4092
     - 0.608038
     - :math:`7.96 \times 10^{-5}`
   * - 64
     - 8188
     - 0.608057
     - :math:`1.89 \times 10^{-5}`

The azimuthal convergence shows a clean reduction factor of ~2--8×
per doubling of :math:`N_\varphi`.  The error at :math:`N_\varphi = 64`
is dominated by the ray spacing, not the angular discretisation.

**Polar convergence** (:math:`N_\varphi = 16`, :math:`t_s = 0.03` cm):

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 20

   * - :math:`N_p`
     - :math:`k_{\text{eff}}`
     - :math:`|\Delta k|` from TY-1
     - Quadrature points
   * - TY-1
     - 0.607538
     -
     - 1 per half-space
   * - TY-2
     - 0.607632
     - :math:`9.4 \times 10^{-5}`
     - 2 per half-space
   * - TY-3
     - 0.607790
     - :math:`2.5 \times 10^{-4}`
     - 3 per half-space

The polar convergence is weak: TY-1 to TY-3 gives only a
:math:`\sim 2.5 \times 10^{-4}` shift.  This is because the TY
quadrature is specifically optimised for the Bickley-function integrals
that arise in MOC, so even TY-1 (a single polar angle) gives a
surprisingly good approximation.  **Recommendation:** TY-3 is the
default; increasing to TY-2 or TY-1 is acceptable for quick
exploratory runs.


Investigation History: ERR-019 (Weight Factor)
===============================================

This section documents the development history of the scalar flux
update formula to prevent future sessions from repeating the same
error.

Symptoms
--------

1. All three homogeneous eigenvalue tests passed to machine precision
   (1G exact, 2G/4G :math:`\sim 10^{-6}`).
2. First heterogeneous test (2-region, 1G): MOC gave
   :math:`k = 1.344`, CP gave :math:`k = 0.902`.
3. The discrepancy was 0.44 --- far too large to be a discretisation
   effect.

The Root Cause
--------------

The delta_phi accumulation weight was ``omega_a * omega_p * ts``,
missing two factors:

1. :math:`4\pi` --- the angular integration normalisation from
   :math:`\int_{4\pi} d\Omega = 4\pi`.
2. :math:`\sin\theta_p` --- the 2-D to 3-D projection factor arising
   from the segment-averaged angular flux formula :eq:`bar-psi`.

Why It Was Invisible to Homogeneous Tests
------------------------------------------

For homogeneous material with converged boundary fluxes:

.. math::

   \psi_{\text{in}} = \frac{Q}{\Sigt{}} \implies \Delta\psi = 0
   \implies \delta\phi = 0

The scalar flux reduces to :math:`\phi = 4\pi Q / \Sigt{}` regardless
of the weight factor.  This is the fundamental degeneracy:
**weights are invisible when** :math:`\Delta\psi = 0`.

What Caught It
--------------

The first heterogeneous cross-verification against the CP solver
showed a 0.44 discrepancy.  Re-deriving the weight from
:eq:`scalar-flux-integral` through :eq:`boyd-eq-45` identified the
two missing factors.  After the fix, MOC agreed with CP to within
:math:`1.2 \times 10^{-3}` (consistent with the white-BC vs
reflective-BC approximation difference).

Lesson
------

The angular integration weight in MOC contains factors that cancel for
spatially uniform solutions:

- :math:`4\pi` cancels because the isotropic source already divides
  by :math:`4\pi`
- :math:`\sin\theta_p` cancels because :math:`\Delta\psi = 0`

Always verify the weight formula against a heterogeneous problem
**before** trusting the homogeneous result.  Derive the weight from
first principles and verify each factor independently.


Design Decisions and Alternatives
===================================

Why Flat-Source (Not Linear-Source)
------------------------------------

The flat-source approximation assumes :math:`Q(\mathbf{r}) = Q_i` within
each FSR.  The alternative is the **linear-source** approximation, which
expands the source as :math:`Q(\mathbf{r}) = Q_i + \nabla Q_i \cdot
(\mathbf{r} - \mathbf{r}_i)`, requiring a spatial gradient per region.

**Trade-off:** Linear-source MOC converges at :math:`O(h^3)` vs
:math:`O(h^2)` for flat-source (where :math:`h` is the characteristic
width of an FSR), allowing coarser meshes for the same accuracy.
However, it requires gradient estimation per region per group per
iteration, which doubles the storage and significantly complicates
the implementation.

**Decision:** For a pin cell with ~3--20 FSRs (annular rings), the
flat-source approximation with fine ray spacing is sufficient.  The
region boundaries are defined by material interfaces (fuel/clad/coolant),
and the flat-source is exact within each material zone.  Linear-source
would only help for coarse radial subdivision within a material zone
(e.g., subdividing the fuel pellet into fewer rings).

**When to revisit:** If the solver is extended to multi-assembly or
whole-core geometry (where FSR counts explode), linear-source becomes
essential for performance.

Why TY Quadrature (Not Gauss-Legendre)
---------------------------------------

The TY quadrature is optimised for the Bickley function
:math:`\text{Ki}_3(\tau)` that appears when the flat-source MOC
angular flux is integrated over polar angle.  In contrast,
Gauss-Legendre (GL) quadrature optimises polynomial integration, which
is not the relevant function form.

For the same number of polar points, TY gives ~2--5× smaller angular
error than GL ([Yamamoto2007]_ Table 3).  TY-3 (3 points) matches GL
with 12--16 points on typical LWR pin-cell problems.

**Connection to CP:** The :ref:`collision probability method
<theory-collision-probability>` uses the same Bickley function
:math:`\text{Ki}_4(\tau)` in its cylindrical kernel.  The CP method
integrates over all rays analytically (via the :math:`y`-quadrature
in :class:`CPMesh`), while MOC traces individual rays and sums
numerically.  Both require accurate approximation of Bickley integrals;
CP handles this with dense Gauss-Legendre quadrature over the impact
parameter, while MOC uses TY quadrature over the polar angle.  The
TY approach is more efficient for MOC because each polar angle
contributes to all tracks simultaneously, whereas CP evaluates the
kernel per-quadrature-point per-region pair.

Why Inverse Wigner-Seitz (Not Direct Square)
----------------------------------------------

An alternative to the inverse Wigner-Seitz approach would be to define
the MOC geometry directly on the square cell (as the old pedagogical
solver did with ``MoCGeometry``, now removed).  This would require a separate
geometry class for the square cell + annuli combination.

**Advantage of the current approach:** The same ``Mesh1D`` created by
:func:`~geometry.factories.pwr_pin_equivalent` is reused by CP, SN,
MC, and now MOC.  The pitch is encoded in the outer Wigner-Seitz radius,
and each solver's augmented geometry recovers whatever it needs.  No
duplication.

**Limitation:** The Wigner-Seitz approximation is exact for the cell
*area* but not for the *shape*.  Corner regions of the square cell
(beyond the Wigner-Seitz circle but within the square) have slightly
different neutronics than the cylindrical approximation suggests.
However, for reflective BC pin-cell calculations, this is standard
practice and the error is negligible for typical LWR geometries.


Gotchas and Subtleties
======================

Void Regions (:math:`\Sigt{} = 0`)
-------------------------------------

If a region has zero total cross section (vacuum), the attenuation
formula :eq:`attenuation` degenerates: :math:`\psi^{\text{out}} =
\psi^{\text{in}}` (no interaction), :math:`\Delta\psi = 0`,
:math:`Q/\Sigt{}` is undefined.  The code guards against this:
segments in void regions are skipped (no contribution to
``delta_phi``), and the scalar flux is set to zero.

In the ORPHEUS verification library, no test material has
:math:`\Sigt{} = 0`, so this code path is exercised only by the
protocol compliance test
``test_moc_verification.py::TestL0ProtocolCompliance``.

Effective Spacing vs Requested Spacing
---------------------------------------

The requested ray spacing :math:`t_s` is adjusted per azimuthal
angle so that rays exactly tile the cell width:
:math:`t_s^{\text{eff}} = (t_{\max} - t_{\min}) / n_{\text{rays}}`
(see :eq:`effective-spacing`).  The effective spacing is always
:math:`\leq t_s` but may differ slightly between angles.

This means the **actual** spatial resolution varies with azimuthal
angle.  Near-horizontal rays (:math:`\varphi \approx 0`) see a
narrower perpendicular extent and thus fewer rays than near-diagonal
rays (:math:`\varphi \approx \pi/4`).  The azimuthal weight
:math:`\omega_m^a = 1/N_\varphi` treats all angles equally, so the
spatial integration is slightly less accurate for some angles.  This
is standard practice and does not affect convergence order.

Flat-Source Breakdown Regime
-----------------------------

The flat-source approximation breaks down when the source varies
significantly within a single FSR.  This occurs when:

- An FSR is optically thick: :math:`\Sigt{} \cdot \ell \gg 1`
  (strong spatial attenuation within the region)
- The flux gradient is steep (near material interfaces)

For a typical LWR pin cell with fuel radius ~0.5 cm and thermal
:math:`\Sigt{} \sim 0.5` cm\ :sup:`-1`, the optical thickness of
the fuel region is :math:`\sim 0.25` (optically thin), so the
flat-source approximation is excellent.  For fast-spectrum systems
with :math:`\Sigt{} \sim 0.1` cm\ :sup:`-1`, the optical thickness
is even smaller.

The flat-source error can be reduced by subdividing regions into
thinner annuli (more cells in the ``Mesh1D``).  The CP solver's
:func:`~geometry.factories.pwr_pin_equivalent` default of 10 fuel
+ 3 clad + 7 coolant sub-cells is adequate for most applications.


Open Improvements
=================

See ``03.Method.Of.Characteristics/IMPROVEMENTS.md`` for the full
tracker.  Key open items:

- **MC-20260406-006**: Sphinx theory chapter (this document) ---
  move from IMPL to DONE
- **MC-20260406-007**: Vectorise the Python transport sweep
  (5-deep loop) for performance
- **MC-20260406-008**: Tighten heterogeneous test tolerances with
  regenerated Richardson references
- **MC-20260406-009**: Cyclic track linking for single-sweep BC
  convergence
- **MC-20260406-010**: CMFD coarse-mesh acceleration


References
==========

.. [Boyd2014] W.R.D. Boyd, S. Shaner, L. Li, B. Forget, K. Smith,
   "The OpenMOC Method of Characteristics Neutral Particle Transport Code,"
   *Annals of Nuclear Energy*, 68, 43--52, 2014.

.. [Yamamoto2007] A. Yamamoto, M. Tabuchi, N. Sugimura, T. Ushio, M. Mori,
   "Derivation of Optimum Polar Angle Quadrature Set for the Method of
   Characteristics Based on Approximation Error for the Bickley Function,"
   *Journal of Nuclear Science and Technology*, 44(2), 129--136, 2007.

.. [Askew1972] J.R. Askew, "A Characteristics Formulation of the Neutron
   Transport Equation in Complicated Geometries," AEEW-M 1108, Winfrith, 1972.

.. [KnottYamamoto2010] D. Knott and A. Yamamoto, "Lattice Physics
   Computations," Chapter 9 in *Handbook of Nuclear Engineering*,
   Springer, 2010.
