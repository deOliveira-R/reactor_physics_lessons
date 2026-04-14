.. _theory-cross-section-data:

==============================================
Cross-Section Data Pipeline
==============================================

.. contents:: Contents
   :local:
   :depth: 3


Key Facts
=========

**Read this before modifying the cross-section pipeline.**

- 421-group microscopic XS from GENDF (GXS) files → HDF5 via ``data/micro_xs/``
- ``Isotope`` dataclass: sig_t, sig_c, sig_f, sig_el, sig_inel, nu, chi (421 groups)
- Sigma-zero iteration: ``data/macro_xs/sigma_zeros.py`` — self-shielding
- ``Mixture`` dataclass: macroscopic XS with ``SigS[l][g_from, g_to]`` convention
- Consistency: :math:`\Sigma_t = \Sigma_c + \Sigma_f + \sum_{g'} \Sigma_{s,g \to g'}`
- ``load_isotope()`` auto-selects HDF5 or fallback .m parser
- Verification uses synthetic XS from ``derivations/_xs_library.py`` (regions A/B/C/D), NOT this pipeline


Overview
========

Every solver in ORPHEUS relies on multi-group microscopic cross sections
for the 12 nuclides in the 421-energy-group JEFF-3.1 library.  This
chapter documents the complete data pipeline from the authoritative IAEA
source files to the internal ``Isotope`` dataclass:

1. **GENDF format** — the IAEA distributes processed nuclear data in
   the GENDF (Groupwise ENDF) format, which uses fixed-width 80-column
   records inherited from punched-card conventions.
2. **Parsing** — the ``gendf.py`` module reads GENDF files directly,
   bypassing the MATLAB CSV intermediary.
3. **Scattering assembly** — elastic, inelastic, and thermal scattering
   matrices are combined with careful treatment of thermal-group
   boundaries.
4. **HDF5 serialisation** — the parsed data is stored in compressed HDF5
   files for fast loading at runtime.
5. **Loading** — ``load_isotope()`` provides a uniform API that
   auto-selects the HDF5 backend when available.

The 12 nuclides in the library are:

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 20

   * - Nuclide
     - GXS File
     - Temperatures (K)
     - Sigma-zeros
   * - H-1
     - ``H_001.GXS``
     - 294, 350, 400, 450, 500, 550, 600, 650
     - 1
   * - O-16
     - ``O_016.GXS``
     - 294, 600, 900, 1200, 1500, 1800
     - 6
   * - B-10
     - ``B_010.GXS``
     - 294, 600, 900, 1200
     - 4
   * - B-11
     - ``B_011.GXS``
     - 294, 600, 900, 1200
     - 4
   * - Na-23
     - ``NA023.GXS``
     - 294, 600, 900, 1200
     - 4
   * - U-235
     - ``U_235.GXS``
     - 294, 600, 900, 1200, 1500, 1800
     - 10
   * - U-238
     - ``U_238.GXS``
     - 294, 600, 900, 1200, 1500, 1800
     - 10
   * - Zr-90
     - ``ZR090.GXS``
     - 294, 600, 900, 1200
     - 4
   * - Zr-91
     - ``ZR091.GXS``
     - 294, 600, 900, 1200
     - 4
   * - Zr-92
     - ``ZR092.GXS``
     - 294, 600, 900, 1200
     - 4
   * - Zr-94
     - ``ZR094.GXS``
     - 294, 600, 900, 1200
     - 4
   * - Zr-96
     - ``ZR096.GXS``
     - 294, 600, 900, 1200
     - 4


The GENDF Format
=================

Source
------

The GENDF (Groupwise ENDF) files are obtained from the IAEA Nuclear Data
Services:

   https://www-nds.iaea.org/ads/adsgendf.html

These are the 421-group JEFF-3.1 processed nuclear data files.  Each file
contains all reaction cross sections and transfer matrices for one
nuclide at multiple temperatures.


Record Layout
--------------

Every line in a GENDF file is exactly 80 characters wide, following the
ENDF-6 format [ENDF102]_ inherited from the punched-card era:

.. code-block:: text

   Columns  1-66: 6 data fields, 11 characters each
   Columns 67-70: MAT number (material identifier)
   Columns 71-72: MF  number (file type)
   Columns 73-75: MT  number (reaction type)
   Columns 76-80: line sequence number

For example, the first data record of H-1 looks like:

.. code-block:: text

    1.001000+3 9.991673-1          0          1         -1          1 125 1451    1
   |----11----|----11----|----11----|----11----|----11----|----11----| MAT|MF|MT |SEQ|


Compact Float Notation
-----------------------

Data fields use a compact Fortran notation where the ``E`` in scientific
notation is omitted.  The exponent sign immediately follows the mantissa:

.. code-block:: text

   1.001000+3  →  1.001000E+3  =  1001.0
   9.991673-1  →  9.991673E-1  =  0.9991673
   2.407191-7  →  2.407191E-7  =  2.407191×10⁻⁷

The parser ``_parse_gendf_field`` in ``gendf.py`` handles this
by inserting ``E`` before any ``+`` or ``-`` sign that follows a digit:

.. code-block:: python

   s = re.sub(r"(\d)([+-])", r"\1E\2", s)
   return float(s)


MF and MT Numbers
------------------

The MF (file) and MT (reaction) numbers identify the type of data in
each section:

**MF=1 — General information:**

.. list-table::
   :header-rows: 1
   :widths: 10 40

   * - MT
     - Content
   * - 451
     - Header: temperatures, sigma-zero base points, energy group
       boundaries (422 values for 421 groups)

**MF=3 — Cross sections** (sigma-zero dependent, one value per group):

.. list-table::
   :header-rows: 1
   :widths: 10 40

   * - MT
     - Reaction
   * - 1
     - Total cross section (does **not** include upscattering)
   * - 18
     - Fission
   * - 102
     - Radiative capture :math:`(n,\gamma)`
   * - 107
     - :math:`(n,\alpha)`
   * - 452
     - Total :math:`\bar{\nu}` (average neutrons per fission)

**MF=6 — Transfer matrices** (group-to-group scattering):

.. list-table::
   :header-rows: 1
   :widths: 10 40

   * - MT
     - Reaction
   * - 2
     - Elastic scattering (sigma-zero dependent)
   * - 16
     - :math:`(n,2n)` reaction
   * - 18
     - Fission spectrum :math:`\chi(g)`
   * - 51–91
     - Discrete inelastic scattering levels
   * - 221
     - Free-gas thermal scattering
   * - 222
     - Thermal scattering for H bound in water (:math:`S(\alpha,\beta)`)


MF=3 Record Structure
----------------------

Each MF=3 section begins with a header record followed by per-group
data records.  The structure for a section with :math:`N_\ell` Legendre
components and :math:`N_{\sigma_0}` sigma-zero values is:

.. code-block:: text

   Record 1 (section header):
     [ZA, AWR, NL, N_sig0, LRFLAG, NG, MAT, MF, MT, 1]

   For each group g = 1, ..., NG:
     Record (group header):
       [TEMP, 0, NL, N_sig0, NW, IG, MAT, MF, MT, line]
     Record(s) (data):
       NW = 2 × NL × N_sig0 words packed 6 per line

The first half of the NW words contains flux weights; the second half
contains the cross-section values organised as:

.. math::

   a[N_\ell N_{\sigma_0} + 1 : N_\ell N_{\sigma_0} + N_{\sigma_0}]
   = \sigma_{x,g}(\sigma_{0,1}), \ldots, \sigma_{x,g}(\sigma_{0,N_{\sigma_0}})

This is the Legendre-0 component for each sigma-zero.  Higher Legendre
components follow in the same block.


MF=6 Record Structure
----------------------

Transfer matrices in MF=6 are stored per source group in a sparse
representation.  For each source group :math:`g`:

.. code-block:: text

   Record (group header):
     [TEMP(?), 0, NG2, IG2LO, NW, IG, MAT, MF, MT, line]
   Record(s) (data):
     NW words packed 6 per line

where:

- ``NG2`` — number of secondary (target) groups with non-zero values
- ``IG2LO`` — 1-based index of the lowest non-zero target group
- ``NW`` — total words to read (includes flux weights)
- ``IG`` — 1-based source group index

The data layout per source group is:

1. **Flux weights**: :math:`N_\ell \times N_{\sigma_0}` values (skipped)
2. **Transfer values**: for each target group from ``IG2LO`` to
   ``IG2LO + NG2 - 2``, and for each sigma-zero and Legendre order:

   .. code-block:: text

      for i_to = IG2LO to IG2LO + NG2 - 2:
          for i_sig0 = 1 to N_sig0:
              for i_lgn = 1 to N_lgn:
                  sigma_s(IG → i_to, Legendre=i_lgn, sig0=i_sig0)


Scattering Matrix Assembly
===========================

The scattering matrix :math:`\Sigma_{\mathrm{s},\ell}^{(\sigma_0)}`
is assembled from three separate GENDF sections.  This is one of the
most delicate parts of the pipeline.


Thermal-Group Boundary
-----------------------

The energy group structure uses a thermal cutoff at group index 95
(corresponding to :math:`E \approx 4` eV).  Below this energy, the
free-atom elastic scattering model breaks down because the target atoms
are bound in a lattice or molecule (thermal motion affects scattering).

The GENDF file provides two models:

- **MT=2** — free-atom elastic scattering (valid above ~4 eV)
- **MT=221** — free-gas thermal scattering (all isotopes except H-1)
- **MT=222** — :math:`S(\alpha,\beta)` thermal scattering for
  H bound in water (H-1 only)


Assembly Algorithm
-------------------

The scattering matrix is built in four stages:

1. **Elastic (MT=2)**: Extract the elastic scattering transfer matrix.
   **Zero out** all entries where the source group :math:`g \le 95`
   (the thermal range), because thermal scattering replaces elastic in
   that range.  Add :math:`10^{-30}` to all values (matching the MATLAB
   convention to avoid exact zeros in sparse matrices).

   .. code-block:: python

      vals[thermal_mask] = 0.0
      vals += 1e-30
      sigS[lgn][sig0] = sparse(ifrom-1, ito-1, vals, NG, NG)

2. **Inelastic (MT=51–91)**: For each discrete inelastic level that
   exists, extract the transfer matrix and **add** it to sigS.
   Inelastic scattering is sigma-zero independent (same values for all
   sigma-zero variants), so the first sigma-zero's data is used for all.

3. **Thermal (MT=221 or MT=222)**: Extract the thermal scattering
   kernel and **add** it to sigS.  This replaces the zeroed elastic
   entries in the thermal range.  Like inelastic, thermal scattering
   is sigma-zero independent.

4. The final scattering matrix structure is a list of lists:
   ``sigS[legendre_order][sig0_index]``, each a ``csr_matrix(NG, NG)``.
   Three Legendre orders (P0, P1, P2) are always stored.

.. important::

   The elastic scattering data for groups :math:`g > 95` (epithermal
   and fast) **does** depend on sigma-zero.  Each sigma-zero variant
   has different elastic values at these groups.  For groups
   :math:`g \le 95`, the elastic is zeroed and replaced by the
   sigma-zero-independent thermal kernel.


Reactions Not Included: :math:`(n,2n)`, :math:`(n,3n)`, :math:`(n,4n)`
-----------------------------------------------------------------------

The GENDF files for heavy isotopes (U-235, U-238, Pu-239, ...) contain
MF=6 scattering entries for three multiplicity-changing reactions that
ORPHEUS **does not extract** into the scattering matrix:

.. csv-table::
   :header: MT, Reaction, Threshold, ENDF name
   :widths: 8, 15, 15, 62

   16, ":math:`(n,2n)`", ~6–8 MeV, neutron-induced two-neutron emission
   17, ":math:`(n,3n)`", ~11–14 MeV, neutron-induced three-neutron emission
   37, ":math:`(n,4n)`", ~20 MeV+, neutron-induced four-neutron emission

The current scattering-matrix assembly loop at
``orpheus/data/micro_xs/gendf.py:281`` only iterates over
MT=51..91 (discrete inelastic levels plus continuum inelastic at
MT=91), matching the original MATLAB ``convertCSVtoM.m`` that ORPHEUS
was ported from.

Why this is (currently) acceptable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two reasons, physical and pragmatic:

1. **Threshold energies are far above the thermal and epithermal
   regime.** A thermal-spectrum reactor has almost no flux above
   6 MeV (the fission spectrum rolls off as :math:`\chi(E) \propto
   e^{-E/a}\sinh\sqrt{bE}`, effectively dead by 10 MeV). The rate
   density for :math:`(n,xn)` at x≥2 is
   :math:`\int_{E_{\mathrm{th}}}^\infty \phi(E)\,\sigma_{(n,xn)}(E)\,dE`
   and both the flux and the cross section are negligible in the
   integration window. Quantitatively, for a PWR-like spectrum, the
   :math:`(n,2n)` rate on U-238 is below :math:`10^{-6}` of the
   absorption rate — below the truncation noise of the 421-group
   multi-group flux itself.

2. **Neutron-multiplication accounting would need to change
   consistently.** Including :math:`(n,xn)` correctly requires more
   than adding an MF=6 block to the scattering matrix. Because these
   reactions change neutron multiplicity, they must be accounted for
   separately from fission in the balance equation (they are *not*
   fission, so they do not carry :math:`\chi` or :math:`\nu`, but
   they *do* produce excess neutrons). ORPHEUS's current balance
   equation assumes a 1-in-1-out scattering model; retrofitting
   :math:`(n,xn)` cleanly means either

   - treating them as sources with a separate multiplicity factor
     (the "``nu_n_xn``" convention in MCNP/Serpent/OpenMC), or
   - folding them into the scattering matrix with an effective
     :math:`\Sigma_{\mathrm{s}}^{(n,xn)}` that scales by the
     multiplicity — which breaks the :math:`\Sigma_t`
     consistency relation :math:`\Sigma_t = \Sigma_c + \Sigma_f +
     \sum_{g'} \Sigma_{s,g\to g'}` unless :math:`\Sigma_c` is
     simultaneously adjusted.

   Doing either correctly is a data-pipeline-wide change, not a
   localized extension.

When this exclusion would matter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three application regimes **do** need :math:`(n,xn)`:

- **Fast reactors** (SFR, LFR, GFR): fission spectrum is harder,
  5–15 % of flux above 1 MeV, :math:`(n,2n)` on U-238 and Pu-240
  contributes measurably to the fast-group balance.
- **Fusion blankets**: 14 MeV D-T source neutrons sit directly in
  the peak of the :math:`(n,2n)`/:math:`(n,3n)` cross section for
  Li, Be, and Pb — these reactions are the *whole point* of a
  breeding blanket.
- **High-energy shielding / accelerator-driven systems**: spallation
  neutron sources produce a significant population above 20 MeV,
  where :math:`(n,4n)` on heavy targets is non-negligible.

None of these are current ORPHEUS use cases. The V&V suite
(:doc:`/verification/index`) exclusively verifies thermal-spectrum
analytical benchmarks; synthetic cross sections in
``orpheus/derivations/_xs_library.py`` do not include an
:math:`(n,xn)` term.

Implementation sketch (deferred)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If / when ORPHEUS expands to fast or fusion applications, the
retrofit would touch:

1. **``orpheus/data/micro_xs/gendf.py``** — extend the scattering
   loop at line 281 to also iterate MT in ``(16, 17, 37)`` and stash
   the extracted block with a ``multiplicity`` attribute (``2``,
   ``3``, ``4`` respectively).

2. **``orpheus/data/micro_xs/isotope.py``** — add a
   ``sig_n_xn: dict[int, np.ndarray]`` field to ``Isotope``.

3. **``orpheus/data/macro_xs/mixture.py``** — decide whether
   :math:`(n,xn)` enters the ``SigS`` matrix (with a multiplicity
   factor baked in, losing the :math:`\Sigma_t` check) or lives as
   an explicit source term in the balance equation.

4. **Every transport solver** (``cp``, ``sn``, ``moc``, ``mc``,
   ``diffusion``) — the balance residual at the cell level must
   account for the multiplicity.

5. **V&V** — add an L2 benchmark against a published fast-reactor
   eigenvalue (e.g., the GODIVA or Jezebel ICSBEP criticals) where
   :math:`(n,xn)` measurably shifts :math:`k_{\mathrm{eff}}`.

Tracked in GitHub issue `#63
<https://github.com/deOliveira-R/ORPHEUS/issues/63>`_ as a
"status: impl, intentional exclusion, documentation complete"
line item.


Total Cross Section
====================

The total cross section :math:`\Sigma_{\mathrm{t},g}(\sigma_0)` is
**computed** from the components rather than read from MF=3 MT=1:

.. math::
   :label: sigT-computed

   \sigma_{\mathrm{t},g}(\sigma_0)
   = \sigma_{\mathrm{c},g}(\sigma_0)
   + \sigma_{\mathrm{f},g}(\sigma_0)
   + \sigma_{\alpha,g}(\sigma_0)
   + \sum_{g'} \sigma_{\mathrm{s},0,g \to g'}(\sigma_0)
   + \sum_{g'} \sigma_{\mathrm{2n},g \to g'}

This approach is used because:

1. MF=3 MT=1 does **not** include upscattering (stated in the MATLAB
   source: *"note that mf=3 mt=1 does not include upscatters"*).
2. Computing from components ensures self-consistency between the total
   and the reaction rates used by the solver.


.. _sigT-consistency:

sigT Consistency Issue (Historical)
-------------------------------------

.. warning::

   The legacy MATLAB ``.m`` data files contain a systematic discrepancy
   in the stored ``sigT`` values.  This section documents the issue for
   future reference.

The MATLAB ``convertCSVtoM.m`` script computes ``sigT`` from
full-precision intermediate variables and writes it with ``%13.6e``
format (6 decimal places in scientific notation).  It independently
truncates all component cross sections (sigC, sigF, sigS) to the same
format.

When the ``.m`` file is loaded and ``sigT`` is **recomputed** from the
stored (truncated) components, the result differs from the stored
``sigT`` by a **constant offset** of 10–30 barns for heavy isotopes:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 15

   * - Isotope
     - .m sigT[0,0]
     - Recomputed
     - Offset
   * - U-235 (294K)
     - 15,523.0
     - 15,504.2
     - 18.8
   * - U-238 (600K)
     - 108.14
     - 77.87
     - 30.27
   * - Zr-90 (600K)
     - (offset)
     - (recomputed)
     - 8.3
   * - H-1 (294K)
     - (matches)
     - (matches)
     - ~0

The offset is **constant across all energy groups and all sigma-zero
rows** for a given isotope/temperature.  Isotopes with :math:`N_{\sigma_0} = 1`
(like H-1) show no discrepancy.

**Root cause**: The full-precision ``sigS`` row sums differ from the
truncated-then-resummed version.  Although each individual truncation
error is :math:`O(10^{-7})` relative, the scattering matrices have
thousands of non-zero entries per row at resonance energies, and the
accumulation of truncation errors is significant.

**Impact on sigma-zero iterations**: The sigma-zero solver interpolates
``sigT(\sigma_0)`` from the tabulated values.  Using the GENDF-computed
``sigT`` instead of the ``.m`` file's stored ``sigT`` shifts the
converged sigma-zeros, which propagates to different interpolated
cross sections and ultimately a ~0.4% shift in the PWR-like mixture's
:math:`\kinf` (1.01771 vs 1.01357).


HDF5 Storage Format
=====================

Each element is stored in a single HDF5 file (e.g., ``U_235.h5``) with
one group per temperature:

.. code-block:: text

   /{temp_K}K/
       @aw          : scalar (atomic weight in amu)
       @temp        : scalar (temperature in K)
       eg           : (NG+1,)    — energy group boundaries (eV)
       sig0         : (N_sig0,)  — sigma-zero base points (barns)
       sigC         : (N_sig0, NG) — radiative capture
       sigL         : (N_sig0, NG) — (n,alpha)
       sigF         : (N_sig0, NG) — fission
       sigT         : (N_sig0, NG) — total
       nubar        : (NG,) — average neutrons per fission
       chi          : (NG,) — fission spectrum (normalised to 1)
       sig2/
           row      : (nnz,) int32  — COO row indices
           col      : (nnz,) int32  — COO column indices
           data     : (nnz,) float64 — COO values
       sigS/
           L{j}_S{k}/          — Legendre order j, sigma-zero k
               row  : (nnz,)
               col  : (nnz,)
               data : (nnz,)

Dense arrays use gzip compression (level 4).  Sparse matrices are stored
as COO triplets to avoid scipy-specific formats.


File Sizes
-----------

.. list-table::
   :header-rows: 1
   :widths: 15 15 15

   * - Element
     - Temperatures
     - HDF5 Size (MB)
   * - H-1
     - 8
     - 12.3
   * - U-235
     - 6
     - 50.0
   * - U-238
     - 6
     - 37.8
   * - O-16
     - 6
     - 10.8
   * - Zr isotopes (×5)
     - 4 each
     - ~11 each


Data Loading API
=================

The ``load_isotope`` function provides a uniform API:

.. code-block:: python

   from data.micro_xs import load_isotope

   iso = load_isotope("U_235", 600)
   # iso.sigC — shape (10, 421), capture XS for 10 sigma-zeros
   # iso.sigS[0][0] — csr_matrix(421, 421), P0 scattering at sig0=0
   # iso.eg — shape (422,), energy group boundaries in eV

The loader reads from the HDF5 files in ``data/micro_xs/{name}.h5``.


Conversion Script
------------------

To regenerate the HDF5 files from the GENDF sources:

.. code-block:: bash

   cd data/micro_xs
   python convert_gxs_to_hdf5.py

This processes all 12 ``.GXS`` files and writes the corresponding
``.h5`` files.  Runtime is approximately 2–3 minutes on a modern
laptop.


Validation
===========

The HDF5 data pipeline was validated by running both homogeneous
reactor cases and comparing against the MATLAB reference:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 10

   * - Case
     - Python :math:`\kinf`
     - MATLAB :math:`\kinf`
     - Match
   * - Aqueous (H₂O + U-235, 294K)
     - 1.03596
     - 1.03596
     - Yes
   * - PWR-like (UO₂ + Zry + H₂O+B, 600K)
     - 1.01357
     - 1.01357
     - Yes

Per-component validation for H-1 at 294K (1 sigma-zero, simplest case):

.. list-table::
   :header-rows: 1
   :widths: 20 20 15

   * - Quantity
     - Max diff (GXS vs .m)
     - Status
   * - ``aw``
     - 0
     - Exact
   * - ``eg``
     - 0
     - Exact
   * - ``sigC``
     - 0
     - Exact
   * - ``sigF``
     - 0
     - Exact
   * - ``nubar``
     - 0
     - Exact
   * - ``chi``
     - 0
     - Exact
   * - ``sigS[0][0]`` (row sums)
     - 0
     - Exact
   * - ``sigS[0][0]`` (nnz)
     - 77,627 = 77,627
     - Exact

Per-component validation for U-235 at 294K (10 sigma-zeros):

.. list-table::
   :header-rows: 1
   :widths: 20 20 15

   * - Quantity
     - Max diff (GXS vs .m)
     - Status
   * - ``sigC``
     - 0
     - Exact
   * - ``sigF``
     - 0
     - Exact
   * - ``nubar``
     - 0
     - Exact
   * - ``sigS[0][0]`` (row sums)
     - :math:`9.6 \times 10^{-7}`
     - Negligible
   * - ``sig2`` (nnz)
     - 6,067 = 6,067
     - Exact


.. seealso::

   - :ref:`theory-homogeneous` — first consumer of the XS pipeline;
     demonstrates the full path from ``load_isotope()`` to :math:`k_\infty`.
   - :ref:`theory-verification` — verification uses :ref:`synthetic cross
     sections <synthetic-xs-library>` (regions A/B/C/D), not this pipeline.
   - :ref:`theory-collision-probability`, :ref:`theory-discrete-ordinates`,
     :ref:`theory-method-of-characteristics`, :ref:`theory-monte-carlo` — all
     transport solvers consume ``Mixture`` objects from this pipeline.


References
===========

.. [ENDF102] M.A. Kellett, O. Bersillon, R.W. Mills, "The JEFF-3.1/-3.1.1
   Radioactive Decay Data and Fission Yields Sub-libraries", OECD/NEA,
   2009.  ENDF-6 format manual: BNL-NCS-44945 (Rev. 2012).
