r"""Macroscopic cross-section field over an SN domain.

Issue #197 PR-TYPED-1 — the typed wrapper that closes the 8 leaked
per-material dispatch loops scattered across :mod:`orpheus.transport.operators.scattering`
and :mod:`orpheus.sn.solver`.  Before this PR the per-material structure
of the cross-section data leaked through every consumer:

* :class:`~orpheus.transport.operators.scattering.ScatteringOperator` carried a
  ``cells_by_mat: dict[int, (ix, iy)]`` constructor parameter and
  iterated it explicitly in :meth:`~ScatteringOperator.add_iso_source`,
  :meth:`~ScatteringOperator.add_n2n_source`,
  :meth:`~ScatteringOperator.foldable_part`,
  :meth:`~ScatteringOperator.residual_part`,
  :meth:`~ScatteringOperator.is_foldable_into_sigma_r`,
  :meth:`~ScatteringOperator.foldable_sigma`, and
  :meth:`~orpheus.transport.operators.transfer.LegendreMomentTransfer.apply`.
* :class:`~orpheus.sn.solver.SNSolver` carried a parallel
  ``_cells_by_mat`` and seven separate XS attributes
  (``sig_t``, ``sig_a``, ``sig_p``, ``chi``, ``sig_s``, ``sig2``,
  ``sig_s0``) all keyed on the same per-material/per-cell topology.
* :meth:`~SNSolver.compute_group_production_rate` ran an explicit
  ``for mid, (ix, iy) in self._cells_by_mat.items()`` to assemble
  the (n,2n) contribution.

The single source of truth is :class:`MaterialXSField`, a typed wrapper
over the per-material :class:`~orpheus.data.macro_xs.mixture.Mixture`
dict plus the spatial distribution carried by the :class:`SNMesh`'s
``mat_map``.  Two access modes:

* **Per-cell views** (``total_cross_section``, ``absorption_cross_section``,
  ``fission_production``, ``emission_spectrum``) — the broadcast
  ``(ng, nx, ny)`` arrays every operator's per-cell math consumes.
  Built lazily on first access via :func:`assemble_cell_xs`; cached.
* **Per-material accessors** (``scattering_legendre``, ``n2n_matrix``,
  ``fission_production_per_material``, ``chi_per_material``,
  ``cells_by_material``) — for operations that genuinely exploit
  per-material structure (group-coupling matmul on small ``(ng, ng)``
  matrices, anisotropic moment scattering).
* **The named typed verbs** (``apply_p0_in_scatter``, ``apply_n2n``,
  ``apply_legendre_scattering_moments``, ``add_n2n_to_group_rate``)
  lived here from PR-TYPED-1 until CS4c step 3 (2026-08-30, O-6/R13):
  they are now the KERNEL FIELDS' array verbs
  (:mod:`orpheus.transport.material_field` — the per-material dispatch
  loop written once over the representation-free channel data, einsums
  verbatim, the (n,2n) multiplicity read from its one home).

Composability framing (the user's three operations):

* **Mixing**: weighted volume-average of two ``MaterialXSField``\ s
  → single homogenised ``MaterialXSField``.  Not yet implemented;
  the homogenisation step lives outside SN today.  Future Wave.
* **Restriction**: per-region subset returns a ``MaterialXSField``
  on the smaller domain.  Future Wave (CP / MoC consumer pattern).
* **Action**: ``mat_xs.fission_production * scalar_flux`` eventually
  reads as the math (after PR-TYPED-2 introduces typed flux fields
  with ``__mul__`` dunders).  This PR keeps bare ``np.ndarray``
  inputs/outputs; the typed-action wiring lands in PR-TYPED-2.

Storage discipline (``coding-elegance`` Pattern 7 — normalise at
definition site): the per-material :class:`Mixture` dict + the
:class:`SNMesh` ARE the source of truth.  The lazy per-cell views
are cached but content-derived from the frozen inputs; they cannot
diverge from the source data.

Capability matrix (which sites collapsed):

==========  ============================================  =================================
Old site    Old pattern                                   New call
==========  ============================================  =================================
scattering  ``for mid in cells_by_mat: ... add ...``      the field verbs (CS4c: ``TransferMaterialField.add_p0_source`` etc.)
scattering  ``for mid in sig_s.items(): off-diag``        :meth:`residual_sig_s` (DSA's read)
scattering  ``for mid in sig_s.items(): diag(...)``       :meth:`foldable_sigma`
solver      ``for mid in _cells_by_mat: sig2 ...``        ``TransferMaterialField.add_to_group_rate`` (CS4c)
==========  ============================================  =================================

Units (the discipline that physics code make units explicit, per
``coding-elegance`` Pattern 3): macroscopic cross sections in
``1/cm`` per energy group; ``chi`` is dimensionless emission spectrum;
the per-material/per-cell expansion is broadcast across cells, units
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

import numpy as np

from orpheus.data.macro_xs.cell_xs import assemble_cell_xs
from orpheus.transport.fields.cross_section_field import CrossSectionField

if TYPE_CHECKING:
    from collections.abc import Mapping

    from orpheus.data.materials import Materials
    from orpheus.data.macro_xs.mixture import Mixture
    from orpheus.numerics.basis.indicator_basis import IndicatorBasis
    from orpheus.numerics.frame import FrameBase
    from orpheus.numerics.measure import DiscreteMeasure
    # ``mesh`` is typed against ``MaterialMesh`` (the method-agnostic
    # mesh+materials carrier): MaterialXSField reads ONLY MaterialMesh data
    # (``materials`` / ``mat_map`` / ``ng`` / ``spatial_shape``) — never the
    # quadrature/trace an ``SNMesh`` adds. This is the #267 ``MaterialXSField``
    # slice: the field is a STANDALONE dataclass (not a ``BulkField`` subclass),
    # so it retypes independently of the full typed-field-hierarchy split (the
    # bulk-data vs quad/trace-dependent ``AngularField``/``AngularBoundaryField`` base
    # split remains the #267 back-half). The MaterialMesh dependency is the
    # method-agnostic carrier: any MaterialMesh (an SNMesh, a DiffusionMesh,
    # a bare carrier) is admitted. The infinite-medium problem builds none
    # since the CS4c coda — its fields are born on ``HomogeneousProblem.space``
    # (until then a mesh-less single-region carrier was admitted here, #276).
    from orpheus.transport.mesh.material_mesh import MaterialMesh


__all__ = ["MaterialXSField"]


@dataclass
class MaterialXSField:
    r"""Macroscopic cross-section field over a material mesh.

    Owns the per-material :class:`~orpheus.data.macro_xs.mixture.Mixture`
    data plus the spatial distribution via the mesh's ``mat_map``.
    Exposes BOTH per-material accessors (for operations that exploit
    per-material structure, e.g. group-coupling matmul on ``(ng, ng)``
    matrices) AND per-cell expanded views (for operations that need
    cell-grid layout, e.g. the streaming/collision algebra).

    The per-cell views (``total_cross_section`` etc.) are CACHED on
    first access via :func:`~orpheus.data.macro_xs.cell_xs.assemble_cell_xs`;
    the per-material side carries the source of truth.

    Parameters
    ----------
    materials : dict[int, Mixture]
        Per-material macroscopic cross sections keyed by integer
        material id.  All materials must agree on ``ng`` (already
        validated by :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh`
        at construction).
    mesh : MaterialMesh
        The mesh+materials carrier — supplies ``materials``, ``mat_map``,
        ``ng``, ``spatial_shape``.  A method-agnostic
        :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` (NOT an
        ``SNMesh``): this field reads no quadrature/trace, so any carrier
        in the hierarchy is admitted (the infinite-medium problem no
        longer builds one — CS4c coda, 2026-09-08).

    Attributes (cached)
    -------------------
    All lazy: populated on first read of the corresponding property
    via :meth:`__post_init__`-free direct construction.  Mutating
    the underlying :attr:`materials` / :attr:`mesh` after construction
    is undefined behaviour — :class:`MaterialXSField` is conceptually
    frozen but uses a non-frozen dataclass to make the lazy caches
    natural.

    Notes
    -----
    Non-frozen dataclass (not :pyfunc:`dataclasses.dataclass(frozen=True)`)
    because the lazy per-cell caches mutate ``self``.  The
    frozen-with-``object.__setattr__`` workaround would obscure the
    storage discipline and complicate testing.  The "frozenness" we
    want is content-immutability of :attr:`materials` and :attr:`mesh`,
    which Python doesn't enforce structurally but which every consumer
    of this class respects by convention.
    """

    materials: "Materials | Mapping[int, Mixture]"
    mesh: "MaterialMesh"

    # Lazy per-cell views — populated on first access.
    _sig_t_cell: np.ndarray | None = field(default=None, init=False, repr=False)
    _sig_a_cell: np.ndarray | None = field(default=None, init=False, repr=False)
    _sig_p_cell: np.ndarray | None = field(default=None, init=False, repr=False)
    _chi_cell: np.ndarray | None = field(default=None, init=False, repr=False)
    _diffusion_cell: np.ndarray | None = field(default=None, init=False, repr=False)
    # Variadic index tuple: ``np.where`` yields one index array PER MESH
    # AXIS, so the arity is the mesh ndim (a 1-tuple on a 1-D mesh, a
    # 2-tuple on 2-D) — declaring a fixed 2-tuple would lie for 1-D.
    _cells_by_mat: dict[int, tuple[np.ndarray, ...]] | None = field(
        default=None, init=False, repr=False,
    )
    # Cached dense (n,2n) matrices to avoid repeated ``.todense()``
    # in the (n,2n) hot path.
    _n2n_dense: dict[int, np.ndarray] | None = field(
        default=None, init=False, repr=False,
    )
    # Cached per-material Legendre scattering lists (already dense).
    _sig_s_dense: dict[int, list[np.ndarray]] | None = field(
        default=None, init=False, repr=False,
    )

    # ── Construction helpers ──────────────────────────────────────────

    @classmethod
    def from_mesh(cls, mesh: "MaterialMesh") -> "MaterialXSField":
        """Build the XS field directly from the mesh's authoritative materials.

        Standard constructor — the materials dict already lives on the
        mesh (Issue #197 PR-TYPED-0).  Reads only
        :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` data
        (``mesh.materials`` / ``mesh.mat_map`` / ``mesh.ng`` /
        ``mesh.spatial_shape``), so it accepts any
        :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` — the
        meshed SN :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` (a
        ``MaterialMesh`` subclass), a ``DiffusionMesh``, or a bare carrier
        (until the CS4c coda also the retired mesh-less single-region
        carrier of the 0-D homogeneous medium, #276).  The parameter is
        honestly typed ``MaterialMesh`` (#267 slice).
        """
        return cls(materials=mesh.materials, mesh=mesh)

    # ── Homogenisation: project the whole field through coarse frames ──

    def project_through(
        self, sigma_frame: "FrameBase", emission_frame: "FrameBase", /,
    ) -> dict[int, "Mixture"]:
        r"""Homogenise the whole cross-section field through two coarse frames.

        Collapse every per-fine-cell channel onto the coarse cells of the frames'
        shared trial (cell-indicator) basis, returning one effective
        :class:`~orpheus.data.macro_xs.mixture.Mixture` per coarse cell. The field
        owns the **channel → weighting taxonomy** and routes accordingly:

        * the **rate-bearing** channels — :math:`\Sigma_t,\Sigma_c,\Sigma_L,
          \Sigma_f,\nu\Sigma_f` (vectors) and :math:`\Sigma_{s,\ell},\Sigma_{2n}`
          (``[g_from, g_to]`` matrices) — collapse through ``sigma_frame``, whose
          flux-weighted test basis makes :meth:`~orpheus.numerics.frame.FrameBase.project`
          the reaction-rate-preserving average :math:`\Sigma_R = \int_R\varphi\Sigma\,
          \mathrm{d}V/\int_R\varphi\,\mathrm{d}V` (matrices weight by the **source**
          group, the leading axis the test weight aligns to);
        * the **emission spectrum** :math:`\chi` collapses through ``emission_frame``,
          whose production-weighted test (:math:`p=\sum_g\nu\Sigma_{f,g}\varphi_g`)
          gives the production-weighted convex average :math:`\chi_R = \int_R p\chi\,
          \mathrm{d}V/\int_R p\,\mathrm{d}V` (a convex combination of simplices, hence
          a simplex).

        Two frames because the two collapses preserve two *different* conserved
        functionals (reaction rate vs emission rate) — the campaign's "one frame
        carries one test weighting" ruling. Both frames share the trial basis +
        geometric measure; the caller (which owns the flux) builds their test
        weightings.

        Parameters
        ----------
        sigma_frame : FrameBase
            The flux-weighted homogenisation frame for the rate-bearing channels.
        emission_frame : FrameBase
            The production-weighted frame for :math:`\chi`.

        Returns
        -------
        dict[int, Mixture]
            One effective :class:`Mixture` per coarse cell, keyed by coarse-cell
            index ``0 .. n_coarse-1``. A group with zero region weight collapses to a
            zero effective cross section there (the frame's Moore–Penrose Gram
            pseudo-inverse — no reaction rate to preserve).
        """
        # Rate-bearing channels → the flux-weighted frame. ``project`` IS the
        # rate-preserving collapse G⁻¹M (the per-channel inline gather/collapse the
        # method body used to carry now lives here, projected as ONE field).
        n_legendre = self._n_legendre("SigS")
        sig_t = sigma_frame.project(self._gather_vector("SigT"))
        sig_c = sigma_frame.project(self._gather_vector("SigC"))
        sig_l = sigma_frame.project(self._gather_vector("SigL"))
        sig_f = sigma_frame.project(self._gather_vector("SigF"))
        sig_p = sigma_frame.project(self._gather_vector("SigP"))
        sig_s = [sigma_frame.project(self._gather_stack("SigS", l)) for l in range(n_legendre)]
        sig2 = [sigma_frame.project(self._gather_stack("Sig2", l)) for l in range(self._n_legendre("Sig2"))]

        # χ → the production-weighted frame (a different conserved rate).
        chi = emission_frame.project(self._gather_vector("chi"))

        return self._assemble_mixtures(
            sig_t=sig_t, sig_c=sig_c, sig_l=sig_l, sig_f=sig_f, sig_p=sig_p,
            sig_s=sig_s, sig2=sig2, chi=chi,
        )

    def project_through_bilinear(
        self,
        trial: "IndicatorBasis",
        measure: "DiscreteMeasure",
        *,
        phi: np.ndarray,
        phi_star: np.ndarray,
        rho: np.ndarray,
    ) -> dict[int, "Mixture"]:
        r"""The eigenvalue-consistent (adjoint-weighted, P6 #281) homogenisation.

        The bilinear sibling of :meth:`project_through`: collapse every channel
        with the rule that zeroes its first-order XS-collapse worth, so the
        coarse :math:`k` is first-order stationary in the flux shapes.  The
        rules are the theorems of the algebra of record
        (:mod:`orpheus.derivations.common.homogenization`); the field owns the
        **channel → morphism taxonomy** (five morphisms here vs the forward
        method's two):

        * :math:`\Sigma_c,\Sigma_L,\Sigma_f` (response vectors) — the **pair
          frame** (T1): test weight :math:`\varphi^*\!\odot\varphi`;
        * :math:`\Sigma_t` (the collision channel of the pencil) — the
          **collision frame** (T1b): test weight :math:`\rho_{i,g} = \sum_n
          w_n\psi^*_{i,g,n}\psi_{i,g,n}` (the exact ANGULAR pairing, of which
          :math:`\varphi^*\varphi` is the isotropic/P0 truncation);
        * :math:`\Sigma_{s,\ell},\Sigma_{2n}` (matrices) — the **per-pair
          collapse** (T2): weight :math:`\varphi^*_{g}\,\varphi_{g'}` (sink
          adjoint × source flux) per :math:`(g',g)` entry;
        * :math:`\nu\Sigma_f` — the **mixed-fold rule** (T3): numerator folded
          by the fine emission importance :math:`\iota_i =
          \sum_g\varphi^*_{i,g}\chi_{i,g}`, denominator by the collapsed
          :math:`\tilde\iota_i = \sum_g\varphi^*_{i,g}\chi_{R,g}` — exact
          TOTAL fission worth for any simplex :math:`\chi_R`;
        * :math:`\chi` — the **canonical convex average** (T3): weights
          :math:`\iota_i\,p_i` (adjoint-weighted emission; the geometric
          :math:`V_i` rides the measure), a simplex by construction.

        Every rule degenerates to the forward (:meth:`project_through`)
        weighting at flat :math:`\varphi^*` / isotropic shapes — proved in the
        derivation module, pinned by its suite.

        .. warning:: **The worth-exact collapse breaks the total-XS balance
           identity** (T4 — worth-exactness and
           :math:`\Sigma_t = \Sigma_c+\Sigma_L+\Sigma_f+\text{rowsums}` are
           mutually exclusive for :math:`\varphi^*\neq` const; the classical
           reactivity-vs-rates property of bilinear-weighted constants).  Do
           NOT ``assert_balanced`` on the returned Mixtures; the imbalance is
           the documented price of first-order :math:`k`-stationarity, ruled
           at the P6 open.

        Parameters
        ----------
        trial : IndicatorBasis
            The coarse cell-indicator trial basis (the coarse mesh yields it).
        measure : DiscreteMeasure
            The fine geometric volume measure ``dV`` (nodes = fine cells).
        phi, phi_star : np.ndarray
            The forward flux and the importance, ``(n_fine, ng)`` in the
            measure's "ij" flat-cell order (the caller — the Solution, which
            owns the fluxes — reshapes).
        rho : np.ndarray
            The angular pair weight :math:`\sum_n w_n\psi^*\psi`,
            ``(n_fine, ng)`` in the same order.

        Returns
        -------
        dict[int, Mixture]
            One effective :class:`Mixture` per coarse cell.  Empty / zero-
            weight regions collapse to zero entries (the same Moore–Penrose
            convention as the frames' pseudo-inverse Gram).
        """
        from orpheus.numerics.basis import WeightedIndicatorBasis
        from orpheus.numerics.frame import PetrovGalerkinFrame

        # Frame-shaped morphisms — still genuine Petrov-Galerkin frames (the
        # discipline type is load-bearing; the Mode-11 sentinel captures these
        # constructions and the weights they carry).
        pair_frame = PetrovGalerkinFrame(
            trial, measure, WeightedIndicatorBasis(trial, phi_star * phi),
        )
        collision_frame = PetrovGalerkinFrame(
            trial, measure, WeightedIndicatorBasis(trial, rho),
        )

        sig_t = collision_frame.project(self._gather_vector("SigT"))
        sig_c = pair_frame.project(self._gather_vector("SigC"))
        sig_l = pair_frame.project(self._gather_vector("SigL"))
        sig_f = pair_frame.project(self._gather_vector("SigF"))

        # The fission dyad (T3). ι and p are per-cell physics folds the field
        # owns (they read the field's own channels); the caller owns only the
        # fluxes.
        nsf = self._gather_vector("SigP")                    # (n_fine, ng)
        chi_fine = self._gather_vector("chi")                # (n_fine, ng)
        iota = (phi_star * chi_fine).sum(axis=1)             # (n_fine,)
        p = (nsf * phi).sum(axis=1)                          # (n_fine,)
        emission_frame = PetrovGalerkinFrame(
            trial, measure, WeightedIndicatorBasis(trial, iota * p),
        )
        chi = emission_frame.project(chi_fine)               # (n_coarse, ng)

        # The explicit (non-frame-shaped) morphisms share the frames' own
        # tabulation surfaces — one membership table, one volume weight.
        membership = np.asarray(
            trial.evaluate(measure.nodes), dtype=float,
        ).T                                                  # (n_coarse, n_fine)
        volumes = np.asarray(measure.weights, dtype=float)   # (n_fine,)
        region_of_fine = np.argmax(membership, axis=0)

        # νΣf mixed-fold (T3): ι-folded numerator / ι̃-folded denominator.
        iota_tilde = (phi_star * chi[region_of_fine]).sum(axis=1)
        num_p = np.einsum("Rn,n,ng->Rg", membership, volumes * iota, nsf * phi)
        den_p = np.einsum("Rn,n,ng->Rg", membership, volumes * iota_tilde, phi)
        sig_p = np.divide(num_p, den_p, out=np.zeros_like(num_p), where=den_p != 0.0)

        # Matrix channels (T2): the per-pair sink×source collapse. The source
        # flux folds into the FIELD (the scattering emission it drives); the
        # sink adjoint is the test side; the pair denominator is the
        # generalized Gram, Moore–Penrose-zeroed like the frames'.
        def _per_pair(channel: np.ndarray) -> np.ndarray:
            num = np.einsum(
                "Rn,n,nf,nt,nft->Rft", membership, volumes, phi, phi_star, channel,
            )
            den = np.einsum("Rn,n,nf,nt->Rft", membership, volumes, phi, phi_star)
            return np.divide(num, den, out=np.zeros_like(num), where=den != 0.0)

        n_legendre = self._n_legendre("SigS")
        sig_s = [_per_pair(self._gather_stack("SigS", l)) for l in range(n_legendre)]
        sig2 = [_per_pair(self._gather_stack("Sig2", l)) for l in range(self._n_legendre("Sig2"))]

        return self._assemble_mixtures(
            sig_t=sig_t, sig_c=sig_c, sig_l=sig_l, sig_f=sig_f, sig_p=sig_p,
            sig_s=sig_s, sig2=sig2, chi=chi,
        )

    # ── The shared gather / assembly surface (both coarsening verbs) ──

    def _mat_of_fine(self) -> np.ndarray:
        """Material id per fine cell, ``(n_fine,)`` in "ij" flat order."""
        return np.asarray(self.mesh.mat_map, dtype=int).ravel()

    def _gather_vector(self, attr: str) -> np.ndarray:
        """Per-fine-cell view of a ``(ng,)`` Mixture channel — ``(n_fine, ng)``."""
        materials = self.materials
        return np.array([getattr(materials[m], attr) for m in self._mat_of_fine()])

    def _n_legendre(self, channel: Literal["SigS", "Sig2"]) -> int:
        """The widest Legendre stack of ``channel`` over the materials."""
        return max(len(getattr(self.materials[m], channel)) for m in self.materials)

    def _gather_stack(self, channel: Literal["SigS", "Sig2"], order: int) -> np.ndarray:
        """Per-fine-cell dense ``Σ_{·,ℓ}`` of ``channel`` — ``(n_fine, ng, ng)``.

        An order a material's stack lacks is exactly zero (the evaluation's
        own statement) — the same padding ``Mixture._macroscopic_stack`` applies
        when it sums isotopes and ``TransferKernel.at_order`` applies at the
        binding. Three spellings of one law on three tiers (isotope sum,
        sigma-zero projection, kernel); each tier's datum is its own. One
        gather for both channels (the ``_gather_vector`` idiom).
        """
        materials, ng = self.materials, self.ng
        return np.array([
            np.asarray(getattr(materials[m], channel)[order].todense())
            if order < len(getattr(materials[m], channel)) else np.zeros((ng, ng))
            for m in self._mat_of_fine()
        ])

    def _assemble_mixtures(
        self, *, sig_t, sig_c, sig_l, sig_f, sig_p, sig_s, sig2, chi,
    ) -> dict[int, "Mixture"]:
        """Assemble the collapsed channels — one :class:`Mixture` per coarse cell.

        Routes through the shared :meth:`Mixture.from_dense_channels` assembler
        (the csr wrapping + eg threading lives once, in data — Cardinal Rule 2;
        the energy verb ``Mixture.condense`` calls the SAME assembler).
        """
        from orpheus.data.macro_xs.mixture import Mixture

        eg = next(iter(self.materials.values())).eg
        n_legendre = len(sig_s)
        n_coarse = sig_t.shape[0]
        return {
            region: Mixture.from_dense_channels(
                SigC=sig_c[region], SigL=sig_l[region], SigF=sig_f[region],
                SigP=sig_p[region], SigT=sig_t[region],
                SigS=[sig_s[l][region] for l in range(n_legendre)],
                Sig2=[sig2[l][region] for l in range(len(sig2))],
                chi=chi[region], eg=eg,
            )
            for region in range(n_coarse)
        }

    # ── Per-cell views (lazy, cached) ─────────────────────────────────

    @property
    def total_cross_section(self) -> np.ndarray:
        r""":math:`\sigma_t` per-cell view, shape ``(ng, nx, ny)``.

        Units: ``1/cm`` per energy group.  Built lazily on first
        access via :func:`~orpheus.data.macro_xs.cell_xs.assemble_cell_xs`
        with a ``.T.reshape(ng, nx, ny)`` to the principled layout
        (Issue #196 PR-INDEX-3).  Cached.
        """
        if self._sig_t_cell is None:
            self._ensure_cell_views()
        return self._sig_t_cell  # type: ignore[return-value]

    @property
    def absorption_cross_section(self) -> np.ndarray:
        r""":math:`\sigma_a` per-cell view, shape ``(ng, nx, ny)``."""
        if self._sig_a_cell is None:
            self._ensure_cell_views()
        return self._sig_a_cell  # type: ignore[return-value]

    @property
    def fission_production(self) -> np.ndarray:
        r""":math:`\nu \Sigma_f` per-cell view, shape ``(ng, nx, ny)``.

        Production cross-section ``νΣ_f`` — the rate at which fission
        emits new neutrons per absorption per group.  Units: ``1/cm``.
        """
        if self._sig_p_cell is None:
            self._ensure_cell_views()
        return self._sig_p_cell  # type: ignore[return-value]

    @property
    def emission_spectrum(self) -> np.ndarray:
        r""":math:`\chi` per-cell view, shape ``(ng, nx, ny)``.

        Fission emission spectrum (dimensionless; ``Σ_g χ_g = 1`` per
        fissile material).  Broadcast across cells.
        """
        if self._chi_cell is None:
            self._ensure_cell_views()
        return self._chi_cell  # type: ignore[return-value]

    @property
    def diffusion_coefficient(self) -> np.ndarray:
        r""":math:`D = 1/(3\Sigma_{\rm tr})` per-cell view, shape ``(ng, *spatial)``.

        The per-cell gather of the #290 P1 data seam
        (:attr:`~orpheus.data.macro_xs.mixture.Mixture.diffusion_coefficient`
        — the outflow transport approximation
        :math:`\Sigma_{\rm tr} = \Sigma_t - \sum_{g'}\Sigma_{s,1}(g\to g')`,
        with the P0-only limit :math:`\Sigma_{\rm tr} = \Sigma_t`
        EXACTLY). Units: ``cm``. Consumed by the diffusion
        :class:`~orpheus.diffusion.operators.LeakageOperator` (#290 P4)
        — the single per-cell read path for D, so the seam's arithmetic
        lives once on :class:`Mixture` and the spatial distribution once
        here (Pattern 2). Lazy + cached like the sibling views.
        """
        if self._diffusion_cell is None:
            ng = self.mesh.ng
            spatial = self.mesh.spatial_shape
            per_material = {
                mid: np.asarray(mix.diffusion_coefficient, dtype=float)
                for mid, mix in self.materials.items()
            }
            flat_ids = np.asarray(self.mesh.mat_map, dtype=int).ravel()
            # (N_cells, ng) gather → the principled (ng, *spatial) layout
            # (the ``_ensure_cell_views`` .T.reshape convention).
            gathered = np.array([per_material[int(m)] for m in flat_ids])
            self._diffusion_cell = gathered.T.reshape(ng, *spatial)
        return self._diffusion_cell

    def _ensure_cell_views(self) -> None:
        """Populate the four per-cell views via :func:`assemble_cell_xs`.

        Single producer of the principled-layout per-cell arrays.
        Run-once: subsequent accesses hit the cache.  This is the
        canonical bridge between the per-material :class:`Mixture`
        flat representation and the principled per-cell ``(ng, nx, ny)``
        layout.
        """
        xs = assemble_cell_xs(self.materials, self.mesh.mat_map)
        ng = self.mesh.ng
        spatial = self.mesh.spatial_shape
        # .T.reshape: producer emits (N_cells, ng); flip to (ng, N_cells)
        # then split N_cells back into (*spatial) — the principled
        # (ng, *spatial) layout (rank == ndim; no phantom ny=1 on 1-D).
        self._sig_t_cell = xs.sig_t.T.reshape(ng, *spatial)
        self._sig_a_cell = xs.sig_a.T.reshape(ng, *spatial)
        self._sig_p_cell = xs.sig_p.T.reshape(ng, *spatial)
        self._chi_cell = xs.chi.T.reshape(ng, *spatial)

    # ── Typed per-cell views (the CrossSectionField promotion, #257 S2) ──
    #
    # The field side of the operator promotion C = M[σ_t] (#257 §5.7): each
    # macroscopic cross section, wrapped as the typed
    # :class:`~orpheus.transport.fields.cross_section_field.CrossSectionField`
    # (units 1/cm; physical cross sections, so cone-valued — but the cone is a
    # property, NOT a per-field invariant: a signed σ−σ′ is still a valid
    # CrossSectionField). These wrap the SAME cached ndarray as the raw views
    # above (``.values is`` the raw array — bit-identical, zero copy), so they
    # are a pure-additive typed lens, NOT a second representation. The raw
    # ndarray views remain the live consumer path; the typed accessors are the
    # migration target the S3 ``MultiplicationOperator`` reads from.
    #
    # All three σ are typed for SYMMETRY even though only σ_t (S3 collision) and
    # νΣ_f (S6 fission) have near-term consumers: the wrap is one proven pattern
    # (its rule-of-two is met in-diff), and a two-typed-one-raw surface would
    # invite a future σ_a consumer to open-code the wrap inline (a single-source
    # violation). Do NOT delete ``absorption_cross_section_field`` as "unused".

    @property
    def total_cross_section_field(self) -> CrossSectionField:
        r""":math:`\Sigma_t` as a typed :class:`CrossSectionField` (1/cm)."""
        return CrossSectionField(values=self.total_cross_section, space=self.mesh.bulk_space)

    @property
    def absorption_cross_section_field(self) -> CrossSectionField:
        r""":math:`\Sigma_a` as a typed :class:`CrossSectionField` (1/cm)."""
        return CrossSectionField(values=self.absorption_cross_section, space=self.mesh.bulk_space)

    @property
    def fission_production_field(self) -> CrossSectionField:
        r""":math:`\nu\Sigma_f` as a typed :class:`CrossSectionField` (1/cm)."""
        return CrossSectionField(values=self.fission_production, space=self.mesh.bulk_space)

    # ── Per-material accessors (source of truth) ─────────────────────

    @property
    def cells_by_material(self) -> dict[int, tuple[np.ndarray, ...]]:
        r"""Per-material cell-index arrays — cached.

        For each material id, returns the ``np.where`` index tuple such
        that ``mat_map[indices] == mid`` everywhere — one index array
        PER MESH AXIS (``(ix,)`` on a 1-D mesh, ``(ix, iy)`` on 2-D).
        This is the single index map the formerly-leaked per-material
        dispatch loops keyed on.  Most consumers should NOT use this
        directly — call one of the typed verbs
        (the kernel fields' ``add_p0_source`` / ``moment_source``
        family since CS4c step 3) that
        encapsulates the loop.

        Returns
        -------
        dict[int, tuple[np.ndarray, ...]]
            ``{mid: indices}`` for every material id in
            :attr:`materials`; the tuple arity is the mesh ndim.
        """
        # CS4c 3a: the partition moved to its native place — the mesh owns
        # its layout machinery; this facade read-through rides until F-1.
        return self.mesh.cells_by_material

    def sig_s_legendre(self, material_id: int) -> list[np.ndarray]:
        r"""Per-material list of dense Legendre scattering matrices.

        Returns ``[Σ_{s,0}, Σ_{s,1}, ..., Σ_{s,L}]`` for the requested
        material, each entry a dense ``(ng, ng)`` array indexed
        ``[g_from, g_to]``.  Materially equivalent to
        ``[mix.SigS[l].todense() for l in ...]`` but cached — and
        READ-ONLY (CS4a-R EE-4): the list entries are the shared cache
        arrays themselves, frozen at build so no consumer can mutate the
        loss matrix through this surface; ``.copy()`` first if you need
        a writable result.
        """
        if self._sig_s_dense is None:
            self._build_dense_caches()
        return self._sig_s_dense[material_id]  # type: ignore[index]

    def n2n_matrix(self, material_id: int) -> np.ndarray:
        r""":math:`\Sigma_{2n}` dense ``(ng, ng)`` P0 matrix for one material.

        Cached dense expansion of ``Mixture.Sig2[0]`` (sparse upstream) — the
        reaction matrix the P0 verbs and the fold test read; the stack's
        higher orders reach the operator layer at #426 step 2.
        """
        if self._n2n_dense is None:
            self._build_dense_caches()
        return self._n2n_dense[material_id]  # type: ignore[index]

    def fission_production_per_material(self, material_id: int) -> np.ndarray:
        r""":math:`\nu \Sigma_f` ``(ng,)`` vector for one material."""
        return self.materials[material_id].SigP

    def chi_per_material(self, material_id: int) -> np.ndarray:
        r""":math:`\chi` ``(ng,)`` vector for one material."""
        return self.materials[material_id].chi

    def _build_dense_caches(self) -> None:
        """Populate dense Legendre scattering + (n,2n) caches.

        Called lazily by :meth:`sig_s_legendre` / :meth:`n2n_matrix`
        on first access.  Caches the dense ``(ng, ng)`` matrices so
        the foldable family (and, until CS4c step 3, the apply_* arms)
        avoid repeated sparse-to-dense conversion in the hot path.
        """
        sig_s_dense: dict[int, list[np.ndarray]] = {}
        n2n_dense: dict[int, np.ndarray] = {}

        def _frozen(dense: np.ndarray) -> np.ndarray:
            # CS4a-R EE-4: the caches are SHARED live views — every
            # consumer of sig_s_legendre / n2n_matrix receives the cache
            # object itself, so a caller mutation used to reach the loss
            # matrix ([M] 2026-08-21: +999 through sig_s_legendre moved
            # the retired apply arms). Freeze at the producer; the two
            # consumers needing mutable results already copy first.
            dense.setflags(write=False)
            return dense
        for mid, mix in self.materials.items():
            sig_s_dense[mid] = [
                _frozen(np.asarray(s.todense())) for s in mix.SigS
            ]
            n2n_dense[mid] = _frozen(np.asarray(mix.Sig2[0].todense()))  # the P0 REACTION block (removal / the fold predicate)
        self._sig_s_dense = sig_s_dense
        self._n2n_dense = n2n_dense

    # ── Convenience metadata ──────────────────────────────────────────

    @property
    def ng(self) -> int:
        """Energy group count — read-through from :attr:`mesh`."""
        return self.mesh.ng

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Per-axis cell counts — read-through from :attr:`mesh`.

        C5.2 (#225): replaces the retired ``nx``/``ny`` pair — ONE
        rank-generic metadata read-through instead of two d-bound ones
        (``ny`` lied at d=1 and silently truncated at d≥3).
        """
        return self.mesh.spatial_shape

    # ── The typed apply_* verbs LIVED here until CS4c step 3c ─────────
    # (O-6/R13): the eight per-material dispatch arms + add_n2n_to_group_rate
    # moved to the kernel fields (orpheus/transport/material_field.py — the
    # per-material loop written once, einsums verbatim, the (n,2n)
    # multiplicity read from the kernel's own datum,
    # TransferKernel.multiplicity, since #426 step 2). The dense per-material
    # accessors below survive for the foldable family's DSA consumer until
    # F-1 dissolves the facade.

    # ── Foldable / residual split (Phase G four-operator algebra) ────
    #
    # These accessors encapsulate the per-material foldable/residual
    # split of the P0 scattering data. Their one live consumer is the
    # DSA coefficient assembly (``orpheus/sn/acceleration/dsa.py``) —
    # ``ScatteringOperator``'s sibling constructors read the bound
    # kernel field directly since CS4c step 3 (the F-1 facade
    # dissolution re-homes these onto the Materials tier).

    def residual_sig_s(self) -> dict[int, list[np.ndarray]]:
        r"""Per-material residual Legendre scattering lists.

        For each material ``mid``, returns
        ``[off_diagonal_P0, Σ_{s,1}, ..., Σ_{s,L}]`` — the cross-group
        P0 (off-diagonal) plus every :math:`\ell \ge 1` block verbatim.
        Consumed by the DSA coefficient assembly
        (``orpheus/sn/acceleration/dsa.py``) for the residual
        transport correction. (``ScatteringOperator.residual_part``
        read this until CS4c step 3; it now derives its sibling
        kernels from the bound field directly.)

        Returns
        -------
        dict[int, list[np.ndarray]]
            ``{mid: [cross_group_P0, Σ_{s,1}, ...]}``.
        """
        out: dict[int, list[np.ndarray]] = {}
        for mid in self.materials:
            mats = self.sig_s_legendre(mid)
            p0 = mats[0]
            cross_group = p0 - np.diag(np.diag(p0))
            out[mid] = [cross_group, *mats[1:]]
        return out

    def foldable_sigma(self) -> dict[int, np.ndarray]:
        r"""Per-material foldable cross-section :math:`(\sigma_{s,0}^{g\to g})_g`.

        For each material ``mid``, returns the ``(ng,)`` array
        ``np.diag(sig_s[mid][0])`` — the per-group within-group
        self-scatter cross-section that the cell-balance denominator
        absorbs into :math:`\sigma_r = \sigma_t - \Sigma_{s,0}^{g\to g}`.

        Returns
        -------
        dict[int, np.ndarray]
            ``{mid: (ng,) array}``.  Each entry is a fresh copy.
        """
        return {
            mid: np.diag(self.sig_s_legendre(mid)[0]).copy()
            for mid in self.materials
        }
