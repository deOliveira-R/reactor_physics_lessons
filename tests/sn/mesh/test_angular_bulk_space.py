r"""S1 gates for the carrier's angular-bulk space mint (campaign 1 CS4b).

G1.1–G1.5 of the CS4b verification plan
(``scratch/cs4b_verification_plan.md`` §11, step S1) plus the scheme-side
``moment_axis`` admission pair. The step is provably behaviour-neutral —
nothing consumes :attr:`SNMesh.angular_bulk_space` yet — so every gate here
is either a RECORD of the mint's content, a LAW comparing it against the
SHIPPED dense composite interior (the §6c witness that exists today), or an
ADMISSION with both legs (vv #11).

Conventions gated (CS4b crosswalk B1/B5, ``.claude/plans/cs4b_crosswalk.md``):

* the axis order is ``(angular, energy, spatial)``, matching the bulk tensor
  ``(N, ng, *spatial)``;
* the energy and spatial arms are ``bulk_space``'s axes REUSED VERBATIM
  (object identity — the energy-arm rule is spelled once);
* the Gram of the axis product equals the hand-built dense ``V·w_n``
  oracle on both the DD and the LD arm (LD composes the scheme-owned MODAL
  ``moment_axis`` carrying ``moment_mass_diagonal``), and since S2b the
  composite's interior IS the cached mint (identity on DD, ``==`` on LD);
* the derived space NAME is never pinned (R4: CS2's typed axis subclasses
  change the digest — every assertion here is per-axis content or relative
  ``is``/``==``);
* cone predicates: nodal bulk families answer ``True``, the harmonic-moment
  and trace families ``None``, the LD moment-tailed product ``False`` (the
  Q6-ratified routing base for ``cone_violations``; the exact vertex test
  is #400).

Fixture: the verification plan's §2 configuration — a NON-uniform 5-cell
slab (uniform volumes would collapse ``V`` to a scalar and blind half the
metric claims), ``gauss_legendre(4)``, ``ng = 2``, vacuum/vacuum.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from orpheus.geometry import BC, CoordSystem, Mesh1D
from orpheus.transport.mesh.axis import AxisCoord, AxisMesh, RadialAxisMesh
from orpheus.numerics.axis import BasisKind, EnergyAxis
from orpheus.numerics.quadrature import Quadrature
from orpheus.numerics.space import FunctionSpace
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.transport.fields.harmonic_moment_flux import HarmonicMomentFlux
from orpheus.transport.spatial import LinearDiscontinuous
from orpheus.transport.spatial.diamond import DiamondDifference
from tests.sn._test_helpers import placeholder_materials

pytestmark = pytest.mark.foundation

#: NON-uniform edges — ``V = [0.2, 0.3, 0.4, 0.7, 1.4]``, a genuine vector.
_EDGES = np.array([0.0, 0.2, 0.5, 0.9, 1.6, 3.0])
_NG = 2


def _slab(*, scheme=None, ng: int = _NG) -> SNMesh:
    mesh = Mesh1D(
        edges=_EDGES,
        mat_ids=np.zeros(_EDGES.size - 1, dtype=int),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("vacuum"),
        bc_right=BC("vacuum"),
    )
    kwargs = {} if scheme is None else {"scheme": scheme}
    return SNMesh(
        mesh, Quadrature.gauss_legendre(4), placeholder_materials(ng=ng), **kwargs
    )


class TestG11AxisTuple:
    """G1.1 — the axis tuple IS (angular w_n, energy, spatial V). RECORD."""

    def test_axes_are_angular_energy_spatial_with_the_carrier_measures(self):
        sn = _slab()
        space = sn.angular_bulk_space
        assert space.axes is not None and len(space.axes) == 3
        angular, energy, spatial = space.axes

        assert angular.label == "angular"
        assert angular.shape == (sn.quad.N,)
        assert angular.kind is BasisKind.NODAL
        assert angular.weights is not None
        assert np.array_equal(angular.weights, sn.quad.weights)

        assert isinstance(energy, EnergyAxis)
        assert energy.shape == (sn.ng,)

        assert spatial.label == "spatial"
        assert spatial.shape == sn.spatial_shape
        assert spatial.kind is BasisKind.NODAL
        assert spatial.weights is not None
        assert np.array_equal(spatial.weights, sn.volumes)

        assert space.shape == (sn.quad.N, sn.ng, *sn.spatial_shape)

    def test_energy_and_spatial_arms_are_bulk_space_axes_verbatim(self):
        """The scalar arms are REUSED objects, not respelled twins — the
        energy-arm rule (``EnergyAxis.from_materials``) is spelled exactly
        once, in ``bulk_space`` (Pattern 2)."""
        sn = _slab()
        scalar_axes = sn.bulk_space.axes
        assert scalar_axes is not None
        assert sn.angular_bulk_space.axes is not None
        assert sn.angular_bulk_space.axes[1] is scalar_axes[0]
        assert sn.angular_bulk_space.axes[2] is scalar_axes[1]


class TestG12Cache:
    """G1.2 — the mint is CACHED. LAW (the is/== asymmetry IS the gate)."""

    def test_same_carrier_reads_the_same_instance(self):
        sn = _slab()
        assert sn.angular_bulk_space is sn.angular_bulk_space

    def test_twin_carriers_mint_equal_but_distinct_spaces(self):
        a, b = _slab(), _slab()
        assert a.angular_bulk_space == b.angular_bulk_space
        assert a.angular_bulk_space is not b.angular_bulk_space


class TestG13GramEquivalenceDD:
    """G1.3 (re-scoped at S2b) — the axis product's Gram equals the
    HAND-BUILT dense ``G_bulk = V·w_n``, and the composite interior IS the
    cached mint.

    Until the Q2 re-point this compared against
    ``full_field_space.interior_space``'s own dense array — the shipped
    §6c witness. S2b re-pointed that interior AT ``angular_bulk_space``,
    which made the comparison tautological (single-sourcing demotes every
    gate that compared the copies), so the dense side moved IN-TEST: a
    fuller-view oracle built from raw mesh data, independent of both
    production spellings. The identity row is the unification claim
    itself."""

    def test_composite_interior_is_the_cached_mint(self):
        sn = _slab()
        assert sn.full_field_space.interior_space is sn.angular_bulk_space

    def test_all_three_metric_faces_agree_with_the_dense_oracle(self):
        sn = _slab()
        axis_built = sn.angular_bulk_space
        # The oracle: G_bulk = V·w_n, densified BY HAND from raw mesh data
        # (broadcast (N, 1, nx) — the retired production spelling, now the
        # test-side reference).
        w = np.asarray(sn.quad.weights, dtype=float)
        V = np.asarray(sn.volumes, dtype=float)
        dense = FunctionSpace(
            name="dense_oracle",
            shape=axis_built.shape,
            inner_product_weights=w.reshape(-1, 1, 1) * V.reshape(1, 1, -1),
        )

        rng = np.random.default_rng(0)
        x = rng.standard_normal(dense.shape)
        y = rng.standard_normal(dense.shape)

        # Scalar faces: bit-equal ([M] verification plan G1.3 — rel diff 0.0).
        assert axis_built.inner_product(x, y) == dense.inner_product(x, y)
        assert axis_built.norm(x) == dense.norm(x)
        # Vector faces: ≤ 4 ulp ([M] max abs Δ 2.78e-17 on this fixture).
        # apply_inverse_metric is the .H sandwich's other half (G⁻¹AᵀG) —
        # the face the composite adjoint path consumes (G2.6's substance).
        npt.assert_array_almost_equal_nulp(
            axis_built.apply_metric(x), dense.apply_metric(x), nulp=4
        )
        npt.assert_array_almost_equal_nulp(
            axis_built.apply_inverse_metric(x),
            dense.apply_inverse_metric(x),
            nulp=4,
        )


class TestG14GramEquivalenceLD:
    """G1.4 (re-scoped at S2b) — the LD arm: the Gram carries the scheme's
    moment mass on the trailing 2^d axis; the axis form reproduces the
    hand-built dense oracle, and the composite interior is the widened
    product. LAW.

    [M] R9 measured its draw's inner product bit-identical; that was the
    draw's luck, not a law — the two spellings associate the weight
    products differently, and on THIS fixture's ``rng(0)`` draw the
    near-cancelling bilinear form lands 6 ULP apart (measured 2026-08-22).
    The honest bound is nulp ≤ 64 on the cancellation-conditioned scalar
    face (vv #16: never assert tighter than construction gives); a
    mass-placement error is O(θ·value) ≈ 1e14 ULP, so the gate's
    discrimination is unharmed."""

    def test_ld_composite_interior_is_the_widened_product(self):
        """The unification claim on the LD arm: the composite's interior
        IS the cached trial mint (CS4b S5 upgraded this from ``==`` to
        ``is`` — the widening composition moved from an inline
        ``of_axes`` here into :attr:`SNMesh.angular_trial_space`, so the
        composite, the trial property, and every LD allocation share ONE
        instance), and that mint equals the moment-widened product of
        the cached base."""
        sn = _slab(scheme=LinearDiscontinuous())
        assert sn.angular_bulk_space.axes is not None
        widened = FunctionSpace.of_axes(
            *sn.angular_bulk_space.axes, sn.scheme.moment_axis(sn.axes)
        )
        assert sn.full_field_space.interior_space == widened
        assert sn.full_field_space.interior_space is sn.angular_trial_space

    def test_all_three_metric_faces_agree_on_the_ld_interior(self):
        sn = _slab(scheme=LinearDiscontinuous())
        base = sn.angular_bulk_space
        assert base.axes is not None
        widened = FunctionSpace.of_axes(
            *base.axes, sn.scheme.moment_axis(sn.axes)
        )
        # The oracle: G_bulk = V·w_n ⊗ moment_mass, densified BY HAND from
        # raw mesh + scheme data (the retired production spelling, now the
        # test-side fuller-view reference).
        w = np.asarray(sn.quad.weights, dtype=float)
        V = np.asarray(sn.volumes, dtype=float)
        mass = sn.scheme.moment_mass_diagonal(sn.axes)
        g = (w.reshape(-1, 1, 1) * V.reshape(1, 1, -1))[..., None] * mass
        dense = FunctionSpace(
            name="dense_oracle_ld",
            shape=widened.shape,
            inner_product_weights=g,
        )
        assert widened.shape == dense.shape

        rng = np.random.default_rng(0)
        x = rng.standard_normal(dense.shape)
        y = rng.standard_normal(dense.shape)

        # Cancellation-conditioned scalar face: standard-normal x·G·y sums
        # ~80 O(1) signed terms to ~0.2, so association differences amplify
        # in result-relative ULPs ([M] 6 here).
        npt.assert_array_almost_equal_nulp(
            np.array([widened.inner_product(x, y)]),
            np.array([dense.inner_product(x, y)]),
            nulp=64,
        )
        # Positive-term faces: no cancellation, tight.
        npt.assert_array_almost_equal_nulp(
            np.array([widened.norm(x)]), np.array([dense.norm(x)]), nulp=4
        )
        npt.assert_array_almost_equal_nulp(
            widened.apply_metric(x), dense.apply_metric(x), nulp=4
        )
        npt.assert_array_almost_equal_nulp(
            widened.apply_inverse_metric(x),
            dense.apply_inverse_metric(x),
            nulp=4,
        )


def _cart_axes():
    # A minimal 1-D Cartesian axis tuple (P4.6: the family consumes axes).
    return (AxisMesh(edges=np.array([0.0, 1.0])),)


def _radial_axes(kind: AxisCoord):
    # A minimal 1-D radial axis tuple of the given kind (P4.6).
    return (RadialAxisMesh(edges=np.array([0.0, 1.0]), coord=kind),)


class TestMomentAxisAdmission:
    """The scheme-side mint's ADMISSION pair (vv #11: both legs)."""

    def test_ld_mints_the_modal_mass_axis(self):
        scheme = LinearDiscontinuous()
        axis = scheme.moment_axis(_cart_axes())
        assert axis.label == "spatial_moment"
        assert axis.shape == (2,)
        assert axis.kind is BasisKind.MODAL
        assert axis.weights is not None
        assert np.array_equal(
            axis.weights,
            scheme.moment_mass_diagonal(_cart_axes()),
        )

    def test_slopeless_closure_refuses(self):
        with pytest.raises(NotImplementedError, match="no moment axis"):
            DiamondDifference().moment_axis(_cart_axes())

    # ── The CHART admission (2026-08-26).  Third arm of the same pair:
    # a multi-moment mass is defined on a Cartesian chart and is NOT
    # expressible on a curved one, so the producer must refuse rather
    # than hand back the slab's diagonal.
    #
    # ⭐ §6c — THE WITNESS IS CONSTRUCTIBLE, and that is the point of this
    # class of gate.  Before the guard, `SNMesh(Mesh1D(coord=SPHERICAL),
    # gauss_legendre(4), ..., scheme=LinearDiscontinuous())` BUILT, and its
    # moment weights measured [1., 0.33333333] -- bit-identical to a slab's,
    # on both the sphere AND the cylinder.  The wrong value was being
    # installed on two shipped charts, silently.  A gate that only mutated
    # the SUT would have proved nothing about that.

    @pytest.mark.parametrize(
        "kind", [AxisCoord.RADIAL_SPHERICAL, AxisCoord.RADIAL_CYLINDRICAL]
    )
    def test_curvilinear_multi_moment_mass_is_refused_not_slab_defaulted(
        self, kind: AxisCoord,
    ) -> None:
        """LD on a curved chart REFUSES; before the guard it returned the slab's.

        The true ``M/V`` there is cell-dependent AND non-diagonal (a
        spherical pole cell wants ``[[1, 0.5], [0.5, 0.4]]``), which a
        per-axis ``Axis`` weight vector cannot express -- so no honest
        value exists to return.  The MACHINERY half of the old two-blocker
        wording (#409, the non-Hadamard metric) was discharged by P7 (the
        dense-metric family); the refusal stands on the VALUE alone --
        #158's cell solve is what gives a chosen ``G`` a consumer.
        """
        with pytest.raises(NotImplementedError, match="no moment mass"):
            LinearDiscontinuous().moment_mass_diagonal(_radial_axes(kind))
        with pytest.raises(NotImplementedError, match="no moment mass"):
            LinearDiscontinuous().moment_axis(_radial_axes(kind))

    @pytest.mark.parametrize(
        "kind", [AxisCoord.RADIAL_SPHERICAL, AxisCoord.RADIAL_CYLINDRICAL]
    )
    def test_the_moment_mass_refusal_names_only_the_value_blocker(
        self, kind: AxisCoord,
    ) -> None:
        """E1 (P7 S4): the refusal stands on ONE blocker after the
        dense-metric family landed.

        The message still refuses (the pinned ``no moment mass`` fragment
        survives), still names #158 (the value's missing consumer), and
        no longer names #409 — the machinery half was discharged by P7.
        The absence assert is the half a ``match=`` cannot pin: it stops
        the two-blocker wording drifting back while reading as a mere
        rewording.
        """
        with pytest.raises(NotImplementedError, match="no moment mass") as exc:
            LinearDiscontinuous().moment_mass_diagonal(_radial_axes(kind))
        message = str(exc.value)
        assert "158" in message, "the VALUE blocker (#158) must stay named"
        assert "409" not in message, (
            "the discharged MACHINERY blocker (#409) must not be re-cited"
        )

    def test_mixed_axes_refuse_and_name_only_the_curved_kind(self):
        """P4.6's granularity witness: a mixed (z, r)-style tuple refuses,
        NAMING only the radial axis kind — the per-axis question the
        whole-mesh enum structurally could not pose (its projection
        refuses mixed multi-axis tuples outright, ``coord_system`` at
        ``transport/mesh/axis.py``).  No mesh ctor builds this today;
        the bare-axes spelling is the constructible witness (§6c).
        """
        axes = (
            _cart_axes()[0],
            _radial_axes(AxisCoord.RADIAL_CYLINDRICAL)[0],
        )
        with pytest.raises(
            NotImplementedError, match="mass on a radial_cylindrical axis",
        ):
            LinearDiscontinuous().moment_mass_diagonal(axes)

    @pytest.mark.parametrize(
        "kind", [AxisCoord.RADIAL_SPHERICAL, AxisCoord.RADIAL_CYLINDRICAL]
    )
    def test_slopeless_mass_is_admitted_on_a_curved_chart(
        self, kind: AxisCoord,
    ) -> None:
        """The width control: the guard must not be too WIDE.

        A single-moment scheme's cell-average mass is :math:`V/V = 1`
        whatever the measure, so DD is unaffected by the chart.  Without
        this leg the refusal above is compatible with a guard that simply
        rejects every curvilinear chart.
        """
        assert np.array_equal(
            DiamondDifference().moment_mass_diagonal(_radial_axes(kind)),
            np.ones(1),
        )


class TestAngularTrialSpace:
    """``SNMesh.angular_trial_space`` — the ONE widening mint (CS4b S5).

    The property replaces the retired ``spatial_moments=`` factory int:
    a call site widens by SELECTING this mint instead of threading the
    scheme's basis size through a factory parameter. Claims: the
    slopeless identity (LAW — the two mints collapse to one instance),
    the LD structure (RECORD — base axes + the scheme's moment axis),
    and the single-source identity with the composite interior (LAW).
    """

    def test_slopeless_trial_space_IS_the_bulk_space(self):
        """DD (width 1): not merely ``==`` — the SAME cached instance,
        so slopeless consumers pay nothing and the mints cannot drift."""
        sn = _slab(scheme=DiamondDifference())
        assert sn.angular_trial_space is sn.angular_bulk_space

    def test_ld_trial_space_appends_the_scheme_moment_axis(self):
        sn = _slab(scheme=LinearDiscontinuous())
        base = sn.angular_bulk_space
        trial = sn.angular_trial_space
        assert base.axes is not None and trial.axes is not None
        # The base's axes verbatim, then the scheme's own moment axis.
        assert trial.axes[: len(base.axes)] == base.axes
        (tail,) = trial.axes[len(base.axes) :]
        assert tail == sn.scheme.moment_axis(sn.axes)
        assert trial.shape == (*base.shape, 2)

    def test_ld_trial_space_is_cached_and_single_sourced(self):
        sn = _slab(scheme=LinearDiscontinuous())
        assert sn.angular_trial_space is sn.angular_trial_space
        assert sn.full_field_space.interior_space is sn.angular_trial_space

    def test_field_allocation_rides_the_trial_mint(self):
        """The S5 end-state, at both widths: a field allocated on the
        trial mint IS an element of it — the DD leg doubles as the
        collapse witness (its trial mint IS the bulk instance). Until
        the sugar retired (S5.4) this row was the BRIDGE gate: ``[M]``
        2026-08-24, ``zeros_on(mesh, spatial_moments=…)``'s derived
        space was ``is``-identical to this mint at DD and ``==`` at LD,
        proving the ~700-site migration a pure re-spelling."""
        from orpheus.transport.fields.angular_flux import AngularFlux

        dd = _slab(scheme=DiamondDifference())
        assert AngularFlux.zeros(dd.angular_trial_space).space is dd.angular_bulk_space
        ld = _slab(scheme=LinearDiscontinuous())
        psi = AngularFlux.zeros(ld.angular_trial_space)
        assert psi.space is ld.angular_trial_space
        assert psi.values.shape == (*ld.angular_bulk_space.shape, 2)


class TestG15ConePredicates:
    """G1.5 — the cone predicates, stated. RECORD ([M] verification plan
    Finding 8 + R9)."""

    def test_nodal_bulk_families_answer_true(self):
        sn = _slab()
        assert sn.angular_bulk_space.has_coordinate_cone is True
        assert sn.bulk_space.has_coordinate_cone is True

    def test_the_harmonic_moment_family_answers_false_and_the_trace_family_none(self):
        """The moment head is a MODAL axis since CS4c step 6 item 6.2c-ii
        (a spectral coefficient may be negative for a positive function), so
        the moment product's cone answer is a definite ``False`` — the typed
        refusal :meth:`Field.cone_violations` turns it into — where the
        axes-less head answered ``None`` (unanswerable). The trace family is
        still name-built and still answers ``None``."""
        sn = _slab()
        moment_space = HarmonicMomentFlux.zeros_for_mesh_and_L(sn, 1).space
        assert moment_space.has_coordinate_cone is False
        assert sn.angular_trace.has_coordinate_cone is None

    def test_the_ld_moment_tail_is_modal_so_the_cone_reads_false(self):
        """The Q6-ratified routing base: a moment-tailed LD bulk space
        answers ``False`` (signed slope coefficients are legal on a
        positive function), which ``Field.cone_violations`` turns into the
        typed refusal. The exact modal test (the vertex theorem) is #400."""
        sn = _slab(scheme=LinearDiscontinuous())
        assert sn.angular_bulk_space.axes is not None
        widened = FunctionSpace.of_axes(
            *sn.angular_bulk_space.axes, sn.scheme.moment_axis(sn.axes)
        )
        assert widened.has_coordinate_cone is False
