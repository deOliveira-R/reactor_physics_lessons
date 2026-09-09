r"""CS1 migration gate D5 — the homogeneous solve is BYTE-stable across 3b.

# CS1 migration gate — retire after the merge cycle. Its one claim is
# "step 3b (bulk_space + chain + solver rewiring) changed no value"; once
# the campaign merges green it is subsumed by ``test_kinf_exact`` (the L1
# correctness anchor) and ``test_as_matrix_equals_retired_as_dense_loop``
# (the materialization byte pin), per the aggressive-retirement rule.
# ⚠ LIVE CONSUMERS (CS4a-R QA-F11): ``_mixture_cases`` is imported from
# this module by ``test_operator_spaces.py`` G2.1 AND (since 2026-09-08)
# by ``test_coda_anchors.py``, which builds its eight-case population
# from it so there is ONE list — move the helper to a shared home in any
# retiring commit, or the retirement ImportErrors two green gates.
# (And CS4a made this module the D5 exit-criterion gate, so the
# retire-note's own premise is stale until the campaign re-rules it.)
#
# ⛔ 2026-09-08 — the retire-note is stale AGAIN, in the direction that
# matters: this module is now the CS4c coda's bit-identity wall. The
# coda re-sourced the whole homogeneous data path (C1 ``5caad3d6``: the
# problem gets a hub and mints its own fields; C2 ``39e7f32f``: the
# fabricated one-cell carrier and its factory retire) and this gate read
# 8 of 8 across both commits, which is what licensed calling the change
# a re-source rather than a re-baseline. The capture
# (``_fixtures/cs1_prewiring.json``, ``24a991ba``) MUST NOT be
# regenerated: its value is precisely that it predates every campaign
# that has since claimed to move no bytes.

Bit-identity (``np.array_equal`` / exact ``==``, never ``allclose``) is the
correct contract here: the counting-measure theorem makes the CS1 space
threading a value NO-OP by construction (identity metrics; guards compare
spaces, never values), so any ULP motion is a real finding, not FP noise.

**Population** — exhaustive over what the tree ships, PRODUCING mixtures
only (ruled Q-T5): the eigenvalue entry is meaningless for a non-fissile
mixture, and ``[M]`` the ``is_producing`` screen at `24a991ba` admits the
four ``derivations.get`` homogeneous cases, the ``eg``-bearing variant,
and ``get_mixture("A", ·)`` — regions B/C/D are all non-producing (the
Q-T5 ruling's mechanism is the screen; an earlier "A/C" enumeration in
the design record was wrong against this measurement).

**Mechanics** — the baseline was captured at 3a-HEAD (`24a991ba`,
post-rename, PRE-wiring) by running THIS module's ``capture`` entry point
(``python tests/homogeneous/test_byte_stability.py capture``), so the
capture script and the gate share ONE payload helper
(:func:`_homogeneous_payload`) — coding-elegance Pattern 2; a generator
computing the payload one way and a gate recomputing it another way is
exactly the snapshot-drift failure. Flux is stored as raw base64 bytes so
a mismatch localizes (a hash says *that* it moved; bytes say *where*).
"""

from __future__ import annotations

import base64
import dataclasses
import json
import pathlib

import numpy as np
import pytest

from orpheus.derivations import get
from orpheus.derivations.common.xs_library import get_mixture
from orpheus.homogeneous.solver import solve_homogeneous_infinite

pytestmark = pytest.mark.foundation

_FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "cs1_prewiring.json"

#: The eg-bearing variant's grid (the one homogeneous eg idiom in the tree,
#: ``test_homogeneous.py`` eg-block).
_EDGES_2G = np.array([1.0e7, 1.0e3, 1.0e-3])


def _mixture_cases() -> dict[str, object]:
    """name -> producing Mixture, exhaustive over what the tree ships."""
    cases: dict[str, object] = {}
    for name in ("homo_1eg", "homo_2eg", "homo_4eg", "homo_2eg_n2n"):
        mix = next(iter(get(name).materials.values()))
        assert mix.is_producing  # population precondition, not a gate
        cases[name] = mix
    eg_mix = dataclasses.replace(
        next(iter(get("homo_2eg").materials.values())), eg=_EDGES_2G
    )
    cases["homo_2eg_with_eg"] = eg_mix
    for k in ("1g", "2g", "4g"):
        mix = get_mixture("A", k)
        assert mix.is_producing
        cases[f"mixture_A_{k}"] = mix
    return cases


def _homogeneous_payload(mix) -> dict[str, object]:
    """The ONE payload both the capture and the gate compute (Pattern 2)."""
    result = solve_homogeneous_infinite(mix)
    flux = np.ascontiguousarray(result.flux, dtype=float)
    return {
        "k_inf": float(result.k_inf).hex(),
        "flux_b64": base64.b64encode(flux.tobytes()).decode("ascii"),
        "flux_shape": list(flux.shape),
        "sig_prod": float(result.sig_prod).hex(),
        "sig_abs": float(result.sig_abs).hex(),
    }


@pytest.mark.parametrize("case", sorted(_mixture_cases()))
def test_homogeneous_results_are_byte_stable(case: str) -> None:
    """D5 — every numeric in the payload is BIT-identical to the 3a-HEAD
    capture: ``k_inf``/``sig_prod``/``sig_abs`` as exact floats, the flux
    as ``np.array_equal`` on the decoded bytes."""
    recorded = json.loads(_FIXTURE.read_text())[case]
    live = _homogeneous_payload(_mixture_cases()[case])
    for scalar in ("k_inf", "sig_prod", "sig_abs"):
        if float.fromhex(str(live[scalar])) != float.fromhex(str(recorded[scalar])):
            pytest.fail(
                f"{case}.{scalar} moved: {recorded[scalar]} -> {live[scalar]} "
                f"(the CS1 threading must be a value no-op by the counting "
                f"theorem — any motion is a real finding)"
            )
    rec_flux = np.frombuffer(
        base64.b64decode(str(recorded["flux_b64"])), dtype=float
    ).reshape(tuple(int(n) for n in recorded["flux_shape"]))
    live_flux = np.frombuffer(
        base64.b64decode(str(live["flux_b64"])), dtype=float
    ).reshape(tuple(int(n) for n in live["flux_shape"]))  # type: ignore[union-attr]
    if not np.array_equal(rec_flux, live_flux):
        moved = np.flatnonzero(rec_flux.ravel() != live_flux.ravel())
        pytest.fail(
            f"{case}.flux moved at flat indices {moved.tolist()}: "
            f"{rec_flux.ravel()[moved]} -> {live_flux.ravel()[moved]}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        _FIXTURE.parent.mkdir(exist_ok=True)
        payload = {name: _homogeneous_payload(mix) for name, mix in _mixture_cases().items()}
        _FIXTURE.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"captured {len(payload)} cases -> {_FIXTURE}")
    else:
        print("usage: python tests/homogeneous/test_byte_stability.py capture")
