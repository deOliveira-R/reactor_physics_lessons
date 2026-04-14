"""Unit tests for the V&V audit CLI scanner.

Covers the ``:label:`` / ``.. vv-status: <label> documented`` parsing
that feeds the orphan-equation gate. See ``tests/_harness/audit.py``
for the full behaviour contract; this file fixes the invariants:

* every ``:label:`` in a theory RST becomes a labelled equation,
* a ``.. vv-status: <label> documented`` comment excludes its
  label from the orphan set even when no test verifies it,
* the ``documented`` set is clipped to labels that actually exist
  (stale sentinels are silently dropped so typos don't leak),
* non-documented status values are ignored (future-proof).

The test writes fixture RST files into a tmp_path so it is
hermetic and independent of the real ``docs/theory`` tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._harness.audit import _scan_theory_equations

pytestmark = pytest.mark.l0


def _write_rst(path: Path, name: str, body: str) -> None:
    (path / f"{name}.rst").write_text(body, encoding="utf-8")


def test_scanner_collects_labels(tmp_path: Path) -> None:
    _write_rst(tmp_path, "sample", """
.. math::
   :label: foo

   \\int f(x)\\,dx = 1

.. math::
   :label: bar

   a + b = c
""")
    labels, documented = _scan_theory_equations(tmp_path)
    assert labels == {"foo", "bar"}
    assert documented == set()


def test_scanner_marks_documented(tmp_path: Path) -> None:
    """A ``.. vv-status: <label> documented`` comment excludes the label."""
    _write_rst(tmp_path, "sample", """
.. math::
   :label: boltzmann

   \\partial_t \\psi + \\Omega \\cdot \\nabla \\psi = S

.. vv-status: boltzmann documented

.. math::
   :label: testable

   x = 1
""")
    labels, documented = _scan_theory_equations(tmp_path)
    assert labels == {"boltzmann", "testable"}
    assert documented == {"boltzmann"}


def test_scanner_drops_sentinels_without_label(tmp_path: Path) -> None:
    """A sentinel referring to a non-existent label is silently dropped.

    This keeps the orphan gate stable under typos: misspelling a
    label in the sentinel fails closed (the real orphan is still
    reported) rather than silently widening the documented set.
    """
    _write_rst(tmp_path, "typo", """
.. math::
   :label: real_label

   x = 1

.. vv-status: rael_label documented
""")
    labels, documented = _scan_theory_equations(tmp_path)
    assert labels == {"real_label"}
    assert documented == set()


def test_scanner_ignores_non_documented_status(tmp_path: Path) -> None:
    """Only ``documented`` is a recognised status; others are ignored.

    Leaves room for future statuses (e.g. ``tested``, ``validated``)
    without changing the semantics of the current gate.
    """
    _write_rst(tmp_path, "misc", """
.. math::
   :label: alpha

   x = 1

.. vv-status: alpha verified
""")
    labels, documented = _scan_theory_equations(tmp_path)
    assert labels == {"alpha"}
    assert documented == set()


def test_scanner_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    labels, documented = _scan_theory_equations(tmp_path / "nope")
    assert labels == set()
    assert documented == set()


def test_scanner_handles_subdirectories(tmp_path: Path) -> None:
    """``rglob`` picks up labels in nested RST files."""
    sub = tmp_path / "nested"
    sub.mkdir()
    _write_rst(tmp_path, "top", """
.. math::
   :label: top_label

   1 = 1
""")
    _write_rst(sub, "deep", """
.. math::
   :label: deep_label

   2 = 2

.. vv-status: deep_label documented
""")
    labels, documented = _scan_theory_equations(tmp_path)
    assert labels == {"top_label", "deep_label"}
    assert documented == {"deep_label"}
