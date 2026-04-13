"""Central V&V metadata registry, populated at pytest collection time.

The ``conftest.py`` hook walks every collected test item, resolves its
V&V level from (in precedence order) explicit markers, ``TestL<N>``
class naming, ``test_l<N>_*`` function naming, and the
``VerificationCase`` inherited from a ``ref()`` fixture call — then
records a :class:`TestMetadata` entry here, keyed by pytest nodeid.

The registry is the single structured source of truth for the audit
tool, the Sphinx verification-matrix generator, and any agent query
answering "which tests verify equation X" or "what's the V&V level
distribution for module Y."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VVLevel = Literal["L0", "L1", "L2", "L3"]

# Resolution tag for how a test got its V&V level. Used by the audit
# tool to show provenance and by CI gates to distinguish explicit
# tagging from inherited/heuristic tagging.
#
# - "explicit"     : @pytest.mark.lN on the test itself
# - "verify"       : @verify.lN(...) decorator on class or function
# - "class-name"   : TestL<N>Foo class naming convention
# - "func-name"    : test_l<N>_* function naming convention
# - "case"         : inherited from VerificationCase.vv_level via ref()
# - "unmarked"     : no level could be determined
LevelSource = Literal[
    "explicit", "verify", "class-name", "func-name", "case", "unmarked"
]


@dataclass(frozen=True)
class TestMetadata:
    """V&V facts about one pytest item.

    Fields are populated by ``conftest.pytest_collection_modifyitems``
    and are frozen from that point on. Read-only from user code.
    """

    nodeid: str
    file: str
    level: VVLevel | None
    level_source: LevelSource
    equations: tuple[str, ...] = ()
    catches: tuple[str, ...] = ()
    case_names: tuple[str, ...] = ()
    slow: bool = False


# Keyed by ``item.nodeid``. Cleared and refilled on every collection
# pass so stale entries don't accumulate during a pytest watch session.
TEST_REGISTRY: dict[str, TestMetadata] = {}


def clear() -> None:
    """Drop all registry entries. Called at the start of each collection."""
    TEST_REGISTRY.clear()


def record(meta: TestMetadata) -> None:
    """Insert or overwrite an entry."""
    TEST_REGISTRY[meta.nodeid] = meta


def by_level(level: VVLevel | None) -> list[TestMetadata]:
    """Return every entry matching ``level`` (``None`` for unmarked)."""
    return [m for m in TEST_REGISTRY.values() if m.level == level]


def by_equation(label: str) -> list[TestMetadata]:
    """Return every entry whose ``equations`` tuple contains ``label``."""
    return [m for m in TEST_REGISTRY.values() if label in m.equations]
