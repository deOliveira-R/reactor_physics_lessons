"""Types for the analytical verification system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VerificationCase:
    """Reference solution from an analytical derivation.

    Attributes
    ----------
    name : str
        Unique identifier following ``method_geometry_Neg_Nrg`` convention,
        e.g. ``"cp_slab_2eg_2rg"``. Homogeneous is ``"homo_Neg"``.
    k_inf : float
        Analytical multiplication factor.
    method : str
        Solver method abbreviation: ``"homo"``, ``"cp"``, ``"sn"``,
        ``"moc"``, ``"mc"``, ``"dif"``.
    geometry : str
        Geometry abbreviation: ``"--"``, ``"slab"``, ``"cyl1D"``, ``"sph1D"``.
    n_groups : int
        Number of energy groups.
    n_regions : int
        Number of spatial regions (1 = homogeneous).
    materials : dict[int, Any]
        Material-ID to Mixture mapping (same convention as solvers).
    geom_params : dict
        Geometry constructor keyword arguments (empty for homogeneous).
    latex : str
        SymPy-generated LaTeX showing the derivation steps.
    description : str
        Human-readable summary of the verification case.
    tolerance : str
        Expected accuracy, e.g. ``"< 1e-10"``, ``"O(h²)"``, ``"z < 5σ"``.
    """

    name: str
    k_inf: float
    method: str
    geometry: str
    n_groups: int
    n_regions: int
    materials: dict[int, Any]
    geom_params: dict
    latex: str
    description: str
    tolerance: str = ""
