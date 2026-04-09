"""Analytical derivations for verification of transport solvers.

SymPy-based symbolic derivations that produce reference eigenvalues
for verifying numerical solvers. Each derivation module returns
VerificationCase objects with the analytical k_inf, materials,
geometry parameters, and LaTeX documentation.
"""

from .reference_values import get, all_names
