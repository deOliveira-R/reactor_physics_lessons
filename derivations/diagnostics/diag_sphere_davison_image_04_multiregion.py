"""Diagnostic: Davison image-series with multi-region σ_t.

Created by numerics-investigator on 2026-04-24.
Issue #132 viability probe — Step 4 (final).

Key generalisation: for multi-region σ_t(r), the Davison reduction
no longer gives the simple E_1 closed form. The chord optical depth
along a ray becomes a piecewise sum:

    τ(r, r') = Σ_k σ_{t,k} · ℓ_k(r, r')

where ℓ_k is the chord segment in region k. For the radial-direction
Davison reduction, the chord between observer at r and source at r'
passes through specific annular boundaries; the kernel becomes:

    K_MR(r, r') = ∫₀^|r-r'| (1/s) exp(-τ_MR(r, r', s)) ds

This integral is no longer expressible as a single E_1 — it's a
chord-traced exponential whose closed form depends on the path. For
1G/2R sphere, the Davison reduction breaks the simple "image at -r'"
algebra unless we approximate the kernel.

CONCLUSION
==========
The Davison image method for sphere with multi-region σ_t requires
SIGNIFICANT additional infrastructure:
1. Per-ray chord tracking through annular boundaries (already done
   in cp_geometry, but for shifts-and-flips of (r, r') this needs
   extension to "image-point" coordinates that may lie outside the
   physical sphere).
2. The image series at -r', 2R±r', etc., requires the optical depth
   of the IMAGE chord — which is nonsensical (the image points are
   outside the physical sphere where σ_t is undefined).

The image method MIXES geometry (mirror reflection) with material
(σ_t along the chord) in a way that's well-defined for HOMOGENEOUS
σ_t (just exp(-σ_t · |chord|)) but NOT for piecewise σ_t (where the
image chord goes through "imaginary material").

This is a STRUCTURAL barrier. The image method as derived for
homogeneous sphere CANNOT extend to multi-region without ad-hoc
material-extension assumptions (e.g., "the outer-region material
extends to infinity"), which would NOT recover the cp_sphere
white-BC reference.

This script is documentation of the structural barrier, not a
numerical test (no clean way to set up the MR image kernel).
"""

from __future__ import annotations

import numpy as np


def explain_image_method_breakdown_for_multiregion():
    """Document why the image method does not generalise to MR sphere."""
    print("=" * 76)
    print("Image method for multi-region σ_t: structural breakdown")
    print("=" * 76)
    print()
    print("Setup")
    print("-----")
    print("  Sphere of radius R with two regions:")
    print("    region 0 (fuel):     0   ≤ r < R/2,   σ_t = σ_t,fuel")
    print("    region 1 (moderator): R/2 ≤ r ≤ R,   σ_t = σ_t,mod")
    print()
    print("Image series at observer r ∈ region 0, source r' ∈ region 1")
    print("--------------------------------------------------------------")
    print("  Image positions (specular at R, vacuum at 0):")
    print("    r', -r', 2R-r', 2R+r', -2R-r', -2R+r', ...")
    print()
    print("  For the n=0 source at r': chord from r to r' lies entirely")
    print("  inside the sphere; optical depth is well-defined as the")
    print("  piecewise sum of σ_{t,k} · ℓ_k along the chord.")
    print()
    print("  For the n=0 image at -r': chord from r to -r' includes the")
    print("  segment from r through 0 to -r'. The point -r' lies OUTSIDE")
    print("  the physical sphere — σ_t(-r') is undefined.")
    print()
    print("  For the n=1 source-image at 2R-r': chord from r to 2R-r'")
    print("  also exits the sphere at r=R, then the 'image chord' would")
    print("  continue into σ_t-undefined region.")
    print()
    print("Why this breaks")
    print("---------------")
    print("  The image method works by saying: 'the field at the boundary")
    print("  is the SAME as if there were image sources placed in a")
    print("  material that extends through the boundary.' For a slab")
    print("  Peierls problem with constant σ_t on (-∞, ∞), this is well-")
    print("  defined: the image sources contribute through paths that")
    print("  wrap around the boundary.")
    print()
    print("  For sphere with MR σ_t, the image sources would need to be")
    print("  placed in a 'virtual material extension'. There is NO unique")
    print("  way to extend σ_t outside the physical sphere — and any")
    print("  choice (e.g., 'extend the outer region to infinity') would")
    print("  give a DIFFERENT answer than the physical white-BC problem.")
    print()
    print("  The white-BC physics is: re-emit returning current with")
    print("  ISOTROPIC angular distribution. This is not equivalent to")
    print("  ANY material extension. There is no method-of-images")
    print("  equivalent for white BC, even on a homogeneous sphere.")
    print()
    print("Empirical evidence from Step 3")
    print("------------------------------")
    print("  Sphere 1G/1R homogeneous (R = 1 mfp, c = 1.25):")
    print("    cp_sphere white-BC k_inf = 1.500000  (reference)")
    print("    Davison image series k_eff = 0.704   (-53% error)")
    print()
    print("  The image series is well-defined for homogeneous σ_t (the")
    print("  material extends naturally), CONVERGES (n_max=5 saturates),")
    print("  but converges to the WRONG number — proving that even when")
    print("  the image method is well-defined, it solves a DIFFERENT")
    print("  problem than white-BC sphere.")
    print()
    print("  Specifically, the image series with sign pattern")
    print("  (specular at R, vacuum/Davison-u(0)=0 at center) is")
    print("  equivalent to a 1D-slab on (-R, R) with vacuum at both")
    print("  ends. This is NOT the spherical white-BC problem.")


if __name__ == "__main__":
    explain_image_method_breakdown_for_multiregion()
