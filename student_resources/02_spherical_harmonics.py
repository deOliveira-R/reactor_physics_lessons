#!/usr/bin/env python3
"""Real spherical harmonics visualisation.

Computes and plots the real spherical harmonic Y_l^m(theta, phi) as a 3-D
surface whose radius equals |Y_l^m|.

Port of MATLAB ``00.sphericalHarmonics/sphericalHarmonics.m`` + ``drive.m``.
"""

from pathlib import Path
from math import factorial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lpmv

OUTPUT = Path("02_results")


def spherical_harmonic_surface(l: int, m: int, n: int = 50):
    """Compute (x, y, z) mesh for the real spherical harmonic Y_l^m."""
    theta = np.linspace(0, np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    # Associated Legendre polynomial P_l^|m|(cos theta)
    P = lpmv(abs(m), l, np.cos(THETA))

    # Normalisation (real spherical harmonics convention)
    norm = np.sqrt((1 + (m != 0)) * factorial(l - abs(m)) / factorial(l + abs(m)))

    # Angular part: cos(m*phi) for m >= 0, sin(|m|*phi) for m < 0
    angular = np.cos(m * PHI) if m >= 0 else np.sin(abs(m) * PHI)

    R = norm * P * angular

    x = np.abs(R) * np.sin(THETA) * np.cos(PHI)
    y = np.abs(R) * np.sin(THETA) * np.sin(PHI)
    z = np.abs(R) * np.cos(THETA)

    return x, y, z, R


def main():
    l, m = 2, 2

    x, y, z, R = spherical_harmonic_surface(l, m)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="coolwarm", edgecolor="none", alpha=0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"$Y_{{{l}}}^{{{m}}}$")
    fig.tight_layout()
    fig.savefig(OUTPUT / "spherical_harmonics.pdf")
    plt.close(fig)

    print(f"Spherical harmonic Y_{l}^{m} plotted.")
    print(f"Plot saved to {OUTPUT.resolve()}/spherical_harmonics.pdf")


if __name__ == "__main__":
    main()
