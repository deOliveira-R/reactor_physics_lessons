#!/usr/bin/env python3
"""Central Limit Theorem demonstration via Monte Carlo dice rolling.

Rolls *batch_size* dice *sample_size* times and histograms the batch means,
showing convergence to a normal distribution centred on 3.5.

Port of MATLAB ``00.centralLimitTheorem/centralLimitTheorem.m``.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path("results")


def main():
    rng = np.random.default_rng(seed=42)

    batch_size = 100
    sample_size = 10_000

    means = np.array([rng.integers(1, 7, size=batch_size).mean()
                       for _ in range(sample_size)])

    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.hist(means, bins=32, edgecolor="black")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_xlim(1, 6)
    fig.tight_layout()
    fig.savefig(OUTPUT / "central_limit_theorem.pdf")
    plt.close(fig)

    print(f"Mean of means: {means.mean():.4f}  (expected: 3.5)")
    print(f"Std of means:  {means.std():.4f}")
    print(f"Plot saved to {OUTPUT.resolve()}/central_limit_theorem.pdf")


if __name__ == "__main__":
    main()
