from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from detector import hybrid_candidates


def test_function(t: np.ndarray) -> np.ndarray:
    """
    Synthetic oscillatory test function with multiple simple zeros.
    """
    return np.sin(0.015 * t**2) + 0.03 * np.cos(0.2 * t)


def main():
    x = np.linspace(0.0, 80.0, 2000)
    fvals = test_function(x)

    candidates, score, P, W = hybrid_candidates(
        x,
        fvals,
        sigma_pts=4.0,
        eps=1e-8,
        max_candidates=12,
        min_spacing=2.0,
    )

    print("Detected candidate basin centers:")
    for c in candidates:
        print(f"  {c:.6f}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(x, fvals, label="sampled function")
    for c in candidates:
        axes[0].axvline(c, linestyle="--", alpha=0.7)
    axes[0].set_ylabel("f(x)")
    axes[0].set_title("Hybrid detector example on sampled oscillatory data")
    axes[0].legend()

    axes[1].plot(x, score, label="hybrid score")
    for c in candidates:
        axes[1].axvline(c, linestyle="--", alpha=0.7)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("D(x)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
