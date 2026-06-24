
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from detector import hybrid_candidates, refine_candidates_newton


def residual(lam: float | np.ndarray) -> float | np.ndarray:
    """
    Analytic residual for the model Sturm--Liouville problem.

    R(lambda) = sin(sqrt(lambda)).
    """
    lam = np.asarray(lam)
    out = np.sin(np.sqrt(np.maximum(lam, 0.0)))
    return out


def exact_eigenvalues(n_max: int, lam_max: float) -> np.ndarray:
    """
    Exact zeros/eigenvalues lambda_n = (n*pi)^2 up to lam_max.
    """
    vals = []
    for n in range(1, n_max + 1):
        val = (n * np.pi) ** 2
        if val <= lam_max:
            vals.append(val)
    return np.array(vals, dtype=float)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Sample the residual on a coarse grid.
    # ------------------------------------------------------------------
    lam_min = 0.0
    lam_max = 1200.0
    n_grid = 1800

    lams = np.linspace(lam_min, lam_max, n_grid)
    clean_vals = residual(lams)

    # Optional mild perturbation to illustrate sampled-data use.
    # Set noise_sigma = 0.0 for a clean deterministic demo.
    rng = np.random.default_rng(1234)
    noise_sigma = 0.01
    fvals = clean_vals + noise_sigma * rng.standard_normal(len(lams))

    # ------------------------------------------------------------------
    # 2. Run the hybrid detector.
    # ------------------------------------------------------------------
    candidates, score, P, W = hybrid_candidates(
        lams,
        fvals,
        sigma_pts=4.0,
        eps=1e-8,
        max_candidates=12,
        min_spacing=25.0,
    )

    # Optional Newton refinement on the clean analytic residual.
    refined = refine_candidates_newton(
        candidates,
        residual_fn=lambda x: float(residual(x)),
        domain=(lam_min, lam_max),
        maxiter=20,
        tol=1e-12,
        dedup_tol=1e-6,
    )

    exact = exact_eigenvalues(n_max=20, lam_max=lam_max)

    # ------------------------------------------------------------------
    # 3. Print a compact summary.
    # ------------------------------------------------------------------
    print("\nHybrid zero detector: minimal Sturm--Liouville example")
    print("=" * 64)
    print(f"Grid interval: [{lam_min:.1f}, {lam_max:.1f}]")
    print(f"Grid size:     {n_grid}")
    print(f"Noise sigma:   {noise_sigma}")
    print("\nDetector candidate basin centers:")
    for c in np.sort(candidates):
        print(f"  {c:12.6f}")

    print("\nNewton-refined roots from detector candidates:")
    for r in np.sort(refined):
        print(f"  {r:12.6f}")

    print("\nExact eigenvalues in interval:")
    for e in exact:
        print(f"  {e:12.6f}")

    # ------------------------------------------------------------------
    # 4. Plot residual, detector score, and detected candidates.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Top: sampled residual.
    axes[0].plot(lams, fvals, label="sampled residual", linewidth=1.2)
    axes[0].plot(lams, clean_vals, label="clean residual", linewidth=1.0, alpha=0.6)

    for e in exact:
        axes[0].axvline(e, linestyle=":", linewidth=1.0, alpha=0.8)

    for c in candidates:
        axes[0].axvline(c, linestyle="--", linewidth=1.0, alpha=0.8)

    axes[0].set_ylabel(r"$R(\lambda)$")
    axes[0].set_title("Hybrid detector example on a Sturm--Liouville residual")
    axes[0].legend(loc="upper right")

    # Bottom: log-scaled detector score for readability.
    score_plot = np.log10(score + 1e-12)
    axes[1].plot(lams, score_plot, label=r"$\log_{10}(D+\epsilon)$", linewidth=1.2)

    for e in exact:
        axes[1].axvline(e, linestyle=":", linewidth=1.0, alpha=0.8)

    for c in candidates:
        axes[1].axvline(c, linestyle="--", linewidth=1.0, alpha=0.8)

    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel(r"$\log_{10}(D)$")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
