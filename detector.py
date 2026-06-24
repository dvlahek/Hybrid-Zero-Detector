from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import newton


def local_minima_indices(arr: np.ndarray) -> np.ndarray:
    """Return indices of strict local minima of a 1D array."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1 or len(arr) < 3:
        return np.array([], dtype=int)

    return np.array(
        [i for i in range(1, len(arr) - 1) if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]],
        dtype=int,
    )


def select_candidates_by_spacing(
    x: np.ndarray,
    score: np.ndarray,
    minima_idx: np.ndarray,
    max_candidates: int | None = None,
    min_spacing: float | None = None,
) -> np.ndarray:
    """
    Rank detector minima by score value and optionally enforce minimum spacing.
    """
    if len(minima_idx) == 0:
        return np.array([], dtype=float)

    minima_idx = np.asarray(minima_idx, dtype=int)
    minima_idx = minima_idx[np.argsort(score[minima_idx])]  # best minima first

    chosen = []
    for k in minima_idx:
        xk = x[k]
        if min_spacing is not None and any(abs(xk - c) < min_spacing for c in chosen):
            continue
        chosen.append(float(xk))
        if max_candidates is not None and len(chosen) >= max_candidates:
            break

    return np.array(chosen, dtype=float)


def gaussian_component(fvals: np.ndarray, sigma_pts: float = 3.0) -> np.ndarray:
    """
    Gaussian-smoothed signal component P used in the detector.
    """
    return gaussian_filter1d(np.asarray(fvals, dtype=float), sigma=sigma_pts, mode="nearest")


def derivative_component(
    x: np.ndarray,
    fvals: np.ndarray,
    sigma_pts: float = 3.0,
) -> np.ndarray:
    """
    Smoothed derivative component W used in the detector.
    """
    x = np.asarray(x, dtype=float)
    fvals = np.asarray(fvals, dtype=float)
    dvals = np.gradient(fvals, x)
    return gaussian_filter1d(dvals, sigma=max(1.0, sigma_pts / 2.0), mode="nearest")


def hybrid_score(
    x: np.ndarray,
    fvals: np.ndarray,
    sigma_pts: float = 3.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the practical hybrid detector score

        D = |P| / (|W| + eps),

    where P is a Gaussian-smoothed signal proxy and W is a smoothed derivative proxy.

    Returns
    -------
    score : np.ndarray
        Hybrid detector score.
    P : np.ndarray
        Gaussian-smoothed signal component.
    W : np.ndarray
        Smoothed derivative component.
    """
    P = gaussian_component(fvals, sigma_pts=sigma_pts)
    W = derivative_component(x, fvals, sigma_pts=sigma_pts)
    D = np.abs(P) / (np.abs(W) + eps)
    return D, P, W


def hybrid_candidates(
    x: np.ndarray,
    fvals: np.ndarray,
    sigma_pts: float = 3.0,
    eps: float = 1e-8,
    max_candidates: int | None = None,
    min_spacing: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect candidate zero basins from sampled data using the hybrid detector.

    Parameters
    ----------
    x : np.ndarray
        Sample locations.
    fvals : np.ndarray
        Sampled function values.
    sigma_pts : float
        Gaussian smoothing width in grid points.
    eps : float
        Small regularization constant in the detector denominator.
    max_candidates : int or None
        If given, return at most this many ranked candidates.
    min_spacing : float or None
        Optional minimum spacing between accepted candidates.

    Returns
    -------
    candidates, score, P, W
    """
    D, P, W = hybrid_score(x, fvals, sigma_pts=sigma_pts, eps=eps)
    idx = local_minima_indices(D)
    cands = select_candidates_by_spacing(
        x,
        D,
        idx,
        max_candidates=max_candidates,
        min_spacing=min_spacing,
    )
    return cands, D, P, W


def deduplicate_roots(roots: np.ndarray | list[float], tol: float = 1e-6) -> np.ndarray:
    """Remove near-duplicate roots from a list/array of root estimates."""
    roots = np.asarray(roots, dtype=float)
    if len(roots) == 0:
        return np.array([], dtype=float)

    roots = np.sort(roots)
    out = [roots[0]]
    for r in roots[1:]:
        if abs(r - out[-1]) > tol:
            out.append(r)
    return np.array(out, dtype=float)


def refine_candidates_newton(
    candidates: np.ndarray,
    residual_fn,
    domain: tuple[float, float] | None = None,
    maxiter: int = 20,
    tol: float = 1e-10,
    dedup_tol: float = 1e-6,
) -> np.ndarray:
    """
    Refine detector candidates using Newton iteration on a supplied residual function.

    Parameters
    ----------
    candidates : np.ndarray
        Initial guesses supplied by the detector.
    residual_fn : callable
        Scalar residual function f(x).
    domain : tuple or None
        Optional admissible interval (xmin, xmax).
    """
    roots = []
    xmin, xmax = domain if domain is not None else (-np.inf, np.inf)

    for c in np.asarray(candidates, dtype=float):
        try:
            r = newton(residual_fn, c, maxiter=maxiter, tol=tol)
            if np.isfinite(r) and xmin <= r <= xmax:
                roots.append(float(r))
        except Exception:
            pass

    return deduplicate_roots(roots, tol=dedup_tol)
