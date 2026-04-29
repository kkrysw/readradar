"""
Spherical k-means++ clustering — implemented directly in NumPy.

Used to group the 5,000 sampled books into semantic "reading neighborhoods"
from their L2-normalized search embeddings.

Algorithm:
    * Input vectors are L2-normalized so cosine similarity equals a dot
      product. The cosine distance used throughout is `1 - cosine_sim`.
    * k-means++ initialization is adapted to cosine distance: the next
      centroid is sampled with probability proportional to `d^2`, where
      `d = 1 - max_similarity_to_existing_centroids`.
    * Lloyd iterations assign each point to its most similar centroid,
      recompute each centroid as the mean of assigned vectors, and
      re-normalize the centroids to keep them on the unit sphere.
    * Empty clusters are reinitialized to the point currently farthest
      (in cosine distance) from its assigned centroid.
    * Convergence: assignments stop changing, or the max centroid shift
      drops below `tol`, or `max_iter` is reached.

Public entry point: `spherical_kmeans`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


EPS = 1e-12


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Return a row-wise L2-normalized copy of `matrix`.

    Rows with norm below `EPS` are left as zero vectors (rather than NaN).
    Float32 is used so the output is compatible with the search artifacts.
    """
    x = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe = np.maximum(norms, EPS)
    return (x / safe).astype(np.float32)


def _cosine_similarity_matrix(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between every row of `x` and every row of `centroids`.

    Both inputs are assumed to be L2-normalized, so similarity reduces to
    a single matrix multiply.

    Returns an (n, k) array.
    """
    return (x @ centroids.T).astype(np.float32)


def kmeans_plus_plus_init(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    k-means++ initialization adapted to cosine distance on the unit sphere.

    Procedure:
        1. Pick the first centroid uniformly at random.
        2. For each subsequent centroid:
            a. For every point, compute `d = 1 - max_sim_to_any_centroid`.
            b. Clamp `d` to [0, 2] (cosine distance range for unit vectors).
            c. Sample the next centroid with probability proportional to d^2.
            d. If all d^2 collapse to zero (e.g. duplicated points), fall
               back to uniform random sampling over unchosen points.
    """
    n, dim = x.shape
    if not (1 <= k <= n):
        raise ValueError(f"k={k} must satisfy 1 <= k <= n={n}")

    centroids = np.empty((k, dim), dtype=np.float32)
    chosen: list[int] = []

    first = int(rng.integers(0, n))
    centroids[0] = x[first]
    chosen.append(first)

    # Running max similarity each point has to any already-chosen centroid.
    max_sim = x @ centroids[0]

    for i in range(1, k):
        dist = np.clip(1.0 - max_sim, 0.0, 2.0).astype(np.float64)
        # Exclude already-chosen points from being re-picked.
        dist[chosen] = 0.0
        weights = dist ** 2
        total = float(weights.sum())

        if total <= 0.0 or not np.isfinite(total):
            # Degenerate: uniform random among unchosen points.
            remaining = np.setdiff1d(np.arange(n), np.asarray(chosen), assume_unique=False)
            if remaining.size == 0:
                break
            idx = int(rng.choice(remaining))
        else:
            probs = weights / total
            idx = int(rng.choice(n, p=probs))

        centroids[i] = x[idx]
        chosen.append(idx)
        # Update the running max similarity with the newly chosen centroid.
        np.maximum(max_sim, x @ centroids[i], out=max_sim)

    return centroids


def assign_clusters(
    x: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign each row of `x` to its most similar centroid under cosine similarity.

    Returns:
        labels (n,):   int32 index of the winning centroid for each point
        max_sim (n,):  float32 similarity of each point to its centroid
    """
    sims = _cosine_similarity_matrix(x, centroids)
    labels = np.argmax(sims, axis=1).astype(np.int32)
    max_sim = sims[np.arange(x.shape[0]), labels].astype(np.float32)
    return labels, max_sim


def update_centroids(
    x: np.ndarray,
    labels: np.ndarray,
    k: int,
    max_sim_to_previous: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Recompute centroids as the mean of their assigned points, then re-normalize.

    Empty-cluster handling:
        If any cluster ends up with zero assigned points, that centroid is
        reinitialized to the point that is currently *farthest* (in cosine
        distance) from its own assigned centroid, and that would not produce
        a duplicate of another reinitialized centroid. This is a standard
        robust recovery strategy; duplicates are possible only for
        pathological inputs.
    """
    n, dim = x.shape
    new_centroids = np.zeros((k, dim), dtype=np.float32)
    counts = np.zeros(k, dtype=np.int64)

    np.add.at(new_centroids, labels, x)
    np.add.at(counts, labels, 1)

    empty_mask = counts == 0
    non_empty = ~empty_mask

    # Row-wise mean for non-empty clusters.
    new_centroids[non_empty] /= counts[non_empty, None].astype(np.float32)

    if empty_mask.any():
        # Pick the globally worst-fitted points as replacement centroids.
        worst_order = np.argsort(max_sim_to_previous)  # lowest similarity first
        used: set[int] = set()
        pointer = 0
        for cluster_idx in np.flatnonzero(empty_mask):
            # Walk forward through the sorted "worst" list to find an unused index.
            while pointer < n and worst_order[pointer] in used:
                pointer += 1
            if pointer >= n:
                # Fallback: pick a random unused index.
                remaining = list(set(range(n)) - used)
                idx = int(rng.choice(remaining)) if remaining else 0
            else:
                idx = int(worst_order[pointer])
                pointer += 1
            used.add(idx)
            new_centroids[cluster_idx] = x[idx]

    return l2_normalize(new_centroids)


@dataclass
class SphericalKMeansResult:
    """Container for the output of `spherical_kmeans`."""
    labels: np.ndarray
    centroids: np.ndarray
    centroid_similarities: np.ndarray  # similarity of each point to its centroid
    inertia: float                     # sum(1 - centroid_similarity)
    inertia_history: list = field(default_factory=list)
    iterations_run: int = 0
    converged: bool = False


def spherical_kmeans(
    x: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: int = 42,
) -> SphericalKMeansResult:
    """
    Run spherical k-means++ on the rows of `x`.

    The input is defensively L2-normalized before iteration regardless of
    whether it already is, to keep the algorithm honest.

    Convergence conditions (any one stops the loop):
        1. labels unchanged between iterations;
        2. max per-centroid cosine shift below `tol`;
        3. `max_iter` reached.

    Returns a `SphericalKMeansResult`.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    rng = np.random.default_rng(seed)
    x_norm = l2_normalize(x)

    centroids = kmeans_plus_plus_init(x_norm, k, rng)
    labels, max_sim = assign_clusters(x_norm, centroids)

    inertia_history: list[float] = [float((1.0 - max_sim).sum())]
    converged = False
    iterations_run = 0

    for it in range(1, max_iter + 1):
        iterations_run = it
        new_centroids = update_centroids(
            x_norm, labels, k, max_sim, rng
        )
        # Centroid shift under cosine similarity.
        centroid_shift = float(np.max(1.0 - np.sum(new_centroids * centroids, axis=1)))
        centroids = new_centroids

        new_labels, new_max_sim = assign_clusters(x_norm, centroids)
        inertia_history.append(float((1.0 - new_max_sim).sum()))

        assignment_changed = not np.array_equal(new_labels, labels)
        labels, max_sim = new_labels, new_max_sim

        if (not assignment_changed) or (centroid_shift < tol):
            converged = True
            break

    return SphericalKMeansResult(
        labels=labels,
        centroids=centroids,
        centroid_similarities=max_sim,
        inertia=float((1.0 - max_sim).sum()),
        inertia_history=inertia_history,
        iterations_run=iterations_run,
        converged=converged,
    )


def rank_cluster_representatives(
    x: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rank each point within its own cluster by descending cosine similarity
    to the cluster centroid.

    Returns:
        similarity (n,):        cosine similarity of each point to its centroid
        cluster_rank (n,):      1-based rank within the cluster (1 = most central)
        is_representative (n,): bool, True for the top-`top_k` in each cluster
    """
    x_norm = l2_normalize(x)
    c_norm = l2_normalize(centroids)

    n = x_norm.shape[0]
    similarity = np.empty(n, dtype=np.float32)
    for i in range(n):
        similarity[i] = float(x_norm[i] @ c_norm[labels[i]])

    cluster_rank = np.zeros(n, dtype=np.int64)
    for cluster_idx in np.unique(labels):
        members = np.flatnonzero(labels == cluster_idx)
        order = members[np.argsort(-similarity[members])]
        cluster_rank[order] = np.arange(1, len(order) + 1)

    is_representative = cluster_rank <= int(top_k)
    return similarity, cluster_rank, is_representative
