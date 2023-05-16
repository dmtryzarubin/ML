import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped

from ..distance.functional import eucledian_dist

__all__ = [
    "k_means",
    "_k_means",
    "recalculate_centroids",
    "find_closest_centroid",
    "get_weights",
    "normalize_weights",
    "plus_plus_init_centroids",
    "init_centroids",
]


@jaxtyped
@typechecker
def k_means(
    X: Float[torch.Tensor, "batch features"],
    n_clusters: int,
    generator: torch.Generator,
    max_iter: int = 100,
) -> Float[torch.Tensor, "n_clusters features"]:
    assert max_iter > 0
    centroids = plus_plus_init_centroids(X, n_clusters, generator)
    for _ in range(max_iter):
        centroids = _k_means(X, centroids)
    return centroids


@jaxtyped
@typechecker
def _k_means(
    X: Float[torch.Tensor, "batch features"],
    centroids: Float[torch.Tensor, "n_clusters features"],
) -> Float[torch.Tensor, "n_clusters features"]:
    dists = eucledian_dist(X, centroids)
    labels = find_closest_centroid(dists)
    centroids = recalculate_centroids(X, centroids, labels)
    return centroids


@jaxtyped
@typechecker
def recalculate_centroids(
    X: Float[torch.Tensor, "batch features"],
    centroids: Float[torch.Tensor, "n_clusters features"],
    labels: Int[torch.Tensor, "batch"],
) -> Float[torch.Tensor, "n_clusters features"]:
    new_centroids = centroids.clone()
    for i in range(len(centroids)):
        new_centroids[i] = X[labels == i].mean(dim=0)
    return new_centroids


@jaxtyped
@typechecker
def plus_plus_init_centroids(
    X: Float[torch.Tensor, "batch features"],
    n_clusters: int,
    generator: torch.Generator,
) -> Float[torch.Tensor, "n_clusters features"]:
    idxs = [torch.randint(len(X), (1,), generator=generator).item()]
    for _ in range(1, n_clusters):
        cur_centers = X[idxs]
        dists = eucledian_dist(X, cur_centers)
        closest = find_closest_centroid(dists)
        weights = get_weights(dists, closest)
        # Normalize weights so weights.sum() = 1.0
        p = normalize_weights(weights)
        idx = torch.multinomial(p, num_samples=1, generator=generator).item()
        idxs.append(idx)
    return X[idxs]


@jaxtyped
@typechecker
def init_centroids(
    X: Float[torch.Tensor, "batch features"],
    n_clusters: int,
    generator: torch.Generator,
) -> Float[torch.Tensor, "n_clusters features"]:
    idxs = torch.randint(len(X), (n_clusters,), generator=generator)
    return X[idxs]


@jaxtyped
@typechecker
def find_closest_centroid(
    dists: Float[torch.Tensor, "batch features"]
) -> Int[torch.Tensor, "batch"]:
    return dists.argmin(1)


@jaxtyped
@typechecker
def get_weights(
    dists: Float[torch.Tensor, "batch n_clusters"],
    closest: Int[torch.Tensor, "batch"],
) -> Float[torch.Tensor, "batch"]:
    weights = dists[range(len(dists)), closest]
    return weights**2


@jaxtyped
@typechecker
def normalize_weights(
    weights: Float[torch.Tensor, "batch"]
) -> Float[torch.Tensor, "batch"]:
    # Keep zeros after softmax
    weights[weights == 0.0] = -torch.inf
    return torch.softmax(weights, dim=0)
