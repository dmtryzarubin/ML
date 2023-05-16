import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

__all__ = ["eucledian_dist_loops", "eucledian_dist"]


@jaxtyped
@typechecker
def eucledian_dist_loops(
    X: Float[torch.Tensor, "batch features"],
    centroids: Float[torch.Tensor, "n_clusters features"],
) -> Float[torch.Tensor, "batch n_clusters"]:
    dists = torch.ones(X.shape[0], centroids.shape[0])
    for i, row in enumerate(X):
        for j, centroid in enumerate(centroids):
            dists[i, j] = F.pairwise_distance(row, centroid, p=2.0, eps=0.0)
    return dists


@jaxtyped
@typechecker
def eucledian_dist(
    X: Float[torch.Tensor, "batch features"],
    centroids: Float[torch.Tensor, "n_clusters features"],
) -> Float[torch.Tensor, "batch n_clusters"]:
    # a^2 - 2ab + b^2
    dists = (
        (X**2).sum(1, keepdim=True)
        - (2 * X @ centroids.T)
        + (centroids**2).sum(1, keepdim=True).T
    )
    if (dists < 0.0).any():
        # If tensor contain negative distances, check that they are close to 0.0
        assert torch.allclose(dists, torch.zeros_like(dists))
        # Fill negative values with 0.0
        dists = torch.nan_to_num(dists, nan=0.0)
    return torch.sqrt(dists)
