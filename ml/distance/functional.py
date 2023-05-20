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
    """
    Used for checking correctness.
    Calculates L2 norm between 2 matrices.
    Uses `torch.nn.functional.pairwise_distance`.

    :return: Distance matrix
    """
    dists = torch.ones(X.shape[0], centroids.shape[0])
    for i, row in enumerate(X):
        for j, centroid in enumerate(centroids):
            dists[i, j] = F.pairwise_distance(row, centroid, p=2.0, eps=0.0)
    return dists


@jaxtyped
@typechecker
def eucledian_dist(
    input1: Float[torch.Tensor, "batch1 features"],
    input2: Float[torch.Tensor, "batch2 features"],
) -> Float[torch.Tensor, "batch1 batch2"]:
    """
    Calculates L2 norm between 2 matrices.
    For each row in `input1` calculates distance
    between it and each row in `input2`.
    If `input1` has shape {batch1, features} and `input2` has shape {batch2, features}
    the output will be of shape {batch1, batch2}

    :return: Distance matrix
    """
    # (a - b) ^ 2 = a^2 - 2ab + b^2
    dists = (
        (input1**2).sum(1, keepdim=True)
        - (2 * input1 @ input2.T)
        + (input2**2).sum(1, keepdim=True).T
    )
    if (dists < 0.0).any():
        # If tensor contain negative distances, check that they are close to 0.0
        assert torch.allclose(dists, torch.zeros_like(dists))
        # Fill negative values with 0.0
        dists = torch.nan_to_num(dists, nan=0.0)
    return torch.sqrt(dists)
