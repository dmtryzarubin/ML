import torch
from jaxtyping import Float, Int

from ..distance.functional import eucledian_dist
from .functional import find_closest_centroid, k_means

__all__ = ["KMeans"]


class KMeans:
    _fit_called = False

    def __init__(self, n_clusters: int, random_state: int, max_iter: int = 100) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X: Float[torch.Tensor, "batch features"]) -> None:
        self.centroids = k_means(
            X, self.n_clusters, torch.manual_seed(self.random_state), self.max_iter
        )
        self._fit_called = True

    def predict(
        self, X: Float[torch.Tensor, "batch features"]
    ) -> Int[torch.Tensor, "batch"]:
        if not self._fit_called:
            raise ValueError("KMeans has not been fitted yet!")

        dists = eucledian_dist(X, self.centroids)
        labels = find_closest_centroid(dists)
        return labels
