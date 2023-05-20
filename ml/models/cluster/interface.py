import torch
from jaxtyping import Float, Int

from ...distance.functional import eucledian_dist
from ..base_model import Model
from .functional import find_closest_centroid, k_means

__all__ = ["KMeans"]


class KMeans(Model):
    def __init__(self, n_clusters: int, random_state: int, max_iter: int = 100) -> None:
        """
        Implements K-Means clusterization algorithm

        :param n_clusters: number of clusters to inspect
        :param random_state: random_state for reproducebility
        :param max_iter: number of iterations of centroids recalculation, defaults to 100
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, input: Float[torch.Tensor, "batch features"]) -> None:
        """
        Searches for best centroids

        :param input: Input data to clusterize
        """
        super().fit(input)
        self._parameters = [
            k_means(
                input,
                self.n_clusters,
                torch.manual_seed(self.random_state),
                self.max_iter,
            )
        ]

    def predict(
        self, input: Float[torch.Tensor, "batch features"]
    ) -> Int[torch.Tensor, "batch"]:
        """
        Assigns cluster label to each point

        :return: 1-d tensor with cluster labels
        """
        super().predict(input)
        dists = eucledian_dist(input, *self._parameters)
        labels = find_closest_centroid(dists)
        return labels
