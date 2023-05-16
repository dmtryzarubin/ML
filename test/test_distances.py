import torch
from ml.distance.functional import eucledian_dist, eucledian_dist_loops


def test_distance():
    X = torch.randn(100, 2)
    c = torch.randn(3, 2)
    dists1 = eucledian_dist(X, c)
    dists2 = eucledian_dist_loops(X, c)
    assert torch.allclose(dists1, dists2)
