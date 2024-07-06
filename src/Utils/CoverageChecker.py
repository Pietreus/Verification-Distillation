import numpy as np
import torch
from scipy.spatial import KDTree


def chernoff_guarantee_eps(x, eta_error_probability):
    """
    Returns the epsilon error for the coverage estimate, given a dataset, a delta and a error probability
    :param x: dataset
    :param delta_radius: robustness radius of coverage
    :param eta_error_probability: allowed probability that the given bound does not hold
    :return: the chernoff bound for parameter estimation reexpressed for eps
    """
    n = torch.shape(x)[0]
    eps = np.sqrt(3/n * np.log(2/eta_error_probability))
    return eps


class CoverageChecker:
    """
    This class takes a dataset, and for a second dataset, tells you the fraction of points which are covered with respect
    to some delta radius and with some distance metric p
    """
    def __init__(self, data, **kwargs):
        self.kdtree = KDTree(data, **kwargs)

    def query(self, x, delta_radius, p=np.inf):
        dist, _ = self.kdtree.query(x, k=1, p=p, distance_upper_bound=delta_radius)
        return dist

    def getCoverage(self, x, delta_radius, p=np.inf):
        return np.mean(self.query(x, delta_radius,p) < np.inf)


if __name__ == "__main__":
    random_matrix = torch.tensor(np.random.uniform(0, 1, size=(10 ** 6, 5)), dtype=torch.float32)

    query_matrix = torch.tensor(np.random.uniform(0, 1, size=(10 ** 6, 5)), dtype=torch.float32)
    print(query_matrix)
    checker = CoverageChecker(random_matrix)
    print(checker.query(query_matrix, 0.01))
    print(checker.getCoverage(query_matrix, 0.01))

    print("perfect tiling")
    print(10**6/(1/0.02)**5)

    print(chernoff_guarantee_eps(query_matrix, eta_error_probability=0.001))
    # results agree :)
