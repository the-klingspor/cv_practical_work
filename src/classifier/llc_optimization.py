import numpy as np
from numba import njit


@njit(parallel=True)
def get_llc_code(codebook, feature, size, alpha, sigma):
    """
    Broadcast the feature into a matrix that is self.size times the feature.
    Example: feature = [1, 2], self.size = 3.
    Then the result will be:
    [[1, 2],
     [1, 2],
     [1, 2]]
    Use this matrix to center the codebook around the input feature vector.
    """
    centered = codebook - feature
    covariance = np.dot(centered, centered.T)
    distance_vector = get_distances(codebook, feature, size, sigma)
    regularization_matrix = alpha * np.diag(distance_vector)
    covariance_regularized = covariance + regularization_matrix

    llc_code_not_norm = np.linalg.solve(covariance_regularized, np.ones(size))
    sum_of_llc_code = np.sum(llc_code_not_norm)
    if sum_of_llc_code != 0:
        llc_code = llc_code_not_norm / sum_of_llc_code
    else:
        llc_code = np.zeros(size)
        # todo: error handling for llc_code with sum of zero
    return llc_code


@njit(parallel=True)
def get_distances(codebook, feature, size, sigma):
    distances = np.zeros(size)
    max_distance = 0
    # get euclidean distance from feature to every visual word of the
    # codebook, save maximum distance for normalization
    for i in range(size):
        distance = np.linalg.norm(feature - codebook[i])
        max_distance = max(max_distance, distance)
        distances[i] = distance

    # normalize every vector with maximum distance and hyper parameter sigma
    distances = distances - max_distance
    distances = distances / sigma

    return np.exp(distances)
