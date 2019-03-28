import numpy as np
from numba import njit, prange


@njit
def encode_spatial_bin_numba(codebook, features, size, alpha, sigma,
                             pooling='max'):
    """
    Encode all features of a spatial bin for a given codebook and
    hyperparameters. Numba optimised numpy is used.
    :author: Joschka Strüber
    """
    num_features = features.shape[0]
    # start = time.time()
    if num_features == 0:
        return np.zeros(size)

    llc_codes_unpooled = np.empty((num_features, size))
    for i in prange(num_features):
        llc_codes_unpooled[i] = get_llc_code(codebook, features[i], size,
                                             alpha, sigma)
    if pooling == 'max':
        llc_code = amax(llc_codes_unpooled)
    elif pooling == 'sum':
        llc_code = np.sum(llc_codes_unpooled, axis=0)
    else:
        raise ValueError("Invalid pooling method was chosen.")
    return llc_code


@njit(parallel=True)
def get_llc_code(codebook, feature, size, alpha, sigma, eps=1e-7):
    """
    Compute a single llc code for a feature.
    :author: Joschka Strüber
    """
    covariance_regularized = get_covariance_regularized(codebook, feature,
                                                        alpha, sigma)
    llc_code_not_norm = np.linalg.solve(covariance_regularized, np.ones(size))
    sum_of_llc_code = np.sum(llc_code_not_norm)
    llc_code = llc_code_not_norm / (sum_of_llc_code + eps)
    return llc_code


@njit(parallel=True)
def get_covariance_regularized(codebook, feature, alpha, sigma):
    """
    Compute a distance regularized covariance matrix for a given feature,
    codebook and the hyperparameters alpha and sigma.
    :author: Joschka Strüber
    """
    # broadcasting applies: subtract feature from every row of the codebook
    centered = codebook - feature
    covariance = np.dot(centered, centered.T)

    # add distance value to the diagonal of the covariance matrix
    distance_vector = get_distances(codebook, feature, sigma)
    regularization_matrix = alpha * np.diag(distance_vector)
    covariance_regularized = covariance + regularization_matrix
    return covariance_regularized


@njit(parallel=True)
def get_distances(codebook, feature, sigma):
    """
    Get a regularized distance vector from the feature to every vector of the
    codebook.
    :author: Joschka Strüber
    """
    size = len(codebook)
    distances = np.zeros(size)
    max_distance = 0
    # get euclidean distance from feature to every visual word of the
    # codebook, save maximum distance for normalization
    for i in prange(size):
        distance = np.linalg.norm(feature - codebook[i])
        max_distance = max(max_distance, distance)
        distances[i] = distance

    # normalize every vector with maximum distance and hyper parameter sigma
    distances = distances - max_distance
    distances = distances / sigma

    return np.exp(distances)


@njit
def amax(features):
    """
    Compute the elementwise maximum for each column of a 2d-array. This
    implementation is necessary because numba currently does not support
    numpy.amax.
    :author: Joschka Strüber
    """
    num_features = features.shape[0]
    if num_features == 0:
        return None
    max_row = features[0]
    for i in range(1, num_features):
        max_row = np.maximum(max_row, features[i])
    return max_row
