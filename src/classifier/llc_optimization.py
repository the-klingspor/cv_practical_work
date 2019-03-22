import numpy as np
from numba import njit


#@njit(parallel=True)
def encode_spatial_bin_numba(codebook, features, size, alpha, sigma,
                             pooling='max'):
    features_f = features.astype(np.float64)
    num_features = features_f.shape[0]

    if num_features == 0:
        return np.zeros(size)
    llc_codes_iter = [get_llc_code(codebook, features_f[i], size, alpha, sigma)
                      for i in range(num_features)]
    llc_codes_unpooled = np.array(llc_codes_iter)
    if pooling == 'max':
        llc_code = np.amax(llc_codes_unpooled, axis=0)
    elif pooling == 'sum':
        llc_code = np.sum(llc_codes_unpooled, axis=0)
    else:
        raise ValueError("Invalid pooling method was chosen: {}".
                         format(pooling))
    return llc_code


@njit(parallel=True)
def get_llc_code(codebook, feature, size, alpha, sigma):
    """
    :author: Joschka Strüber
    """
    covariance_regularized = get_covariance_regularized(codebook, feature,
                                                        alpha, sigma)
    llc_code_not_norm = np.linalg.solve(covariance_regularized, np.ones(size))
    sum_of_llc_code = np.sum(llc_code_not_norm)
    if sum_of_llc_code != 0:
        llc_code = llc_code_not_norm / sum_of_llc_code
    else:
        llc_code = np.zeros(size)
        # todo: error handling for llc_code with sum of zero
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
    for i in range(size):
        distance = np.linalg.norm(feature - codebook[i])
        max_distance = max(max_distance, distance)
        distances[i] = distance

    # normalize every vector with maximum distance and hyper parameter sigma
    distances = distances - max_distance
    distances = distances / sigma

    return np.exp(distances)

