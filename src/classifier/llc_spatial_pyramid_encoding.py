import numpy as np
import multiprocessing as mp

from sklearn.cluster import MiniBatchKMeans


class LlcSpatialPyramidEncoder:
    """
    Class which computes locality-constrained linear codes for a codebook of
    features and a set of features, that have to be encoded. The features must
    be in a spatial pyramid structure. The encoding algorithm is based on Wang
    et als. publication "Locality-constrained Linear Coding for Image
    Classification" (2010 CVPR).

    :author: Joschka Strüber
    """

    def __init__(self, size=None, codebook=None, alpha=500, sigma=100):
        """
        :author: Joschka Strüber
        :param size: int
        The number of features in the codebook.
        :param codebook: ndarray
        Set of features in a matrix, which is a good representation of the
        feature space. Usually, this will be trained based on a larger set of
        features, but a precomputed codebook can be used as well.
        :param alpha: float (default = 500)
        Hyperparameter for regularization, which affects how much choosing
        non-local bases of the codebook is penalized.
        :param sigma: float (default = 100)
        Hyperparameter for regularization, which is used for adjusting the
        weight decay speed of the locality adaptor.
        """
        self._size = size
        self._codebook = codebook
        self._alpha = alpha
        self._sigma = sigma

    def train_codebook(self, features, size=256):
        """
        Trains a codebook as the cluster centers for a larger set of features.
        k-means is the used clustering algorithm.
        :author: Joschka Strüber
        :param size: unsigned int (default 1024)
        Number of cluster centers for the codebook.
        :param features: ndarray
        ndarray of features, which will be clustered for the codebook.
        :return: None
        """
        # size of codebook must be maximum of number of features and wanted size
        if features.shape[0] >= size:
            self._size = size
        else:
            self._size = features.shape[0]

        mini_batch_kMeans = MiniBatchKMeans(n_clusters=self._size)
        k_means_clusters = mini_batch_kMeans.fit(X=features)
        self._codebook = k_means_clusters.cluster_centers_

    def encode(self, spatial_pyramid, pooling='max', normalization='eucl'):
        """
        Computes a locality-constrained linear code for a spatial pyramid of
        feature vectors based on the classes codebook using the specified
        pooling and normalization methods.
        For each of the 21 spatial bins (level 0: 1, level 1: 4, level 2: 16) a
        pooled and normalized code will be computed. These 21 codes are
        concatenated and returned.

        :author: Joschka Strüber
        :param spatial_pyramid: ndarray
        A spatial pyramid of feature vectors, which is three layers deep. It is
        a nested list of four lists of four  numpy arrays each of features. C.f.
        Lazebnik et al.'s publication "Beyond bags of features: spatial pyramid
        matching for recognizing natural scene categories" (2006 CVPR).
        Each of the 16 most inner arrays holds the features of a level 2 spatial
        bin. A level 1 spatial bin consists of all features of all level 2 bins
        that are part of the same inner list. The level 0 spatial bin consists
        of all features.
        Example:
        [[f0, f1, f2, f3], ..., [f12, f13, f14, f15]]
        The array of features f0 is a level 2 spatial bin, {f0, ..., f3} is a
        level 1 spatial bin and {f0, ..., f15} is the level 0 spatial bin.
        :param pooling: {'max' (default), 'sum'}
        The method used for pooling the locality-constrained linear codes of
        each feature. The supported pooling methods are max pooling (default)
        and sum pooling.
        :param normalization: {'eucl' (default), 'sum'}
        The method used for normalizing the pooled encoding. The euclidean norm
        (default) and sum normalization are supported.
        :return: ndarray, dtype=float64
        The normalized concatenation of the pooled codes for all 21 spatial
        bins. The first part corresponds to the level 0 bin, followed by level
        1, followed by level 2. In case of an error, a ValueError will be
        raised.
        """
        if self._codebook is None:
            print("No code book was given or trained.")
            return

        # index 0: level 0 bin; 1-4: level 1 bins; 5-20: level 2 bins
        spm_code = np.zeros((21, self._size))

        # encode all features of all level 2 bins
        feature_list = []
        for l1_bin in range(4):
            for l2_bin in range(4):
                feature_list.append(spatial_pyramid[l1_bin][l2_bin])
        pool = mp.Pool(processes=16)
        l2_codes = [pool.apply(self._encode_spatial_bin,
                               args=(features, pooling)) for features
                    in feature_list]
        # skip the l0 and the four l1 codes
        start_index = 5
        for l2_index in range(16):
            spm_code[l2_index + start_index] = l2_codes[l2_index]

        # use associativity of pooling methods to compute pooled codes for l1
        # bins and l0 bin
        if pooling == 'max':
            for l1_index in range(1, 5):
                start_index = 5 + 4 * (l1_index - 1)
                spm_code[l1_index] = spm_code[start_index]
                for l2_bin in range(1, 4):
                    spm_code[l1_index] = np.maximum(spm_code[l1_index],
                                                spm_code[start_index + l2_bin])
            spm_code[0] = spm_code[1]
            for l1_bin in range(2,5):
                spm_code[0] = np.maximum(spm_code[0], spm_code[l1_bin])
        elif pooling == 'sum':
            for l1_index in range(1, 5):
                start_index = 5 + 4 * l1_index
                spm_code[l1_index] = spm_code[start_index]
                for l2_bin in range(1, 4):
                    spm_code[l1_index] += spm_code[start_index + l2_bin]
            spm_code[0] = spm_code[1]
            for l1_bin in range(2, 5):
                spm_code[0] += spm_code[l1_bin]
        else:
            raise ValueError("Invalid pooling method was chosen: {}".
                             format(pooling))
        spm_code = spm_code.ravel()

        # normalization
        if normalization == 'eucl':
            spm_code = spm_code / np.linalg.norm(spm_code)
        elif normalization == 'sum':
            spm_code = spm_code / np.sum(spm_code)
        else:
            raise ValueError("Invalid normalization method was chosen: {}".
                             format(normalization))

        return spm_code

    def _encode_spatial_bin(self, features, pooling='max'):
        """
        Computes the LLC codes for a set of features and pools them with the
        specified pooling method. In case of an empty set, a zero vector will be
        returned.
        """
        num_features = features.shape[0]

        if num_features == 0:
            return np.zeros(self._size)
        llc_codes_iter = [self._get_llc_code(features[i])
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

    def _get_llc_code(self, feature):
        """
        Computes an LLC code based on the feature vector and the codebook.
        """

        """
        Broadcast the feature into a matrix that is self.size times the feature.
        Example: feature = [1, 2], self.size = 3.
        Then the result will be:
        [[1, 2],
         [1, 2],
         [1, 2]]
        Use this matrix to center the codebook around the input feature vector.
        """
        centered = self._codebook - np.broadcast_to(feature, (self._size,
                                                              feature.shape[0]))
        covariance = np.dot(centered, centered.T)
        regularization_matrix = self._alpha * np.diag(
            self._get_distance_vector(feature))
        covariance_regularized = covariance + regularization_matrix

        # check if the regularized covariance matrix is singular
        try:
            llc_code_not_norm = np.linalg.solve(covariance_regularized,
                                                np.ones(self._size))
        except np.linalg.LinAlgError:
            llc_code_not_norm = np.linalg.lstsq(covariance_regularized,
                                                np.ones(self._size), rcond=None)[0]
        sum = np.sum(llc_code_not_norm)
        if sum != 0:
            llc_code = llc_code_not_norm / sum
        else:
            llc_code = np.zeros(self._size)
            # todo: error handling for llc_code with sum of zero
        return llc_code

    def _get_distance_vector(self, feature):
        """
        Computes a normalized distance vector from a feature to every visual
        word of the codebook.
        """
        distances = np.zeros(self._size)
        max_distance = 0
        # get euclidean distance from feature to every visual word of the
        # codebook, save maximum distance for normalization
        for i in range(self._size):
            distance = np.linalg.norm(feature - self._codebook[i])
            max_distance = max(max_distance, distance)
            distances[i] = distance

        # normalize every vector with maximum distance and hyperparameter sigma
        distances = distances - max_distance
        distances = distances / self._sigma

        return np.exp(distances)

    def _is_invertible(self, matrix):
        """
        Checks if a matrix is invertible and can be used for solving linear
        equations
        """
        square = (matrix.shape[0] == matrix.shape[1])
        full_rank = False
        if square:
            full_rank = (np.linalg.matrix_rank(matrix) == matrix.shape[0])
        return square and full_rank


