from sklearn.cluster import k_means
from itertools import chain as chain
import numpy as np


class LlcSpatialPyramidEncoder:
    """
    Class which computes locality-constrained linear codes for a codebook of
    features and a set of features, that have to be encoded. The features must
    be in a spatial pyramid structure. The encoding algorithm is based on Wang
    et als. publication "Locality-constrained Linear Coding for Image
    Classification" (2010 CVPR).

    :author: Joschka Strüber
    """

    def __init__(self, size=None, codebook=None, alpha=1, sigma=1):
        """
        :author: Joschka Strüber
        :param size: int
        The number of features in the codebook.
        :param codebook: ndarray
        Set of features in a matrix, which is a good representation of the
        feature space. Usually, this will be trained based on a larger set of
        features, but a precomputed codebook can be used as well.
        :param alpha: float (default = 1)
        Hyperparameter for regularization, which affects how much choosing
        non-local bases of the codebook is penalized.
        :param sigma: float (default = 1)
        Hyperparameter for regularization, which is used for adjusting the
        weight decay speed of the locality adaptor.
        """
        self.size = size
        self.codebook = codebook
        self.alpha = alpha
        self.sigma = sigma

    def train_codebook(self, features, size=1024):
        """
        Trains a codebook as the cluster centers for a larger set of features.
        k-means is the used clustering algorithm.
        :author: Joschka Strüber
        :param size: Number of cluster centers for the codebook.
        :param features: ndarray of features, which will be clustered for the
        codebook.
        :return: None
        """
        self.size = size
        kMeans = k_means(n_clusters=size).fit(features)
        self.codebook = kMeans.cluster_centers_

    def encode(self, spatial_pyramid, pooling='max', normalization='eucl'):
        """
        Computes a locality-constrained linear code for a spatial pyramid of
        feature vectors based on the classes codebook using the specified
        pooling and normalization methods.
        For each of the 21 spatial bins (level 0: 1, level 1: 4, level 2: 16) a
        pooled and normalized code will be computed. These 21 codes are
        concatenated and returned.

        :author: Joschka Strüber
        :param spatial_pyramid: A spatial pyramid of feature vectors, which is
        three layers deep. It is a nested list of 4 lists of 4 lists each of
        features. C.f. Lazebnik et al.'s publication "Beyond bags of features:
        spatial pyramid matching for recognizing natural scene categories" (2006
        CVPR).
        Each of the 16 most inner lists holds the features of a level 2 spatial
        bin. A level 1 spatial bin consists of all features of all level 2 bins
        that are part of the same inner list. The level 0 spatial bin consists
        of all features.
        Example:
        [[f0, f1, f2, f3], ..., [f12, f13, f14, f15]]
        The list of features f0 is a level 2 spatial bin, {f0, ..., f3} is a
        level 1 spatial bin and {f0, ..., f15} is the level 0 spatial bin.
        :param pooling: {'max' (default), 'sum'}
        The method used for pooling the locality-constrained linear codes of
        each feature. The supported pooling methods are max pooling (default)
        and sum pooling.
        :param normalization: {'eucl' (default), 'manh'}
        The method used for normalizing the pooled encoding. The euclidean norm
        (default) and manhattan norm are supported.
        :return: array of doubles
        The concatenation of the pooled and normalized codes for all 21 spatial
        bins. The first part corresponds to the level 0 bin, followed by level
        1, followed by level 2. In case of an error, None will be returned.
        """
        codes = []
        # flatten nested spatial pyramid and compute code
        level0_code = self._encode_spatial_bin(list(chain.from_iterable(
            chain.from_iterable(spatial_pyramid))), pooling, normalization)
        codes.append(level0_code)

        for level1_spatial_bin in spatial_pyramid:
            level1_code = self._encode_spatial_bin(list(chain.from_iterable(
                level1_spatial_bin)), pooling, normalization)
            codes.append(level1_code)

        for level1_spatial_bin in spatial_pyramid:
            for level2_spatial_bin in level1_spatial_bin:
                level2_code = self._encode_spatial_bin(level2_spatial_bin,
                                                       pooling, normalization)
                codes.append(level2_code)

        spm_code = np.concatenate(codes).ravel()
        return spm_code


    def _encode_spatial_bin(self, features, pooling='max',
                            normalization='eucl'):
        """

        :param features:
        :param pooling:
        :param normalization:
        :return:
        """
        return 0

    def _get_llc_code(self, feature):
        """

        :param feature:
        :return:
        """
        return 0


