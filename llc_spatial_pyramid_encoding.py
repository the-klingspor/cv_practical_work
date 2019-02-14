from sklearn.cluster import k_means
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

    def __init__(self, size=None, codebook=None):
        """
        :author: Joschka Strüber
        :param size: The number of features in the codebook.
        :param codebook: Set of features in a matrix, which is a good
        representation of the feature space. Usually, this will be trained based
        on a larger set of features, but a precomputed codebook can be used as
        well.
        """
        self.size = size
        self.codebook = codebook

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
        bins. 
        """
