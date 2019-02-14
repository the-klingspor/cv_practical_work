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

    def train_codebook(self, size, features):
        """
        Trains a codebook as the cluster centers for a larger set of featurse.
        k-means is the used clustering algorithm.
        :author: Joschka Strüber
        :param size: Number of cluster centers for the codebook.
        :param features: ndarray of features, which will be clustered for the
        codebook.
        :return: None
        """
        self.size = size
        kMeans = k_means(n_clusters=size)
        kMeans.fit(features)
        codebook = kMeans.cluster_centers_
