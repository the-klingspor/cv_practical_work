import cv2
import numpy as np
import random

from sklearn.base import BaseEstimator, TransformerMixin

from src.classifier.llc_spatial_pyramid_encoding import LlcSpatialPyramidEncoder
from src.datautils.feature_extraction import FeatureExtraction
from src.datautils.classify_util import get_roi, print_progress_bar, print_h_m_s


class SpmTransformer(BaseEstimator, TransformerMixin):
    """
    Class which implements spatial pyramid matching as means of a Scikit-learn
    Transformer. For the transformation, an encoder attribute with an encode
    method is used. The encoders codebook will be trained with the fit method
    and the features will be transformed using its encode method.

    The BaseEstimator and TransformerMixin base classes allow easy use in a
    Scikit-learn pipeline.

    :author: Joschka Strüber
    :attr feature_extractor: default: cv2.xfeatures2d.SIFT_create()
        The feature extractor used for computing dense or sparse features on the
        images.
    :attr density: {'dense', 'sparse} : deprecated
        If the features should be computed on dense grid or sparse.
    :attr cb_train_size: default: 300
        The number of random images used for training the encoder's codebook.
    :attr codebook_size: default: 256
        The number of features that are used for the encoder's codebook.
    :attr pooling: {'max' (default), 'sum'}
        The method used for pooling the locality-constrained linear codes of
        each feature. The supported pooling methods are max pooling (default)
        and sum pooling.
    :attr normalization: {'eucl' (default), 'sum'}
        The method used for normalizing the pooled encoding. The euclidean norm
        (default) and sum normalization are supported.
    """

    def __init__(self,
                 extractor=None,
                 density='dense',
                 cb_train_size=300,
                 codebook_size=256,
                 alpha=500,
                 sigma=100,
                 pooling='max',
                 normalization='eucl'):
        self.extractor = extractor
        self.density = density
        self.cb_train_size = cb_train_size
        self.codebook_size = codebook_size
        self.alpha = alpha
        self.sigma = sigma
        self.pooling = pooling
        self.normalization = normalization

    def fit(self, X, y=None):
        """
        Fit this transformer by selecting random images from the training data
        to train the encoder's codebook.

        :author: Joschka Strüber
        :param X: The image data as a list of tuples [(path, roi), ..., ]
        :return: self
        """
        if self.density == 'sparse':
            print("Sparse density is currently not supported.\n")
            return
        if self.extractor is None:
            self.extractor = cv2.xfeatures2d.SIFT_create()
        self.feature_extractor = FeatureExtraction(self.extractor, self.density)
        self.encoder = LlcSpatialPyramidEncoder(alpha=self.alpha,
                                                sigma=self.sigma)
        n_training_data = len(X)
        # don't use more than ten percent of all images to train the codebook
        train_size = min(int(n_training_data / 10), self.cb_train_size)

        # select random images from the training data for training the code book
        training_subset = [X[i] for i in random.sample(range(n_training_data),
                                                       train_size)]
        self._train_codebook(training_subset)
        return self

    def transform(self, X):
        """
        Given a set of data  X as tuples of the image path and the region of
        interest inside this image, return the descriptors as numpy array of
        encoded features for the given encoder, codebook and hyperparameters.

        :author: Joschka Strüber
        :param X: List of tuples : [(path, roi), ..., (path, roi)]
        :return: descriptors: 2d numpy array of encoded features, for example
            llc codes
        """
        descriptors = []
        n_training_data = len(X)

        print_progress_bar(0, n_training_data)
        for index, (path, roi_coordinates) in enumerate(X):
            if index % 5 == 0 or index + 1 == n_training_data:
                print_progress_bar(index + 1, n_training_data)
            img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
            roi = get_roi(img, roi_coordinates)
            spatial_pyramid = self.feature_extractor.get_spatial_pyramid(roi)
            llc_code = self.encoder.encode(spatial_pyramid,
                                           pooling=self.pooling,
                                           normalization=self.normalization)
            descriptors.append(llc_code)
        descriptors = np.array(descriptors)
        return descriptors

    def _train_codebook(self, training_data):
        """
        Train the code book of the llc encoder by extracting features on a
        given set of images with regions of interest and using these features
        for clustering.

        :author: Joschka Strüber
        :param training_data: List of tuples : [(path, roi), ...,
                                                (path, roi)]
            Image data as paths to image files and regions of interest as
            tuples.
        :return: None
        """
        imgs = []
        for path, roi_coordinates in training_data:
            img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
            roi = get_roi(img, roi_coordinates)
            imgs.append(roi)
        feature_subset = self.feature_extractor.get_features(imgs)
        self.encoder.train_codebook(feature_subset, self.codebook_size)

