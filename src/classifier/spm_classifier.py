import cv2
import numpy as np

from sklearn.svm import LinearSVC

from src.classifier.llc_spatial_pyramid_encoding import LlcSpatialPyramidEncoder
from src.datautils.feature_extraction import FeatureExtraction
from src.datautils.classify_util import get_roi, map_labels_to_int


class SpmClassifier:
    """
    Class which implements spatial pyramid matching with locality-constrained
    linear coding and an SVM with linear kernel on SIFT features.

    :author: Joschka Strüber
    """

    def __init__(self,
                 extractor=cv2.xfeatures2d.SIFT_create(),
                 code_book_size=256,
                 alpha=500,
                 sigma=100,
                 pooling='max',
                 normalization='eucl',
                 C=1):
        self.feature_extractor = FeatureExtraction(extractor)
        self.code_book_size = code_book_size
        self.pooling = pooling
        self.normalization = normalization
        self.encoder = LlcSpatialPyramidEncoder(alpha=alpha, sigma=sigma)
        self.svm = LinearSVC(C=C)

    def train_codebook(self, training_data):
        """
        Train the code book of the llc encoder by extracting dense features on a
        given set of images with regions of interest and using these features
        for clustering.

        :author: Joschka Strüber
        :param training_data: List of tupels : [(path, roi, label), ...,
                                                (path, roi, label)]
            Image data as paths to image files and regions of interest as
            tupels. The labels are not needed, but part of the given data.
        :return: None
        """
        imgs = []
        for path, roi_coordinates, in training_data:
            img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
            roi = get_roi(img, roi_coordinates)
            imgs.append(roi)
        feature_subset = self.feature_extractor.get_dense_features(imgs)
        self.encoder.train_codebook(feature_subset, self.code_book_size)

    def get_descr_and_labels(self, training_data):
        """
        Given a set of trainings data as tupels of the image path, the region of
        interest inside this image and the label as string, get the descriptors
        as numpy array of llc features and labels as ints.

        :author: Joschka Strüber
        :param training_data: List of tupels : [(path, roi, label), ...,
                                                (path, roi, label)]
        :return: descriptors: 2d numpy array of llc codes
                 labels: the labels for all encoded features
        """
        descriptors = []
        str_labels = []

        for path, roi_coordinates, label in training_data:
            img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
            roi = get_roi(img, roi_coordinates)
            spatial_pyramid = self.feature_extractor.get_spatial_pyramid(roi)
            llc_code = self.encoder.encode(spatial_pyramid,
                                           pooling=self.pooling,
                                           normalization=self.normalization)
            descriptors.append(llc_code)
            str_labels.append(label)
        descriptors = np.array(descriptors)
        labels = map_labels_to_int(self, str_labels)
        return descriptors, labels

    def fit(self, X, y):
        self.svm.fit(X, y)

    def predict(self, X):
        return self.svm.predict(X)

    def score(self, X, y):
        return self.svm.score(X, y)

