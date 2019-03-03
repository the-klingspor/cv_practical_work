import cv2

from sklearn.svm import LinearSVC

from src.llc_spatial_pyramid_encoding import LlcSpatialPyramidEncoder
from src.feature_extraction import FeatureExtraction
from src.datautils.classify_util import get_roi, map_labels_to_int


class SpmClassifier:
    """
    Class which implements spatial pyramid matching with locality-constrained
    linear coding and an SVM with linear kernel on SIFT features.

    :author: Joschka Strüber
    """

    def __init__(self, pooling='max', normalization='eucl'):
        sift = cv2.xfeatures2d.SIFT_create()
        self.feature_extractor = FeatureExtraction(sift)
        self.pooling = pooling
        self.normalization = normalization
        self.encoder = LlcSpatialPyramidEncoder()
        self.svm = LinearSVC()

    def train_codebook(self, imgs):
        feature_subset = self.feature_extractor.get_dense_features(imgs)
        self.encoder.train_codebook(feature_subset)

    def get_descr_and_labels(self, training_data):
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
        labels = map_labels_to_int(self, str_labels)
        return descriptors, labels

    def fit(self, X, y):
        self.svm.fit(X, y)

    def predict(self, X):
        return self.svm.predict(X)

    def score(self, X, y):
        return self.svm.score(X, y)

