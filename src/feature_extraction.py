import cv2
import numpy as np


class FeatureExtraction:
    """
    Class which can be used to extract features for random subsets or spatial
    pyramids of images. These are the two kinds of data, which are necessary
    for spatial pyramid matching as described by Lazebnik et al.'s paper
    "Beyond bags of features: spatial pyramid matching for recognizing natural
    scene categories".

    :author: Joschka Strüber
    """

    def __init__(self, feature_extractor):
        """
        :author: Joschka Strüber
        :param feature_extractor: An object with a method compute(image,
        keypoints[]) that computes a set of features of an image for a given set
        of key points. An example for this is cv2.xfeatures2d.SIFT from OpenCV.
        """
        self._feature_extractor = feature_extractor

    def get_features(self, images, step_size=16):
        """
        Compute a dense set of features for the given step size for all images.
        :author: Joschka Strüber
        :param images: A list of images as numpy arrays. They have to be single
        channel,  grayscale images.
        :return: ndarray
        A 2d array of features where every row is a feature.
        """
        feature_list = []
        # collect dense keypoints for every image and compute features
        for im in images:
            key_points = [cv2.KeyPoint(x, y, step_size)
                          for y in range(0, im.shape[0], step_size)
                          for x in range(0, im.shape[1], step_size)]
            desc, dense_features = self._feature_extractor.compute(im,
                                                                   key_points)
            feature_list.append(dense_features)
        features = np.concatenate(feature_list)
        return features

    def get_spatial_pyramid(self, image, stepsize=16):
        return
