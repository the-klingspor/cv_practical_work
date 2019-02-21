import cv2
import numpy as np
import time

from src.llc_spatial_pyramid_encoding import LlcSpatialPyramidEncoder

class FeatureExtraction:
    """
    Class which can be used to extract features as dense grids or spatial
    pyramids of images. These are the two kinds of data, which are necessary
    for spatial pyramid matching as described by Lazebnik et al.'s paper
    "Beyond bags of features: spatial pyramid matching for recognizing natural
    scene categories".

    :author: Joschka Strüber
    """

    def __init__(self, feature_extractor, max_size=300):
        """
        :author: Joschka Strüber
        :param feature_extractor: An object with a method compute(image,
        keypoints[]) that computes a set of features of an image for a given set
        of key points. An example for this is cv2.xfeatures2d.SIFT from OpenCV.
        :param max_size: int (default = 300)
        The maximum size of the image prior to feature extraction. If an image
        is too large, it will be resized to the maximum size with the same
        aspect ratio as before.
        """
        self._feature_extractor = feature_extractor
        self._max_size = max_size

    def get_dense_features(self, images, step_size=16):
        """
        Compute a dense set of features for the given step size for all images.
        :author: Joschka Strüber
        :param images: A list of images as numpy arrays. They have to be single
        channel,  grayscale images.
        :param step_size: int (default = 16)
        The step size between the features in x and y directions.
        :return: ndarray
        A 2d array of features where every row is a feature.
        """
        feature_list = []
        # collect dense keypoints for every image and compute features
        for im in images:
            resized = self._resize_fixed_aspect_ratio(im)
            key_points = [cv2.KeyPoint(x, y, step_size)
                          for y in range(0, resized.shape[0], step_size)
                          for x in range(0, resized.shape[1], step_size)]
            desc, dense_features = self._feature_extractor.compute(resized,
                                                                   key_points)
            feature_list.append(dense_features)
        features = np.concatenate(feature_list)
        return features

    def get_spatial_pyramid(self, image, step_size=16):
        """

        :param image:
        :param step_size:
        :return:
        """
        height, width = image.shape[:2]

        return

    def _resize_fixed_aspect_ratio(self, image):
        """
        Resize the image to the maximum size allowed, while keeping the aspect
        ratio fixed.
        :param image: The image that is to be resized.
        :return: The resized image.
        """
        height, width = image.shape[:2]
        output = image

        if height > self._max_size or width > self._max_size:
            larger_aspect = max(height, width)
            scaling_factor = self._max_size / float(larger_aspect)
            output = cv2.resize(output, None, fx=scaling_factor,
                                fy=scaling_factor, interpolation=cv2.INTER_AREA)
        return output


if __name__ == '__main__':
    start = time.clock()
    image1 = cv2.imread("/home/joschi/Pictures/pangolin.jpg", 1)
    image2 = cv2.imread("/home/joschi/Pictures/pangolin2.jpg", 1)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    image_list = [gray1, gray2]

    feature_extraction = FeatureExtraction(cv2.xfeatures2d.SIFT_create())

    dense_features = feature_extraction.get_dense_features(image_list)
    print(dense_features.shape[0])

    encoder = LlcSpatialPyramidEncoder()
    start = time.clock()
    encoder.train_codebook(dense_features)
    end = time.clock()
    print (end - start)

