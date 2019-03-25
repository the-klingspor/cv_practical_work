import cv2
import numpy as np


class FeatureExtraction:
    """
    Class which can be used to extract features as dense grids or spatial
    pyramids of images. These are the two kinds of data, which are necessary
    for spatial pyramid matching as described by Lazebnik et al.'s paper
    "Beyond bags of features: spatial pyramid matching for recognizing natural
    scene categories".

    :author: Joschka Strüber
    """

    def __init__(self, feature_extractor, max_size=None):
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
        self._feature_extractor = feature_extractor if feature_extractor is \
            not None else cv2.xfeatures2d.SIFT_create()
        self._max_size = max_size if max_size is not None else 300

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
        # collect dense key points for every image and compute features
        for im in images:
            resized = self._resize_fixed_aspect_ratio(im)
            key_points = [cv2.KeyPoint(x, y, step_size)
                          for y in range(0, resized.shape[0], step_size)
                          for x in range(0, resized.shape[1], step_size)]
            desc, dense_features = self._feature_extractor.compute(resized,
                                                                   key_points)
            feature_list.append(dense_features.astype(np.float64))
        features = np.concatenate(feature_list)
        return features

    def get_spatial_pyramid(self, image, step_size=16):
        """
        Computes dense features of an image and returns them in a spatial
        pyramid. The spatial pyramid has a size of three layers. That means the
        image will be divided into 16 parts where every one of these holds the
        feature of a level 2 bin. The four level 2 bins at the top left compose
        the first level 1 bin, the next four level 2 bins at the top right the
        second level 1 bin and so forth. All four level 1 bins together yield
        the level 0 bin.

        :author: Joschka Strüber
        :param image: The image as numpy array, which features will be computed
        as a spatial pyramid.
        :param step_size: The step size between the dense features.
        :return: [[array, array, array, array], [...], [...], [...]]
        A list that contains four list, that contain four numpy arrays of
        features each.
        """
        resized = self._resize_fixed_aspect_ratio(image)

        l1_bin_top_left = []
        l1_bin_top_right = []
        l1_bin_bottom_left = []
        l1_bin_bottom_right = []

        height, width = resized.shape[:2]
        bin_size_horizontal = int(width / 4)
        bin_size_vertical= int(height / 4)

        for row in range(4):
            for column in range(4):
                row_start = row * bin_size_vertical
                row_end = row_start + bin_size_vertical
                col_start = column * bin_size_horizontal
                col_end = col_start + bin_size_horizontal
                key_points = [cv2.KeyPoint(x, y, step_size)
                              for y in range(row_start, row_end, step_size)
                              for x in range(col_start, col_end, step_size)]
                desc, bin_features = self._feature_extractor.compute(resized,
                                                                     key_points)
                bin_features = bin_features.astype(np.float64)
                if row < 2:
                    if column < 2:
                        l1_bin_top_left.append(bin_features)
                    else:
                        l1_bin_top_right.append(bin_features)
                else:
                    if column < 2:
                        l1_bin_bottom_left.append(bin_features)
                    else:
                        l1_bin_bottom_right.append(bin_features)
        spatial_pyramid = [l1_bin_top_left, l1_bin_top_right,
                           l1_bin_bottom_left, l1_bin_bottom_right]
        return spatial_pyramid

    def _resize_fixed_aspect_ratio(self, image):
        """
        Resize the image to the maximum size allowed, while keeping the aspect
        ratio fixed.
        :author: Joschka Strüber
        :param image: The image that is to be resized.
        :return: The resized image.
        """
        height, width = image.shape[:2]
        output = image

        if height > self._max_size or width > self._max_size:
            larger_aspect = max(height, width)
            scaling_factor = self._max_size / float(larger_aspect)
            output = cv2.resize(output, None, fx=scaling_factor,
                                fy=scaling_factor,
                                interpolation=cv2.INTER_AREA)
        return output
