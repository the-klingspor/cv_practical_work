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
        keypoints[]) that computes a dense set of features of an image for a
        given set of keypoints. An example for this is cv2.xfeatures2d.SIFT
        from OpenCV.
        Fei-Fei and Perona
        """
        self._feature_extractor = feature_extractor

    def get_features(self, images):
        """
        :author: Joschka Strüber
        :param images:
        :return:
        """
        return

    def get_spatial_pyramid(self, image, stepsize=16):
        return
