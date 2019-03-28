import numpy as np
import cv2

from skimage import feature


class LocalBinaryPatterns:
    """
    Class which computes uniform local binary patterns as histograms on images.
    The local binary pattern is a texture feature which was proposed by Ojala
    et al.'s paper "Performance evaluation of texture measures with
    classification based on Kullback discrimination of distributions".
    It can either compute the features for a set of key points with "compute" or
    compute them on a dense grid with "detectAndCompute".

    :author: Joschka Strüber
    """

    def __init__(self, n_points=8, radius=1, block_size=16):
        """
        :author: Joschka Strüber
        :param n_points: Number of circularly symmetric neighbour set points.
        :param radius: Radius of circle. radius = 1 means that only points of
            the 8-neighborhood are considered.
        :param block_size: Size of each block that is used to compute a lbp
            histogram.
        """
        self._n_points = n_points
        self._radius = radius
        self._block_size = block_size

    def compute(self, img, key_points, eps=1e-7):
        """
        Compute uniform local binary pattern histograms for the given image and
        key points.

        :author: Joschka Strüber
        :param img: 2darray:
            The grayscale image of which the lbps are computed.
        :param key_points: [cv2.KeyPoint, ...]
            The key points for which the lbp histograms are computed. Each key
            point marks the pixel in the top left of its block.
        :param eps: small epsilon for normalizing lbp histograms with sum 0.
        :return: [array, ...]
            List of numpy arrays which are lbp histograms.
        """
        lbp_img = feature.local_binary_pattern(img, self._n_points,
                                               self._radius, method='uniform')
        lbp_histograms = []
        for kp in key_points:
            col, row = kp.pt
            col = int(col)
            row = int(row)
            lbp_slice = lbp_img[col:col+self._block_size,
                                row:row+self._block_size]
            # Compute the histogram of the lbp slice. The number of bins has to
            # be n_points + 2, because there can be at most n_points + 1 uniform
            # patterns and we need another bin for all non-uniform patterns
            (hist, _) = np.histogram(lbp_slice.ravel(),
                                     bins=np.arange(0, self._n_points + 3),
                                     range=(0, self._n_points + 2))
            hist = hist.astype(np.float64)
            hist = hist / (hist.sum() + eps)
            lbp_histograms.append(hist)
        lbp_histograms = np.array(lbp_histograms)
        return key_points, lbp_histograms

    def detectAndCompute(self, img, eps=1e-7):
        """
        Compute uniform local binary pattern histograms for the given image on a
        dense grid with the given block_size.

        :author: Joschka Strüber
        :param img: 2darray:
            The grayscale image of which the lbps are computed.
        :return: [array, ...]
            List of numpy arrays which are lbp histograms.
        """
        height, width = img.shape[:2]
        key_points = [cv2.KeyPoint(x, y, self._block_size)
                      for y in range(0, height, self._block_size)
                      for x in range(0, width, self._block_size)]
        return self.compute(img, key_points, eps)

