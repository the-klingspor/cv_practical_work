import unittest
import numpy as np

from src.llc_spatial_pyramid_encoding import LlcSpatialPyramidEncoder


class LlcSpatialPyramidEncoderTest(unittest.TestCase):
    def setUp(self):
        cb = np.array([[2, 1],
                       [-1, 2],
                       [0, -1]])
        self.encoder = LlcSpatialPyramidEncoder(size=3, codebook=cb, alpha=2,
                                                sigma=3)

    def test_train_codebook(self):
        features = np.array([[0, 0],
                             [1, 1],
                             [3, 3],
                             [5, 5],
                             [2, 0],
                             [1, -1]])
        self.encoder.train_codebook(features, 2)

        expected_codebook_1 = np.array([[1, 0], [4, 4]])
        expected_codebook_2 = np.array([[4, 4], [1, 0]])

        permutation_1 = (self.encoder.codebook == expected_codebook_1).all()
        permutation_2 = (self.encoder.codebook == expected_codebook_2).all()

        print(self.encoder.codebook)
        self.assertTrue(permutation_1 or permutation_2)


    def test_get_llc_code(self):
        self.assertTrue(True)

    def test_encode_spatial_bin(self):
        self.assertTrue(True)

    def test_encode(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
