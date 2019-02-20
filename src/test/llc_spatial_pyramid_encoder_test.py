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

        permutation_1 = (self.encoder._codebook == expected_codebook_1).all()
        permutation_2 = (self.encoder._codebook == expected_codebook_2).all()

        self.assertTrue(permutation_1 or permutation_2)

    def test_get_distance_vector(self):
        result = self.encoder._get_distance_vector(np.array([3, 0]))
        expected = np.array([0.36084, 1, 0.64622])
        self.assertArrayAlmostEqual(result, expected)

        result = self.encoder._get_distance_vector(np.array([-1, -1]))
        expected = np.array([1, 0.81722, 0.41957])
        self.assertArrayAlmostEqual(result, expected)

        result = self.encoder._get_distance_vector(np.array([1, -1]))
        expected = np.array([0.63350, 1, 0.41957])
        self.assertArrayAlmostEqual(result, expected)

        result = self.encoder._get_distance_vector(np.array([1, 2]))
        expected = np.array([0.55840, 0.6788, 1])
        self.assertArrayAlmostEqual(result, expected)

    def test_get_llc_code(self):
        result = self.encoder._get_llc_code(np.array([3, 0]))
        expected = np.array([1.11015, -0.344779, 0.23463])
        self.assertArrayAlmostEqual(result, expected)

        result = self.encoder._get_llc_code(np.array([-1, -1]))
        expected = np.array([-0.20915, 0.22858, 0.98056])
        self.assertArrayAlmostEqual(result, expected)

        result = self.encoder._get_llc_code(np.array([1, -1]))
        expected = np.array([0.35012, -0.14449, 0.79437])
        self.assertArrayAlmostEqual(result, expected)

        result = self.encoder._get_llc_code(np.array([1, 2]))
        expected = np.array([0.65405, 0.45485, -0.10890])
        self.assertArrayAlmostEqual(result, expected)

    def test_encode_spatial_bin_empty(self):
        result = self.encoder._encode_spatial_bin(np.array([]))
        expected = np.zeros(self.encoder._size)
        self.assertTrue((result == expected).all())

    def test_encode_spatial_bin_l2_max(self):
        result = self.encoder._encode_spatial_bin(np.array([[3, 0], [-1, -1]]),
                                                  pooling='max')
        expected = np.array([1.11015, 0.22858, 0.98056])
        self.assertArrayAlmostEqual(result, expected)

    def test_encode_spatial_bin_l1_sum(self):
        result = self.encoder._encode_spatial_bin(np.array([[3, 0], [-1, -1],
                                                            [1, -1]]),
                                                  pooling='sum')
        expected = np.array([1.25112, -0.260689, 2.00956])
        self.assertArrayAlmostEqual(result, expected)

    def test_encode_spatial_bin_l1_max(self):
        result = self.encoder._encode_spatial_bin(np.array([[3, 0], [-1, -1],
                                                            [1, -1]]),
                                                  pooling='max')
        expected = np.array([1.11015, 0.22858, 0.98056])
        self.assertArrayAlmostEqual(result, expected)

    def test_encode_spatial_bin_l0_max(self):
        result = self.encoder._encode_spatial_bin(np.array([[3, 0], [-1, -1],
                                                            [1, -1], [1, 2]]),
                                                  pooling='max')
        expected = np.array([1.11015, 0.45485, 0.98056])
        self.assertArrayAlmostEqual(result, expected)

    def test_encode_max_eucl(self):
        # each row is a level 2 bin, each four rows are a level 1 bin
        spatial_pyramid_features = [[np.array([[3, 0], [-1, -1]]),
                                     np.array([[1, -1]]),
                                     np.array([]),
                                     np.array([])],  # l1 bin top left
                                    [np.array([]),
                                     np.array([]),
                                     np.array([]),
                                     np.array([])],  # l1 bin top right
                                    [np.array([]),
                                     np.array([]),
                                     np.array([]),
                                     np.array([])],  # l1 bin bottom left
                                    [np.array([[1, 2]]),
                                     np.array([]),
                                     np.array([]),
                                     np.array([])]]  # l1 bin bottom right
        result = self.encoder.encode(spatial_pyramid_features, pooling='max',
                                     normalization='eucl')
        expected = np.array([0.371106, 0.152049, 0.327786,      # l0 bin
                             0.371106, 0.076411, 0.327786,      # l1 bins
                             0, 0, 0,
                             0, 0, 0,
                             0.218639, 0.152049, 0,
                             0.371106, 0.076411, 0.327786,      # l2 bins top left
                             0.117040, -0.048301, 0.265370,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,                           # l2 bins top right
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,                           # l2 bins bottom left
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0.218639, 0.152049, -0.036404,    # l2 bins bottom right
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0])
        self.assertArrayAlmostEqual(result, expected)

    def test_encode_max_sum(self):
        # each row is a level 2 bin, each four rows are a level 1 bin
        spatial_pyramid_features = [[np.array([[3, 0], [-1, -1]]),
                                     np.array([[1, -1]]),
                                     np.array([]),
                                     np.array([])],  # l1 bin top left
                                    [np.array([]),
                                     np.array([]),
                                     np.array([]),
                                     np.array([])],  # l1 bin top right
                                    [np.array([]),
                                     np.array([]),
                                     np.array([]),
                                     np.array([])],  # l1 bin bottom left
                                    [np.array([[1, 2]]),
                                     np.array([]),
                                     np.array([]),
                                     np.array([])]]  # l1 bin bottom right
        result = self.encoder.encode(spatial_pyramid_features, pooling='max',
                                     normalization='sum')
        expected = np.array([0.107854, 0.044190, 0.095264,      # l0 bin
                             0.107854, 0.022207, 0.095264,      # l1 bins
                             0, 0, 0,
                             0, 0, 0,
                             0.063543, 0.044190, 0,
                             0.107854, 0.022207, 0.095264,      # l2 bins top left
                             0.034015, -0.014038, 0.077175,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,                           # l2 bins top right
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,                           # l2 bins bottom left
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0.063543, 0.044190, -0.010580,     # l2 bins bottom right
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0])
        self.assertArrayAlmostEqual(result, expected)

    def assertArrayAlmostEqual(self, arr1, arr2):
        try:
            np.testing.assert_allclose(arr1, arr2, atol=5e-4)
            result = True
        except AssertionError as err:
            result = False
            print(err)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
