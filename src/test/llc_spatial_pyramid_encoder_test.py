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

    def test_get_llc_code(self):
        result = self.encoder._get_llc_code(np.array([3, 0]))
        expected = np.array([1.12383, -0.23659, 0.21778])
        self.assertArrayAlmostEqual(result, expected)

        result = self.encoder._get_llc_code(np.array([-1, -1]))
        expected = np.array([-0.20915, 0.22858, 0.98056])
        self.assertArrayAlmostEqual(result, expected)

        result = self.encoder._get_llc_code(np.array([1, -1]))
        expected = np.array([0.33553, -0.13899, 0.80346])
        self.assertArrayAlmostEqual(result, expected)

        result = self.encoder._get_llc_code(np.array([1, 2]))
        expected = np.array([3.02107, -1.96539, -0.05568])
        self.assertArrayAlmostEqual(result, expected)

    def test_encode_spatial_bin_empty(self):
        result = self.encoder._encode_spatial_bin(np.array([]))
        expected = np.zeros(self.encoder._size)
        self.assertTrue((result == expected).all())

    # todo: normalization only for full encoding, not individual bin encodings

    def test_encode_spatial_bin_l2_max(self):
        result = self.encoder._encode_spatial_bin(np.array([[3, 0], [-1, -1]]),
                                                  pooling='max')
        expected = np.array([1.12383, 0.22858, 0.98056])
        self.assertArrayAlmostEqual(result, expected)

    def test_encode_spatial_bin_l1_sum(self):
        result = self.encoder._encode_spatial_bin(np.array([[3, 0], [-1, -1],
                                                            [1, -1]]),
                                                  pooling='sum')
        expected = np.array([1.25021, -0.25202, 2.0018])
        self.assertArrayAlmostEqual(result, expected)

    def test_encode_spatial_bin_l1_max(self):
        result = self.encoder._encode_spatial_bin(np.array([[3, 0], [-1, -1],
                                                            [1, -1]]),
                                                  pooling='max')
        expected = np.array([1.12383, 0.22858, 0.98056])
        self.assertArrayAlmostEqual(result, expected)

    def test_encode_spatial_bin_l0_max(self):
        result = self.encoder._encode_spatial_bin(np.array([[3, 0], [-1, -1],
                                                            [1, -1], [1, 2]]),
                                                  pooling='max')
        expected = np.array([3.02107, 0.22858, 0.98056])
        self.assertArrayAlmostEqual(result, expected)

    def test_encode_max_eucl(self):
        # each row is a level 2 bin, each four rows are a level 1 bin
        spatial_pyramid_features = np.array([[[[3, 0], [-1, -1]],
                                              [[1, -1]],
                                              [],
                                              []],  # l1 bin top left
                                             [[],
                                              [],
                                              [],
                                              []],  # l1 bin top right
                                             [[],
                                              [],
                                              [],
                                              []],  # l1 bin bottom left
                                             [[1, 2],
                                              [],
                                              [],
                                              []]])  # l1 bin bottom right
        result = self.encoder.encode(spatial_pyramid_features, pooling='max',
                                     normalization='eucl')
        expected = np.array([0.469204, 0.0355009, 0.152291,     # l0 bin
                             0.174543, 0.0355009, 0.152291,     # l1 bins
                             0, 0, 0,
                             0, 0, 0,
                             0.469204, -0.305246, -0.00864769,
                             0.174543, 0.0355009, 0.152291,     # l2 bins top left
                             0.0521114, -0.0215866, 0.124786,
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
                             0.469204, -0.305246, -0.00864769,  # l2 bins bottom right
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0])
        self.assertArrayAlmostEqual(result, expected)

    def test_encode_max_sum(self):
        # each row is a level 2 bin, each four rows are a level 1 bin
        spatial_pyramid_features = np.array([[[[3, 0], [-1, -1]],
                                              [[1, -1]],
                                              [],
                                              []],  # l1 bin top left
                                             [[],
                                              [],
                                              [],
                                              []],  # l1 bin top right
                                             [[],
                                              [],
                                              [],
                                              []],  # l1 bin bottom left
                                             [[1, 2],
                                              [],
                                              [],
                                              []]])  # l1 bin bottom right
        result = self.encoder.encode(spatial_pyramid_features, pooling='max',
                                     normalization='sum')
        expected = np.array([0.253955, 0.0192147, 0.082427,     # l0 bin
                             0.0944705, 0.0192147, 0.082427,    # l1 bins
                             0, 0, 0,
                             0, 0, 0,
                             0.253955, -0.165213, -0.00468053,
                             0.0944705, 0.0192147, 0.082427,    # l2 bins, top left
                             0.028205, -0.0116837, 0.0675398,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,                           # l2 bins, top right
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,                           # l2 bins, bottom left
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0.253955, -0.165213, -0.00468053,  # l2 bins, bottom right
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0])
        self.assertArrayAlmostEqual(result, expected)

    def assertArrayAlmostEqual(self, arr1, arr2):
        try:
            np.allclose(arr1, arr2, atol=1e-5)
            result = True
        except AssertionError as err:
            result = False
            print(err)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
