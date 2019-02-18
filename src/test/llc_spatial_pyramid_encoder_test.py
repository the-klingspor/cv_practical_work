import unittest

from src.llc_spatial_pyramid_encoding import LlcSpatialPyramidEncoder


class LlcSpatialPyramidEncoderTest(unittest.TestCase):
    def setUp(self):
        self.encoder = LlcSpatialPyramidEncoder(alpha=2, sigma=3)

    def test_train_codebook(self):
        self.assertTrue(True)

    def test_get_llc_code(self):
        self.assertTrue(True)

    def test_encode_spatial_bin(self):
        self.assertTrue(True)

    def test_encode(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
