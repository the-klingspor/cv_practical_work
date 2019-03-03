import cv2
import time
from sklearn.svm import LinearSVC

from src.feature_extraction import FeatureExtraction
from src.datautils.data_provider import DataProvider
from src.llc_spatial_pyramid_encoding import LlcSpatialPyramidEncoder
from src.segment import segment


if __name__ == '__main__':
    provider = DataProvider(image_data_dir=None,
                            sequences_data_dir="/home/joschi/Documents/testDDD_seq",
                            segments_dir="/home/joschi/Documents/testDDD_segments",
                            show_images_with_roi=True,
                            folder_names_to_process=None,
                            max_training_data_percentage=0.6,
                            train_with_equal_image_amount=True,
                            shuffle_data=True,
                            seed=0)
    start = time.clock()
    provider.segment_sequences()
    segment_time = time.clock()
    print(segment_time - start)
    training_images = provider.get_training_data()
    # select random images from every category to get features to train the
    # codebook
    sift = cv2.xfeatures2d.SIFT_create()
    extractor = FeatureExtraction(sift)

    # extract dense features from every image for encoding

    # encode all training and test images

    # save LLC codes and labels in arrays for training and prediction
    training_codes = None
    testing_codes = None
    training_labels = None
    testing_labels = None

    # train an SVM with linear kernel
    classifier = LinearSVC()
    classifier.fit(training_codes, training_labels)

    # use test images for verification
    result = classifier.score(testing_codes, testing_labels)
    print("The mean accuracy of the classification was: {}".format(result))


