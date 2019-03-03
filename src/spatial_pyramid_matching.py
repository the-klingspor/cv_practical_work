import cv2
import time

from src.datautils.data_provider import DataProvider
from src.classifier.spm_classifier import SpmClassifier


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

    training_data = provider.get_training_data()
    # select random images from every category to get features to train the
    # codebook todo
    codebook_imgs = None

    classifier = SpmClassifier()
    classifier.train_codebook(codebook_imgs)

    tr_features, tr_labels = classifier.get_descr_and_labels(training_data)
    classifier.fit(tr_features, tr_labels)

    test_data = provider.get_test_data()
    test_features, test_labels = classifier.get_descr_and_labels(test_data)

    result = provider.score(test_features, test_labels)
    print("The mean accuracy of the classification was: {}".format(result))


