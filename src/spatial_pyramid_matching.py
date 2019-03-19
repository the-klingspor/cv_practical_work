import time
import random

from sklearn.metrics import confusion_matrix

from src.datautils.data_provider import DataProvider
from src.classifier.spm_classifier import SpmClassifier
from src.datautils.classify_util import print_h_m_s

# number of random images used for training the code book of the encoder
SUB_SAMPLING_SIZE = 300
SEQ_DATA_DIR = "/home/joschi/Documents/DDD+_seq"
SEGMENT_DATA_DIR = "/home/joschi/Documents/DDD+_segs"

if __name__ == '__main__':
    """
    Main method that shows how to use the data provider for a set of sequences
    to segment animals and classify them with a spatial pyramid classifier using
    llc encoding and an svm with a linear kernel.
    
    The images are expected to be already ordered in sequences in the
    SEQ_DATA_DIR. For this you can use the "Camera Trap Sequencer" for example.
    
    :author: Joschka Strüber
    """
    provider = DataProvider(image_data_dir=None,
                            sequences_data_dir=SEQ_DATA_DIR,
                            segments_dir=SEGMENT_DATA_DIR,
                            show_images_with_roi=True,
                            folder_names_to_process=None,
                            max_training_data_percentage=0.7,
                            train_with_equal_image_amount=False,
                            shuffle_data=True,
                            seed=0)
    start = time.time()

    # provider.segment_sequences()

    segment_time = time.time()
    print_h_m_s(segment_time - start, "Segmentation time: ")

    training_data = provider.get_training_data()
    # select random images from the training data for training the code book
    code_book_data = []
    n_training_data = len(training_data)
    for i in range(SUB_SAMPLING_SIZE):
        random_index = random.randint(1, n_training_data) - 1
        code_book_data.append(training_data[random_index])

    classifier = SpmClassifier(codebook_size=1024,
                               alpha=500,
                               sigma=100)
    classifier.train_codebook(code_book_data)

    code_book_time = time.time()
    print_h_m_s(code_book_time - segment_time, "Code book time: ")

    tr_features, tr_labels = classifier.get_descr_and_labels(training_data)
    classifier.fit(tr_features, tr_labels)

    fit_time = time.time()
    print_h_m_s(fit_time - code_book_time, "Fit time: ")

    test_data = provider.get_test_data()
    test_features, test_labels = classifier.get_descr_and_labels(test_data)

    prediction = classifier.predict(test_features)

    result = classifier.score(test_features, test_labels)
    print("The mean accuracy of the classification was: {}".format(result))

    conf = confusion_matrix(test_labels, prediction)
    print(conf)
    conf_norm = conf / conf.sum(axis=0)
    print(conf_norm)
    score_time = time.time()
    print_h_m_s(score_time - fit_time, "Score time: ")


