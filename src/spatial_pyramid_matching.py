import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from src.datautils.data_provider import DataProvider
from src.classifier.spm_transformer import SpmTransformer
from src.datautils.classify_util import print_h_m_s, split_data_labels


def call_DDD_pipeline():
    seq_data_dir = "/home/joschi/Documents/DDD_seqs"
    segment_data_dir = "/home/joschi/Documents/DDD_segs"
    provider = DataProvider(image_data_dir=None,
                            sequences_data_dir=seq_data_dir,
                            segments_dir=segment_data_dir,
                            show_images_with_roi=True,
                            folder_names_to_process=None,
                            max_training_data_percentage=0.7,
                            train_with_equal_image_amount=False,
                            shuffle_data=True,
                            seed=0)
    # provider.segment_sequences()
    labeled_data = provider.get_data_list()
    train_labeled_data, test_labeled_data = train_test_split(labeled_data,
                                                             test_size=0.2,
                                                             random_state=42)
    train_data, train_labels, label_map = split_data_labels(
        train_labeled_data)

    llc_svm_pipeline = Pipeline([
        ('spm_transformer', SpmTransformer()),
        ('classififer', SVC(gamma='auto'))
    ])

    param_grid = [{
        'spm_transformer__codebook_size': [128, 256, 1024, 2048],
        'spm_transformer__alpha': [10, 100, 500],
        'spm_transformer__sigma': [10, 100, 500],
        'spm_transformer__pooling': ['max', 'sum'],
        'spm_tranformer__normalization': ['sum', 'eucl'],
    }]

    grid_search = GridSearchCV(llc_svm_pipeline, param_grid, cv=5, verbose=2,
                               n_jobs=4)

    grid_search.fit(train_data, train_labels)
    cv_results = grid_search.cv_results_

    for mean_score, params in zip(cv_results["mean_test_score"],
                                  cv_results["params"]):
        print(mean_score, params)

    test_data, test_labels, x = split_data_labels(test_labeled_data)

    score = llc_svm_pipeline.score(test_data, test_labels)
    print(score)


def call_DDD_plus_pipeline():
    seq_data_dir = "/home/joschi/Documents/DDD+_seqs"
    segment_data_dir = "/home/joschi/Documents/DDD+_segs"
    provider = DataProvider(image_data_dir=None,
                            sequences_data_dir=seq_data_dir,
                            segments_dir=segment_data_dir,
                            show_images_with_roi=True,
                            folder_names_to_process=None,
                            max_training_data_percentage=0.7,
                            train_with_equal_image_amount=False,
                            shuffle_data=True,
                            seed=0)
    # provider.segment_sequences()


if __name__ == '__main__':
    """
    Main method that shows how to use the data provider for a set of sequences
    to segment animals and classify them with a spatial pyramid classifier using
    llc encoding and an svm with a linear kernel.
    
    The images are expected to be already ordered in sequences in the
    SEQ_DATA_DIR. For this you can use the "Camera Trap Sequencer" for example.
    
    :author: Joschka Strüber
    """

    call_DDD_pipeline()

    """
    conf = confusion_matrix(test_labels, prediction)
    print(conf)
    conf_norm = conf / conf.sum(axis=0)
    print(conf_norm)
    """

