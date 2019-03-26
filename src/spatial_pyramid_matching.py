from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from scipy.stats import reciprocal
from sklearn.metrics import confusion_matrix, f1_score, precision_score

from src.datautils.data_provider import DataProvider
from src.classifier.spm_transformer import SpmTransformer
from src.datautils.classify_util import split_data_labels


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
    X, y, label_map = split_data_labels(labeled_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=None)

    transformer = SpmTransformer(density='dense', cb_train_size=100)

    llc_svm_pipeline = Pipeline([
        ('spm_transformer', transformer),
        ('classifier', LinearSVC(C=0.5511, class_weight='balanced'))
    ])

    param_dist = {
        'spm_transformer__codebook_size': [512, 1024, 2048],
        'spm_transformer__alpha': reciprocal(100, 1000),
        'spm_transformer__sigma': reciprocal(50, 500),
        'spm_transformer__pooling': ['max', 'sum'],
        'spm_transformer__normalization': ['sum', 'eucl'],
        'classifier__C': reciprocal(0.1, 5)
    }

    # Unfortunately, the extractors of the FeatureExtraction can't be pickled.
    # This prevents us from setting n_jobs to more than one thread. However,
    # the encoding uses more threads anyway, so the core utilization should be
    # fairly good.
    random_search = RandomizedSearchCV(llc_svm_pipeline, param_dist, n_iter=20,
                                       cv=5, verbose=3, n_jobs=1, scoring='f1')

    random_search.fit(X_train, y_train)
    cv_results = random_search.cv_results_

    for mean_score, params in zip(cv_results["mean_test_score"],
                                  cv_results["params"]):
        print(mean_score, params)

    final_classifier = random_search.best_estimator_

    predictions = final_classifier.predict(X_test)
    conf = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:\n", conf)

    precision = precision_score(y_test, predictions, average='weighted')
    print("Precision score: ", precision)
    score_f1 = f1_score(y_test, predictions, average='weighted')
    print("F1 score: ", score_f1)


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
    labeled_data = provider.get_data_list()
    X, y, label_map = split_data_labels(labeled_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100,
                                                        train_size=100,
                                                        random_state=None)

    llc_svm_pipeline = Pipeline([
        ('spm_transformer', SpmTransformer(cb_train_size=300)),
        ('classifier', LinearSVC(class_weight='balanced'))
    ])

    param_dist = {
        'spm_transformer__density': ['dense'],
        'spm_transformer__codebook_size': [512, 1024, 2048],
        'spm_transformer__alpha': reciprocal(100, 1000),
        'spm_transformer__sigma': reciprocal(50, 500),
        'spm_transformer__pooling': ['max'],
        'spm_transformer__normalization': ['eucl'],
        'classifier__C': reciprocal(0.1, 5)
    }

    # Unfortunately, the extractors of the FeatureExtraction can't be pickled.
    # This prevents us from setting n_jobs to more than one thread. However,
    # the encoding uses more threads anyway, so the core utilization should be
    # fairly good.
    random_search = RandomizedSearchCV(llc_svm_pipeline, param_dist, n_iter=10,
                                       cv=5, verbose=5, n_jobs=1,
                                       scoring='f1_weighted')

    random_search.fit(X_train, y_train)
    cv_results = random_search.cv_results_

    for mean_score, params in zip(cv_results["mean_test_score"],
                                  cv_results["params"]):
        print(mean_score, params)

    final_classifier = random_search.best_estimator_

    predictions = final_classifier.predict(X_test)
    conf = confusion_matrix(y_test, predictions)
    print("Label: \n", label_map)
    print("Confusion Matrix:\n", conf)

    precision = precision_score(y_test, predictions, average='weighted')
    print("Precision score: ", precision)
    score_f1 = f1_score(y_test, predictions, average='weighted')
    print("F1 score: ", score_f1)


if __name__ == '__main__':
    """
    Main method that shows how to use the data provider for a set of sequences
    to segment animals and classify them with a spatial pyramid classifier using
    llc encoding and an svm with a linear kernel.
    
    The images are expected to be already ordered in sequences in the
    SEQ_DATA_DIR. For this you can use the "Camera Trap Sequencer" for example.
    
    :author: Joschka Strüber
    """

    # call_DDD_pipeline()

    call_DDD_plus_pipeline()

