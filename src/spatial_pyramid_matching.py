from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.stats import reciprocal
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from src.datautils.data_provider import DataProvider
from src.classifier.spm_transformer import SpmTransformer
from src.datautils.classify_util import split_data_labels
from src.datautils.local_binary_patterns import LocalBinaryPatterns


# directories for sequences and segmentation files for DDD and DDD+
DIR_DDD_SEQUENCES = "/home/joschi/Documents/DDD_seqs"
DIR_DDD_SEGMENTS = "/home/joschi/Documents/DDD_segs"
DIR_DDD_PLUS_SEQUENCES = "/home/joschi/Documents/DDD+_seqs"
DIR_DDD_PLUS_SEGMENTS = "/home/joschi/Documents/DDD+_segs"

def call_DDD_sift_pipeline():
    """
    Call a pipeline with SIFT features and spatial pyramid matching on the DDD.
    Randomized 5-fold cross validation is used for hyperparameter search.

    :author: Joschka Strüber
    """
    print("Calling SIFT pipeline on the DDD.")
    provider = DataProvider(image_data_dir=None,
                            sequences_data_dir=DIR_DDD_SEQUENCES,
                            segments_dir=DIR_DDD_SEGMENTS,
                            show_images_with_roi=True,
                            folder_names_to_process=None,
                            max_training_data_percentage=0.7,
                            train_with_equal_image_amount=False,
                            shuffle_data=True,
                            seed=0)
    provider.segment_sequences()

    labeled_data = provider.get_data_list()
    X, y, label_map = split_data_labels(labeled_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=None)

    transformer = SpmTransformer(density='dense', cb_train_size=100)

    sift_pipeline = Pipeline([
        ('spm_transformer', transformer),
        ('classifier', LinearSVC(class_weight='balanced'))
    ])

    param_dist = {
        'spm_transformer__codebook_size': [256, 512, 1024],
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
    random_search = RandomizedSearchCV(sift_pipeline, param_dist, n_iter=20,
                                       cv=5, verbose=3, n_jobs=1, scoring='f1')

    random_search.fit(X_train, y_train)
    cv_results = random_search.cv_results_

    for mean_score, params in zip(cv_results["mean_test_score"],
                                  cv_results["params"]):
        print(mean_score, params)

    final_classifier = random_search.best_estimator_

    predictions = final_classifier.predict(X_test)
    conf = confusion_matrix(y_test, predictions)
    print("Labels: ", label_map)
    print("Confusion Matrix:\n", conf)

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: ", accuracy)
    score_f1 = f1_score(y_test, predictions, average='weighted')
    print("F1 score: ", score_f1)


def call_DDD_lbp_pipeline():
    """
    Call a pipeline with uniform LBP features and spatial pyramid matching on
    the DDD.
    Randomized 5-fold cross validation is used for hyperparameter search.

    :author: Joschka Strüber
    """
    print("Calling LBP pipeline on the DDD.")
    provider = DataProvider(image_data_dir=None,
                            sequences_data_dir=DIR_DDD_SEQUENCES,
                            segments_dir=DIR_DDD_SEGMENTS,
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

    extractor = LocalBinaryPatterns(n_points=16, radius=2)
    transformer = SpmTransformer(extractor=extractor, density='dense',
                                 cb_train_size=100)

    lbp_pipeline = Pipeline([
        ('spm_transformer', transformer),
        ('classifier', LinearSVC(class_weight='balanced'))
    ])

    param_dist = {
        'spm_transformer__codebook_size': [512, 1024],
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
    random_search = RandomizedSearchCV(lbp_pipeline, param_dist, n_iter=15,
                                       cv=5, verbose=3, scoring='f1')

    random_search.fit(X_train, y_train)
    cv_results = random_search.cv_results_

    for mean_score, params in zip(cv_results["mean_test_score"],
                                  cv_results["params"]):
        print(mean_score, params)

    final_classifier = random_search.best_estimator_

    predictions = final_classifier.predict(X_test)
    conf = confusion_matrix(y_test, predictions)
    print("Labels: ", label_map)
    print("Confusion matrix:\n", conf)

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: ", accuracy)
    score_f1 = f1_score(y_test, predictions, average='weighted')
    print("F1 score: ", score_f1)


def call_DDD_plus_sift_pipeline():
    """
    Call a pipeline with SIFT features and spatial pyramid matching on the DDD+.
    Randomized 5-fold cross validation is used for hyperparameter search.
    :author: Joschka Strüber
    """
    print("Calling SIFT pipeline on the DDD+.")
    provider = DataProvider(image_data_dir=None,
                            sequences_data_dir=DIR_DDD_PLUS_SEQUENCES,
                            segments_dir=DIR_DDD_PLUS_SEGMENTS,
                            show_images_with_roi=True,
                            folder_names_to_process=None,
                            max_training_data_percentage=0.7,
                            train_with_equal_image_amount=False,
                            shuffle_data=True,
                            seed=0)
    provider.segment_sequences()
    
    labeled_data = provider.get_data_list()
    X, y, label_map = split_data_labels(labeled_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=None)

    sift_pipeline = Pipeline([
        ('spm_transformer', SpmTransformer(cb_train_size=500)),
        ('classifier', LinearSVC(class_weight='balanced'))
    ])

    param_dist = {
        'spm_transformer__density': ['dense'],
        'spm_transformer__codebook_size': [256, 512, 1024],
        'spm_transformer__alpha': reciprocal(100, 1000),
        'spm_transformer__sigma': reciprocal(50, 500),
        'spm_transformer__pooling': ['max', 'sum'],
        'spm_transformer__normalization': ['eucl', 'sum'],
        'classifier__C': reciprocal(0.1, 5)
    }

    # Unfortunately, the extractors of the FeatureExtraction can't be pickled.
    # This prevents us from setting n_jobs to more than one thread. However,
    # the encoding uses more threads anyway, so the core utilization should be
    # fairly good.
    random_search = RandomizedSearchCV(sift_pipeline, param_dist, n_iter=10,
                                       cv=5, verbose=5, scoring='f1_weighted')

    random_search.fit(X_train, y_train)
    cv_results = random_search.cv_results_

    for mean_score, params in zip(cv_results["mean_test_score"],
                                  cv_results["params"]):
        print(mean_score, params)

    final_classifier = random_search.best_estimator_

    predictions = final_classifier.predict(X_test)
    conf = confusion_matrix(y_test, predictions)
    print("Labels: ", label_map)
    print("Confusion matrix:\n", conf)

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: ", accuracy)
    score_f1 = f1_score(y_test, predictions, average='weighted')
    print("F1 score: ", score_f1)


def call_DDD_plus_sift_lbp_pipeline():
    """
    Call a pipeline with uniform LBP features and SIFT features on the DDD+. The
    two kinds of features are concatenated using a feature union and used to
    train a linear SVM.

    :author: Joschka Strüber
    """
    print("Calling LBP/SIFT pipeline on the DDD+.")
    provider = DataProvider(image_data_dir=None,
                            sequences_data_dir=DIR_DDD_PLUS_SEQUENCES,
                            segments_dir=DIR_DDD_PLUS_SEGMENTS,
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

    sift_transformer = SpmTransformer(density='dense',
                                      cb_train_size=300,
                                      codebook_size=1024,
                                      alpha=350,
                                      sigma=300,
                                      pooling='max',
                                      normalization='eucl')

    extractor = LocalBinaryPatterns(n_points=16, radius=2)
    lbp_transformer = SpmTransformer(extractor=extractor,
                                     density='dense',
                                     cb_train_size=300,
                                     codebook_size=512,
                                     alpha=350,
                                     sigma=100,
                                     pooling='max',
                                     normalization='eucl')

    feature_union_pipeline = FeatureUnion(transformer_list=[
        ('sift_trans', sift_transformer),
        ('lbp_trans', lbp_transformer)
    ])

    full_pipeline = Pipeline([
        ('transformer', feature_union_pipeline),
        ('classifier', LinearSVC(C=1.3, class_weight='balanced'))
    ])

    full_pipeline.fit(X_train, y_train)

    bias_preds = full_pipeline.predict(X_train)
    bias_score = f1_score(y_train, bias_preds, average="weighted")
    print("Bias score: ", bias_score)

    predictions = full_pipeline.predict(X_test)
    conf = confusion_matrix(y_test, predictions)
    print("Labels: ", label_map)
    print("Confusion matrix:\n", conf)

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: ", accuracy)
    score_f1 = f1_score(y_test, predictions, average='weighted')
    print("F1 score: ", score_f1)


if __name__ == '__main__':
    """
    Main method that shows how to use the data provider for a set of sequences
    to segment animals and classify them with a spatial pyramid classifier using
    llc encoding and an svm with a linear kernel.
    
    Several pipelines with SIFT features, LBP features and a combination of both
    are built and called on both the DDD and DDD+. However, even more
    combinations or enhancements such as ensemble like a Voting Classifier are
    possible.
    
    The images are expected to be already ordered in sequences in the
    directories DIR_DDD_SEQUENCES and DIR_DDD_PLUS_SEQUENCES. For this you can 
    use the "Camera Trap Sequencer" for example.
    
    :author: Joschka Strüber
    """

    call_DDD_sift_pipeline()

    call_DDD_lbp_pipeline()

    call_DDD_plus_sift_pipeline()
    
    call_DDD_plus_sift_lbp_pipeline()
