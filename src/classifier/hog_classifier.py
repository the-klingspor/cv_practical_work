import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple

from src.datautils.classify_util import get_roi_with_aspect_ratio
from datautils.data_provider import DataProvider

# For type hints
TupleList = List[Tuple[str, Tuple[int, int, int, int], str]]


class HogClassifier:

    PATCH_SIZE = (64, 128)

    # for debug purpose only
    SHOW_TRAINING_ROI = False
    SHOW_TRAINING_TIME = 1000

    # private class parameters
    _hog_descriptor = None
    _svm = None
    _show_classified_img = False
    _label_map =[]

    def __int__(self):
        self._hog_descriptor = None
        self._svm = None
        self._show_classified_img = False
        self._label_map = []

    def _read_image(self, file_path):
        """Private helper function to unify image loading"""
        return cv2.imread(file_path, flags=cv2.IMREAD_GRAYSCALE)

    def _rescale(self, img, scale=PATCH_SIZE):
        """Private helper function to unify and simplify image rescale operations"""
        return cv2.resize(img, scale, interpolation=cv2.INTER_CUBIC)

    def _create_hog_descriptor(self):
        """Helper function to create a cv2.HOGDescriptor and set necessary parameters.

          The parameter values can be seen in the openCV documentation"""
        winSize = self.PATCH_SIZE
        cellSize = (16, 16)
        blockSize = (int(cellSize[0]*2), int(cellSize[1]*2))
        blockStride = (int(blockSize[0]/2), int(blockSize[1]/2))
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradient = False
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
        self._hog_descriptor = hog
        return hog

    def _svm_init(self, C=12.5, gamma=0.50625, kernel=cv2.ml.SVM_RBF, type=cv2.ml.SVM_C_SVC):
        """Private helper function to create a Support Vector Machine"""
        svm = cv2.ml.SVM_create()
        svm.setGamma(gamma)
        svm.setC(C)
        svm.setKernel(kernel)
        svm.setType(type)
        self._svm = svm

    def _svm_train(self, descriptors, labels):
        """Private helper function to training a SVM

        :param descriptors: A list of features. E.g.: HOG
        :param labels: The labels assigned to the 'training_data' features in the saem order. This can be a list of
        strings or numerical values and will be mapped to integers. This maping needs to be undone in the detection
        process
        """
        labels = self._map_labels_to_int(labels)
        self._svm.train(descriptors, cv2.ml.ROW_SAMPLE, labels)


    def _svm_predict(self, sample):
        """Private helper function to simply the access to the SVM prediction function and perform prediction on a
        singel descriptor

        :param sample: A feature
        """
        return self._svm.predict(np.array([sample]))[1].ravel()

    def _get_descriptor(self, hog, img):
        """Private helper function to calculate a HOG feature for an image

        :param hog: openCV HOGDescriptor used to calculate the feature
        :param img: An openCV image. This image is rescaled and HOG is calculated
        :return: A HOG feature
        """
        img = self._rescale(img)
        img = cv2.equalizeHist(img)

        if self.SHOW_TRAINING_ROI:
            window_name = f'Training image'
            cv2.namedWindow(window_name, cv2.WND_PROP_ASPECT_RATIO)
            cv2.imshow(window_name, img)
            cv2.waitKey(self.SHOW_TRAINING_TIME)
            cv2.destroyAllWindows()
        return hog.compute(img)

    def _get_descriptors_and_labels(self, training_data, calc_additional_data):
        """A private helper function to return a HOG feature and the label.

        :param training_data: A list of tuples containing (<a image path>, <A ROI x, y, w, h>, <A label string>)
        :param calc_additional_data: If True the vertical mirrored image with used to calculate a feature and to the
        results
        :return: A list of features and a list of labels
        """
        descriptors = []
        label = []
        for data_3_tuple in training_data:
            img = self._read_image(data_3_tuple[0])
            img = get_roi_with_aspect_ratio(img, data_3_tuple[1], self.PATCH_SIZE[0]/self.PATCH_SIZE[1])
            descriptors.append(self._get_descriptor(self._hog_descriptor, img))
            label.append(data_3_tuple[2])
            if calc_additional_data:
                img = cv2.flip(img, 1);
                descriptors.append(self._get_descriptor(self._hog_descriptor, img))
                label.append(data_3_tuple[2])
        return np.array(descriptors), np.array(label)

    def train(self, training_data: TupleList, calc_additional_data=True, auto_train = False):
        """Used for training this classifier

        This method manages the training of a SVM using HOG features.

        :param training_data: A list of tuples containing (<a image path>, <A ROI x, y, w, h>, <A label string>)
        :param calc_additional_data: If True the vertical mirrored image with used to calculate a feature and to the
        results
        """
        print('Training started ...')
        hog = self._create_hog_descriptor()
        descriptors, labels = self._get_descriptors_and_labels(training_data, calc_additional_data)
        self._svm_init()
        if not auto_train:
            self._svm_train(descriptors, labels)
        else:
            self._svm_train_auto(descriptors, labels)
        print(f"SVM parameters: c = {self._svm.getC()}, gamma = {self._svm.getGamma()}")
        print('Training finished!')

    def _svm_train_auto(self, descriptors, labels):
        # ToDo: use the the available auto_train function
        labels = self._map_labels_to_int(labels)
        self._svm.trainAuto(descriptors, cv2.ml.ROW_SAMPLE, labels)
        pass

    def predict(self, img, roi=None, expected_label=None):
        predicted_correct = False
        if self._show_classified_img:
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            x, y, w, h = roi
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        if roi:
            img = get_roi_with_aspect_ratio(img, roi, self.PATCH_SIZE[0]/ self.PATCH_SIZE[1])
        img = self._rescale(img)
        cv2.equalizeHist(img)
        descriptor = self._hog_descriptor.compute(img)
        prediction = self._svm_predict(descriptor)
        if self._label_map:
            prediction = self._label_map[int(prediction)]
        if expected_label:
            # print(f"predicted: {prediction} expected {expected_label}")
            if prediction == expected_label:
                predicted_correct = True
        else:
            print(f"predicted: {prediction}")

        if self._show_classified_img:
            ax.set_title(prediction, fontsize=15)
            ax.axis('off')
            plt.show()
        return prediction, predicted_correct

    def _map_labels_to_int(self, labels):
        """
        Map labels that are given as strings to integers that can be used with
        predefined classifiers that expect int labels.

        :author: Thomas Poschadel, Joschka Strüber
        """
        int_labels = []
        for label in labels:
            if label not in self._label_map:
                self._label_map.append(label)
            int_label = self._label_map.index(label)
            int_labels.append(int_label)

        return np.array(int_labels)

    def test(self,  test_data: TupleList):
        evaluation_dict = dict()
        for data_3_tuple in test_data:
            img = self._read_image(data_3_tuple[0])
            roi = data_3_tuple[1]
            prediction, predicted_correct = self.predict(img, roi, data_3_tuple[2])
            if data_3_tuple[2] in evaluation_dict.keys():
                ok_count = evaluation_dict[data_3_tuple[2]][0]
                total_count = evaluation_dict[data_3_tuple[2]][1] + 1
                if predicted_correct:
                    ok_count += 1
                evaluation_dict[data_3_tuple[2]] = (ok_count, total_count)
            else:
                ok_count = 0
                if predicted_correct:
                    ok_count += 1
                evaluation_dict[data_3_tuple[2]] = (ok_count, 1)
        total_ok_sum = 0
        total_count_sum = 0
        for key, eval_tuple in evaluation_dict.items():
            total_ok_sum += eval_tuple[0]
            total_count_sum += eval_tuple[1]
            print(f"{key}: {eval_tuple[0]}/{eval_tuple[1]}={eval_tuple[0]/eval_tuple[1]:1.3f}")
        print(f"total: {total_ok_sum}/{total_count_sum}={total_ok_sum/total_count_sum:1.3f}")
        return evaluation_dict

    def detect(self, img):
        # self.dirty_test()
        ############## NOT REAL EXAMPLE
        # tada = cv2.ml.SVM_load("svm.xml")
        self._hog_descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        ###################################################
        detected = False
        (rois, weights) = self._hog_descriptor.detectMultiScale(img, 0, (256, 128), (4, 4), 1.1)
        for (x, y, w, h) in rois:
            if not detected:
                detected = True
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = img[y:y + h, x:x + w]
        if detected:
            window_name = f'detected image'
            cv2.namedWindow(window_name, cv2.WND_PROP_ASPECT_RATIO)
            cv2.imshow(window_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print('|', end='', flush=True)
        else:
            print('.', end='', flush=True)

    def my_detect(self, img):
        # iterate over image
            # calculate descriptor with 128 x 64 roi
            # calculate descriptor with 64 x 128 roi
            # ca
        pass

    def __init__(self, show_classified_image: bool = False):
        self._show_classified_img = show_classified_image


def test_multiple_times(number_of_runns):
    evals = []
    for i in range(0, number_of_runns):
        print(f"RUN {i} STARTED: ")
        provider = DataProvider("/home/tp/Downloads/CVSequences/data",
                                "/home/tp/Downloads/CVSequences/sequ",
                                "/home/tp/Downloads/CVSequences/npy",
                                # "/home/tp/Downloads/CVSequences/CVSequences",
                                True,
                                # {"dayvision", "day", "nightvision", "night"},  # the subfoldernames that are used for sequence separations
                                {"dayvision", "day"},  # the subfoldernames that are used for sequence separations
                                0.66,  # the maximum % of images of a kind that are used as training data
                                True,  # If any animal should be trained with equal amount of images
                                True,  # if the images should be shuffled
                                0)  # the random seed for shuffle. If 0 is choosen the seed is random too. Any other number can be choosen to increase the reproducibility of the experiment
        classifier = HogClassifier()
        classifier.train(provider.get_training_data(), True, False)
        eval = classifier.test(provider.get_test_data())
        evals.append(eval)
        del provider
        del classifier

    min_v = dict()
    max_v = dict()
    avg = dict()

    for eval in evals:
        for key, value in eval.items():
            good, total = value
            percent = good/total
            if key in min_v.keys():
                value = min_v[key]
                if value > percent:
                    min_v[key] = percent
            else:
                min_v[key] = percent
            if key in max_v.keys():
                value = max_v[key]
                if value < percent:
                    max_v[key] = percent
            else:
                max_v[key] = percent
            if key in avg.keys():
                avg_list = avg[key]
                avg_list.append(percent)
                avg[key] = avg_list
            else:
                avg[key] = [percent]
            print(f"{key}: {percent:1.3f}")
    for key, my_list in avg.items():
        summe = sum(my_list)
        my_std = np.std(np.array(my_list), dtype=np.float64)
        print(f"{key} avg: {summe/len(my_list):1.3f}, min: {min_v[key]:1.3f}, max: {max_v[key]:1.3f}, std: {my_std:1.3f}")


if __name__ == '__main__':
    # provider = DataProvider("/home/tp/Downloads/CVSequences/data",
    #                         "/home/tp/Downloads/CVSequences/sequ",
    #                         "/home/tp/Downloads/CVSequences/CVSequences",
    #                         # "/home/tp/Downloads/CVSequences/npy",
    #                         True,
    #                         {"dayvision", "day", "nightvision", "night"},  # the subfoldernames that are used for sequence separations
    #                         0.66,  # the maximum % of images of a kind that are used as training data
    #                         True,  # If any animal should be trained with equal amount of images
    #                         True,  # if the images should be shuffled
    #                         0, # the random seed for shuffle. If 0 is choosen the seed is random too. Any other number can be choosen to increase the reproducibility of the experiment
    #                         1)  # overtrain factor

    # good seeds: 8034652065224866011

    test_multiple_times(50)
    #
    # tr_data = provider.get_training_data()
    # provider.generate_sequences()
    # provider.segment_sequences()
    # classifier = HogClassifier()
    # classifier.train(provider.get_training_data(), False, True)
    # classifier.test(provider.get_test_data())

    # classifier2 = HogClassifier()
    # classifier2.train(provider.get_training_data(), False, False)
    # classifier2.test(provider.get_test_data())

    # test_data = provider.get_test_data()
    # for data_tuple in test_data:
    #     img = classifier._read_image(data_tuple[0])
    #     classifier.detect(img)


