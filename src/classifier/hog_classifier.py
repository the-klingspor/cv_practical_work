import cv2
import numpy as np
import random
import os
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from datautils.data_provider import DataProvider
from typing import List, Tuple
import xml.etree.ElementTree as ET
import pickle
import re

# For type hints
TupleList = List[Tuple[str, Tuple[int, int, int, int], str]]


class HogClassifier:

    PATCH_SIZE = (64, 32)
    SHOW_TRAINING_ROI = False
    SHOW_TRAINING_TIME = 512
    MANUAL_TRAINING_ROI = False
    TEST_DATA_FRACTION = 0.15
    SHUFFLE_TRAINING_DATA = True

    _hog_descriptor = None
    _svm = None
    _label_map = []
    _show_classified_img = False

    def _read_image(self, file_path):
        """Private helper function to unify image loading"""
        return cv2.imread(file_path, flags=cv2.IMREAD_GRAYSCALE)

    def _get_roi(self, img, roi):
        y = roi[1]
        x = roi[0]
        w = roi[2]
        h = roi[3]
        return img[y:y + h, x:x + w]

    def _rescale(self, img, scale=PATCH_SIZE):
        """Private helper function to unify and simplify image rescale operations"""
        return cv2.resize(img, scale, interpolation=cv2.INTER_CUBIC)

    def _create_hog_descriptor(self):
        """Helper function to create a cv2.HOGDescriptor and set necessary parameters.

          The parameter values can be seen in the openCV documentation"""
        winSize = self.PATCH_SIZE
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 0
        nlevels = 64
        signedGradient = False
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
        self._hog_descriptor = hog
        return hog

    def _svm_init(self, C=12.5, gamma=0.50625):
        """Private helper function to create a Support Vector Machine"""
        svm = cv2.ml.SVM_create()
        svm.setGamma(gamma)
        svm.setC(C)
        svm.setKernel(cv2.ml.SVM_RBF)
        svm.setType(cv2.ml.SVM_C_SVC)
        self._svm = svm

    def _svm_train(self, training_data, labels):
        """Private helper function to training a SVM

        :param training_data: A list of features. E.g.: HOG
        :param labels: The labels assigned to the 'training_data' features in the saem order. This can be a list of
        strings or numerical values and will be mapped to integers. This maping needs to be undone in the detection
        process
        """
        labels = self._map_labels_to_int(labels)
        self._svm.train(training_data, cv2.ml.ROW_SAMPLE, labels)

    def _svm_predict(self, sample):
        """Private helper function to simply the access to the SVM prediction function and perform prediction on a
        singel descriptor

        :param sample: A feature
        """
        return self._svm.predict([sample])[1].ravel()

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
            img = self._get_roi(img, data_3_tuple[1])
            descriptors.append(self._get_descriptor(self._hog_descriptor, img))
            label.append(data_3_tuple[2])
            if calc_additional_data:
                img =  cv2.flip(img, 1);
                descriptors.append(self._get_descriptor(self._hog_descriptor, img))
                label.append(data_3_tuple[2])
        return np.array(descriptors), np.array(label)

    def train(self, training_data: TupleList, calc_additional_data=True):
        """Used for training this classifier

        This method manages the training of a SVM using HOG features.

        :param training_data: A list of tuples containing (<a image path>, <A ROI x, y, w, h>, <A label string>)
        :param calc_additional_data: If True the vertical mirrored image with used to calculate a feature and to the
        results
        """
        print('Training started ...')
        hog = self._create_hog_descriptor()
        descriptors, labels = self._get_descriptors_and_labels(hog, training_data, calc_additional_data)
        self._svm_init()
        self._svm_train(descriptors, labels)
        print('Training finished!')

    def train_auto(self):
        # ToDo: use the the available auto_train function
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
            img = self._get_roi(img, roi)
        img = self._rescale(img)
        cv2.equalizeHist(img)
        descriptor = self._hog_descriptor.compute(img)
        prediction = self._svm_predict([descriptor])
        if self._label_map:
            prediction = self._label_map[int(prediction)]
        if expected_label:
            print(f"predicted: {prediction} expected {expected_label}")
            if prediction == expected_label:
                predicted_correct = True
        else:
            print(f"predicted: {prediction}")

        if self._show_classified_img:
            ax.set_title(prediction, fontsize=15)
            ax.axis('off')
            plt.show()
        return prediction, predicted_correct

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

    # ToDo: Dirty test hack from http://answers.opencv.org/question/56655/how-to-use-a-custom-svm-with-hogdescriptor-in-python/
    def dirty_test(self):
        self._svm.save("svm.xml")
        tree = ET.parse('svm.xml')
        root = tree.getroot()
        # now this is really dirty, but after ~3h of fighting OpenCV its what happens :-)
        SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0]
        rho = float(root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text)
        svmvec = [float(x) for x in re.sub('\s+', ' ', SVs.text).strip().split(' ')]
        svmvec.append(-rho)
        pickle.dump(svmvec, open("svm.pickle", 'wb'))
        svm = pickle.load(open("svm.pickle", "rb"))
        self._hog_descriptor.setSVMDetector(np.array(svm))
        del svm

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

    def _map_labels_to_int(self, labels):
        int_labels = []
        for label in labels:
            if label not in self._label_map:
                self._label_map.append(label)
            int_label = self._label_map.index(label)
            int_labels.append(int_label)
        return np.array(int_labels)

    def __init__(self, show_classified_image: bool = False):
        self._show_classified_img = show_classified_image


if __name__ == '__main__':
    provider = DataProvider("/home/tp/Downloads/CVSequences/data",
                            "/home/tp/Downloads/CVSequences/sequ",
                            "/home/tp/Downloads/CVSequences/npy",
                            True,
                            {"dayvision", "day"},  # the subfoldernames that are used for sequence separations
                            0.6,  # the maximum % of images of a kind that are used as training data
                            True,  # If any animal should be trained with equal amount of images
                            True,  # if the images should be shuffled
                            0)  # the random seed for shuffle. If 0 is choosen the seed is random too. Any other number can be choosen to increase the reproducibility of the experiment
    #1001461683744044007
    tr_data = provider.get_training_data()
    # provider.generate_sequences()
    # provider.segment_sequences()
    classifier = HogClassifier()
    classifier.train(provider.get_training_data(), True)
    classifier.batch_predict(provider.get_test_data())
    # classifier.test(provider.get_test_data())

    # classifier2 = HogClassifier()
    # classifier2.train(provider.get_training_data(), False)
    # classifier2.test(provider.get_test_data())

    # test_data = provider.get_test_data()
    # for data_tuple in test_data:
    #     img = classifier._read_image(data_tuple[0])
    #     classifier.detect(img)


