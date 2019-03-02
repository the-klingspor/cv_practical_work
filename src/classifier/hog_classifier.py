import cv2
import numpy as np
import random
import os
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from datautils.data_provider import DataProvider
from typing import List, Tuple

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
        return cv2.imread(file_path, flags=cv2.IMREAD_GRAYSCALE)

    def _get_roi(self, img, roi):
        y = roi[1]
        x = roi[0]
        w = roi[2]
        h = roi[3]
        return img[y:y + h, x:x + w]

    def _rescale(self, img, scale=PATCH_SIZE):
        return cv2.resize(img, scale, interpolation=cv2.INTER_CUBIC)

    def _create_hog_descriptor(self):
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
        svm = cv2.ml.SVM_create()
        svm.setGamma(gamma)
        svm.setC(C)
        svm.setKernel(cv2.ml.SVM_RBF)
        svm.setType(cv2.ml.SVM_C_SVC)
        self._svm = svm

    def _svm_train(self, training_data, labels):
        labels = self._map_labels_to_int(labels)
        self._svm.train(training_data, cv2.ml.ROW_SAMPLE, labels)

    def _svm_predict(self, model, samples):
        return model.predict(samples)[1].ravel()

    def _get_descriptor(self, hog, img, roi):
        t_roi = roi
        if self.MANUAL_TRAINING_ROI:
            window_name = "Select ROI to learn ..."
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            t_roi = cv2.selectROI(window_name, img, showCrosshair=True)

        img = self._get_roi(img, t_roi)
        img = self._rescale(img)
        img = cv2.equalizeHist(img)

        if self.SHOW_TRAINING_ROI:
            window_name = f'Training image'
            cv2.namedWindow(window_name, cv2.WND_PROP_ASPECT_RATIO)
            cv2.imshow(window_name, img)
            cv2.waitKey(self.SHOW_TRAINING_TIME)
            # cv2.destroyAllWindows()
        return hog.compute(img)

    def _get_descriptors_and_labels(self, hog, training_data):
        descriptors = []
        label = []
        for data_3_tuple in training_data:
            img = self._read_image(data_3_tuple[0])
            descriptors.append(self._get_descriptor(hog, img, data_3_tuple[1]))
            label.append(data_3_tuple[2])
        return np.array(descriptors), np.array(label)

    def train(self, training_data: TupleList):
        print('Training started ...')
        hog = self._create_hog_descriptor()
        descriptors, labels = self._get_descriptors_and_labels(hog, training_data)
        self._svm_init()
        self._svm_train(descriptors, labels)
        print('Training finished!')

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
        prediction = self._svm.predict(np.array([descriptor]))[1].ravel()
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
                            0.7,  # the maximum % of images of a kind that are used as training data
                            True,  # If any animal should be trained with equal amount of images
                            True,  # if the images should be shuffled
                            0)  # the random seed for shuffle. If 0 is choosen the seed is random too. Any other number can be choosen to increase the reproducibility of the experiment
    tr_data = provider.get_training_data()
    # provider.generate_sequences()
    # provider.segment_sequences()
    classifier = HogClassifier()
    classifier.train(provider.get_training_data())
    classifier.test(provider.get_test_data())


