import cv2
import numpy as np
import random
import HirschData
import DachsData
import os
from matplotlib import pyplot as plt
import matplotlib.patches as patches


PATCH_SIZE = (64, 32)
LABEL_MAP = ["Wood", "Deer", "Badger"]
SHOW_TRAINING_ROI = False
SHOW_TRAINING_TIME = 512
MANUAL_TRAINING_ROI = False
TEST_DATA_FRACTION = 0.15
SHUFFLE_TRAINING_DATA = True


def read_image(fp):
    return cv2.imread(fp, flags=cv2.IMREAD_GRAYSCALE)


def get_roi(img, roi):
    y = roi[1]
    x = roi[0]
    w = roi[2]
    h = roi[3]
    return img[y:y + h, x:x + w]


def rescale(img, scale=PATCH_SIZE):
    return cv2.resize(img, scale, interpolation=cv2.INTER_CUBIC)


def get_hog():
    winSize = PATCH_SIZE
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
    return hog


def svm_init(C=12.5, gamma=0.50625):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)
    return model


def svm_train(model, samples, responses):
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model


def svm_predict(model, samples):
    return model.predict(samples)[1].ravel()


def get_descriptor(hog, img, roi):
    t_roi = roi
    if MANUAL_TRAINING_ROI:
        window_name = "Select ROI to learn ..."
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        t_roi = cv2.selectROI(window_name, img, showCrosshair=True)

    img = get_roi(img, t_roi)
    img = rescale(img)
    img = cv2.equalizeHist(img)

    if (SHOW_TRAINING_ROI):
        window_name = f'Training image'
        cv2.namedWindow(window_name, cv2.WND_PROP_ASPECT_RATIO)
        cv2.imshow(window_name, img)
        cv2.waitKey(SHOW_TRAINING_TIME)
        # cv2.destroyAllWindows()
    return hog.compute(img)


# not working with current code changes, because the global variables are accessed
# First test showed a reduced accuracy if used
#
# def train_wood(hog, img, roi_non_wood, descriptors, desc_labs, show_roi=False):
#     # train left
#     counts = 2
#     if roi_non_wood[0] > PATCH_SIZE[0]:
#         for i in range(counts):
#             h = random.randint(int(PATCH_SIZE[1]/2), 2 * PATCH_SIZE[1])
#             x = random.randint(0, int(roi_non_wood[0]/2))
#             w = min(random.randint(int(PATCH_SIZE[0]/2), 2 * PATCH_SIZE[0]), roi_non_wood[0] - x)
#             y = random.randint(0, img.shape[0] - h)
#             t_img = get_roi(img, (x, y, w, h))
#             t_img = rescale(t_img)
#             cv2.equalizeHist(t_img)
#             if show_roi:
#                 cv2.imshow(f'No Animal', t_img)
#                 cv2.waitKey(SHOW_TRAINING_TIME)
#                 cv2.destroyAllWindows()
#             descriptors.append(hog.compute(t_img))
#             desc_labs.append(0)


def get_descriptors_and_labels(hog, pathes, rois, label: int):
    descriptors = []
    desc_labs = []
    for i, _ in enumerate(pathes):
        img = read_image(pathes[i])
        descriptors.append(get_descriptor(hog, img, rois[i]))
        desc_labs.append(label)
        # train_wood(hog, img, rois[i], descriptors, desc_labs, SHOW_TRAINING_ROI)

    return descriptors, desc_labs


def shuffle_list(*ls):
    l = list(zip(*ls))
    random.shuffle(l)
    return zip(*l)


if __name__ == '__main__':
    print('Training SVM model ...')
    hog = get_hog()
    badger = np.load(os.path.dirname(os.path.realpath(__file__)) + '\\badger.npy').item()
    deer = np.load(os.path.dirname(os.path.realpath(__file__)) + '\deer.npy').item()
    badger_data = list(badger.keys())
    badger_roi = list(badger.values())
    deer_data = list(deer.keys())  # HirschData.data#[:int(len(badger_data)+100)]
    deer_roi = list(deer.values())  # HirschData.rois#[:int(len(badger_data)+100)]

    if SHUFFLE_TRAINING_DATA:
        deer_data, deer_roi = shuffle_list(deer_data, deer_roi)
        badger_data, badger_roi = shuffle_list(badger_data, badger_roi)

    deer_test_data_length = int(len(deer_data) * TEST_DATA_FRACTION)
    badger_test_data_length = int(len(badger_data) * TEST_DATA_FRACTION)
    print(f"The first {deer_test_data_length} deer images and the first {badger_test_data_length} badger images are used for tests")
    test_data = deer_data[: deer_test_data_length] + badger_data[: badger_test_data_length]
    test_rois = deer_roi[: deer_test_data_length] + badger_roi[: badger_test_data_length]

    # learn
    descriptors = []
    desc_labs = []
    des, labs = get_descriptors_and_labels(hog, deer_data[deer_test_data_length:], deer_roi[deer_test_data_length:], 1)
    descriptors += des
    desc_labs += labs
    des, labs = get_descriptors_and_labels(hog, badger_data[badger_test_data_length:],
                                           badger_roi[badger_test_data_length:], 2)
    descriptors += des
    desc_labs += labs

    model = svm_init()
    svm_train(model, np.array(descriptors), np.array(desc_labs))
    print('Training SVM model done!')

    # let the user select rois and predict them
    count_deer_good = 0
    count_badger_good = 0
    for i in range(len(test_data)):
        img = read_image(test_data[i])
        fig, ax = plt.subplots(1)
        ax.imshow(img, cmap='gray')
        roi = test_rois[i]
        x,y,w,h = roi
        # window_name = "Select ROI to predice ..."
        # window = cv2.namedWindow(window_name, flags=cv2.WINDOW_NORMAL)
        # roi = cv2.selectROI(window_name, img)
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        img = get_roi(img, roi)
        img = rescale(img)
        cv2.equalizeHist(img)
        descriptor = hog.compute(img)
        prediction = model.predict(np.array([descriptor]))[1].ravel()

        ax.set_title(LABEL_MAP[int(prediction[0])], fontsize=25)
        plt.show()
        if (prediction == 1 and i < deer_test_data_length):
            count_deer_good += 1
        elif prediction == 2 and i >= deer_test_data_length:
            count_badger_good += 1

        print("#", i + 1, ":", LABEL_MAP[int(prediction[0])])

    print(
        f"Evaluation Deer: {count_deer_good}/{deer_test_data_length} = {count_deer_good/deer_test_data_length * 100}%")
    print(
        f"Evaluation Badger: {count_badger_good}/{badger_test_data_length} = {count_badger_good/badger_test_data_length * 100}%")
    print('done!')
    cv2.destroyAllWindows()
