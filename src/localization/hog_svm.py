import numpy as np
from sklearn.model_selection import train_test_split
import os
from matplotlib import pyplot as plt, patches
import cv2
from sklearn.svm import SVC
from functools import reduce
import pickle

target_names = ['deer', 'badger', 'empty']
patch_size = (64, 64)
vec_patch_size = reduce((lambda x, y: x * y), patch_size)

winSize = patch_size
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


def rescalePatch(img, scale=patch_size):
    return cv2.resize(img, scale, interpolation=cv2.INTER_CUBIC)


def slidingWindows(image, w, h, stepSize=None):
    factor = 0.8
    width, height, _ = image.shape
    while factor > 0.2:
        new_w = int(h * factor)
        new_h = int(h * factor)
        step_w = int(new_w / 4)
        step_h = int(new_w / 4)
        for y in range(0, width - new_w, step_w):
            for x in range(0, height - new_h, step_h):
                yield (x, y, new_w, new_h, image[y:y + new_w, x:x + new_h, :])
        factor = factor - 0.1


def localisation(clf, hog, im, target=''):
    fig, ax = plt.subplots(1)
    probability = -1
    bestRoi = (0, 0, 1, 1)
    sliding_window = slidingWindows(im, im.shape[0], im.shape[1])
    for x, y, w, h, image in sliding_window:
        proj = hog.compute(rescalePatch(image))
        p = np.amax(clf.predict_proba(proj.T)[0])
        prediction = clf.predict(proj.T)[0]
        if target == target_names[prediction] and (probability < p or probability is -1):
            probability = p
            bestRoi = (x, y, w, h)
    x, y, w, h = bestRoi
    ax.imshow(im, cmap='gray')
    rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    fig.show()


def load_images(paths):
    data = []
    labels = []
    for path in paths:
        for folder in os.listdir(path):
            im = plt.imread(os.path.join(path, folder))
            if len(im.shape) == 3:
                hogData = hog.compute(rescalePatch(im))
                data.append(hogData)
                labels.append(paths.index(path))
    return np.array(data), np.array(labels)


def TrainingsPhase():
    modelpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'svm.sav')
    try:
        return pickle.load(open(modelpath, 'rb')), hog
    except FileNotFoundError as err:
        print('Model nicht gefunden..Training ist angefangen')
        # Phase I: Lade die Trainingsbilder. Die Schnittbilder sollen in Quadrate das Tier enthalten
        # redefine target_names accordingly
        pathFirstLabel = "Pfad für Schnittbilder von Label deer"
        pathSecondLabel = "Pfad für Schnittbilder von Label badger"
        pathThirdLabel = "Pfad für Schnittbilder von Label empty"
        x_data, y_data = load_images([pathFirstLabel, pathSecondLabel, pathThirdLabel])
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=0)
        X_train = np.squeeze(X_train, axis=2)
        # Initialisierung von SVM
        svm = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
                  max_iter=-1, probability=True, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
        svm = svm.fit(X_train, y_train)
        # Speichere SVM
        pickle.dump(svm, open(modelpath, 'wb'))
        return svm, hog


if __name__ == '__main__':
    # Phase I: Initialisierung
    svm, hog = TrainingsPhase()

    # Phase II: Lokalisirung Testen
    imagesPaths = ['IMG_0123', 'IMG_0142', 'IMG_0297', 'IMG_0417', 'IMG_0310']
    for path in imagesPaths:
        image = plt.imread(path)[33:1465, :, :]  # Crop header and footer with 33, 1465 respectively
        localisation(svm, hog, image, 'deer')  # possible labels ['deer', 'badger', 'empty']
