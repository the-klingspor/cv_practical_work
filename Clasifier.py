import cv2
import numpy as np
import random
import HirschData
import DachsData

PATCH_SIZE = (64, 64)
LABEL_MAP = ["Wood", "Deer", "Dachs"]
SHOW_TRAINING_ROI = False
SHOW_TRAINING_TIME = 1000
MANUAL_TRAINING_ROI = False

fps = [
    'images/dachs/IMG_1850.JPG',
    'images/dachs/IMG_2083.JPG',
    'images/dachs/IMG_1842.JPG',
    'images/dachs/IMG_1834.JPG',
    'images/dachs/IMG_1846.JPG',
    'images/dachs/IMG_1849.JPG',
    'images/dachs/IMG_1785.JPG',
    'images/dachs/IMG_1836.JPG',
    'images/hirsch/IMG_0553.JPG',
    'images/hirsch/IMG2_2808.jpg',
    'images/hirsch/IMG_0131.JPG',
    'images/hirsch/IMG_0134.JPG',
    'images/hirsch/IMG_0541.JPG',
    'images/hirsch/IMG_0543.JPG',
    'images/hirsch/IMG_0184.JPG',
    'images/hirsch/IMG_0150.JPG',
    'images/hirsch/IMG_0422.JPG']

# (x, y, w, h)
rois = [
    (1736, 1136, 312, 235),
   (1675, 833, 205, 111),
   (1045, 1102, 579, 247),
   (19, 1018, 244, 138),
   (1648, 1130, 400, 250),
   (1671, 1154, 377, 231),
   (1982, 1225, 66, 114),
   (0, 969, 61, 155),
    (4, 32, 1756, 1471),
   (1523, 684, 525, 541),
   (1403, 758, 168, 146),
   (1392, 745, 186, 149),
   (382, 32, 1666, 1471),
   (368, 38, 1675, 1463),
   (5, 1012, 391, 489),
   (1929, 790, 119, 100),
   (1198, 509, 312, 433)]

lables =[2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def read_image(fp):
    return cv2.imread(fp, flags=cv2.IMREAD_GRAYSCALE)


def get_roi(img, roi):
    y = roi[1]
    x = roi[0]
    w = roi[2]
    h = roi[3]
    return img[y:y+h, x:x+w]


def rescale(img, scale=PATCH_SIZE):
    return cv2.resize(img, scale, interpolation = cv2.INTER_CUBIC)


def get_hog():
    winSize = (64, 64)
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


def svmInit(C=12.5, gamma=0.50625):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)
    return model


def svmTrain(model, samples, responses):
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model


def svmPredict(model, samples):
    return model.predict(samples)[1].ravel()


def getDescriptor(hog, img, roi):
    t_roi = roi
    if MANUAL_TRAINING_ROI:
        window_name = "Select ROI to learn ..."
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        t_roi = cv2.selectROI(window_name, img, showCrosshair=True)

    img = get_roi(img, t_roi)
    img = rescale(img)
    cv2.equalizeHist(img)

    if (SHOW_TRAINING_ROI):
        cv2.imshow(f'Training image', img)
        cv2.waitKey(SHOW_TRAINING_TIME)
        cv2.destroyAllWindows()
    return hog.compute(img)


def train_wood(hog, img, roi_non_wood, descriptors, desc_labs, show_roi=False):
    # train left
    counts = 2
    if roi_non_wood[0] > PATCH_SIZE[0]:
        for i in range(counts):
            h = random.randint(int(PATCH_SIZE[1]/2), 2 * PATCH_SIZE[1])
            x = random.randint(0, int(roi_non_wood[0]/2))
            w = min(random.randint(int(PATCH_SIZE[0]/2), 2 * PATCH_SIZE[0]), roi_non_wood[0] - x)
            y = random.randint(0, img.shape[0] - h)
            t_img = get_roi(img, (x, y, w, h))
            t_img = rescale(t_img)
            cv2.equalizeHist(t_img)
            if show_roi:
                cv2.imshow(f'No Animal', t_img)
                cv2.waitKey(SHOW_TRAINING_TIME)
                cv2.destroyAllWindows()
            descriptors.append(hog.compute(t_img))
            desc_labs.append(0)


def trainData(hog, pathes, rois, label: int):
    descriptors = []
    desc_labs = []
    for i, _ in enumerate(pathes):
        img = read_image(pathes[i])
        descriptors.append(getDescriptor(hog, img, rois[i]))
        desc_labs.append(label)
        # train_wood(hog, img, rois[i], descriptors, desc_labs, SHOW_TRAINING_ROI)

    return descriptors, desc_labs


if __name__ == '__main__':
    print('Training SVM model ...')
    hog = get_hog()

    # learn
    descriptors = []
    desc_labs = []
    des, labs = trainData(hog, HirschData.data[10:30], HirschData.rois[10:30], 1)
    descriptors += des
    desc_labs += labs
    des, labs = trainData(hog, DachsData.data[10:30], DachsData.rois[10:30], 2)
    descriptors += des
    desc_labs += labs

    model = svmInit()
    svmTrain(model, np.array(descriptors), np.array(desc_labs))
    print('Training SVM model done!')

    # let the user select rois and predict them
    for i in range(10):
        img = read_image(fps[random.randint(0, len(fps) - 1)])
        window_name = "Select ROI to predice ..."
        window = cv2.namedWindow(window_name, flags=cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(window_name, img)
        img = get_roi(img, roi)
        img = rescale(img)
        cv2.equalizeHist(img)
        descriptor = hog.compute(img)
        prediction = model.predict(np.array([descriptor]))[1].ravel()
        print("Prediction is ", LABEL_MAP[int(prediction[0])], "\n")

    print('done!')
    cv2.destroyAllWindows()
