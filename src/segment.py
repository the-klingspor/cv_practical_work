import os

import cv2
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as sci
from skimage import exposure as eq
from skimage.feature import local_binary_pattern
from skimage.segmentation import active_contour

from background_separation import pca
from background_separation import rpca


def segment(path, label, root_out_path = '/home/tp/Downloads/CVSequences/CVSequences/damhirsch/out/',
            write_images_with_roi = False):
    data = dict()
    print(f"Segmenting {label}")
    for folder in os.listdir(path):
        if folder:
            seqPath = os.path.join(path, folder)
            imageList = os.listdir(seqPath)
            p = len(imageList)
            print(f'Processing sequence {folder} with {p} images')
            x, y, z = plt.imread(os.path.join(seqPath, imageList[0])).shape
            x -= (x - 1504) + 33
            seq = np.zeros((x, y, p))
            m = np.zeros((x * y, p))
            c = np.zeros((x, y, z, p))
            for i in range(p):
                im = plt.imread(os.path.join(seqPath, imageList[i]))[33:1504, :, :]
                c[:, :, :, i] = im
                seq[:, :, i] = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
                seq[:, :, i] = pre_processing_method_1(seq[:, :, i])
                m[:, i] = seq[:, :, i].reshape(x * y, order='F').copy()
            L1, S1 = pca(m)
            # L1, S1 = rpca(m)
            L1, S1 = reshape(L1, S1, x)
            O = hard_threshold(S1)
            L1 = None
            S1 = None
            print('Finished pca - start finding ROIs')
            for ii in range(p):
                bw = O[:, :, ii]
                bw = post_processing_method_2(bw)
                box = boundingBox(bw)
                if write_images_with_roi: # added in order control if images should be written to hd {Thomas}
                    fig, ax = plt.subplots(1)
                    ax.imshow(c[:, :, :, ii].astype(np.uint8))
                    box = show_localisation(bw, ax)
                # Only ROIs with a minimum size are allowed. This is done to prevent the results from containing empty
                # ROIs and to force a minimum size of at least 64 pixel width and 48 height to reduce the number of
                # artifacts. {Thomas}
                if box[2] > 64 and box[3] > 48:
                    data[seqPath + os.sep + imageList[ii]] = box
                    if write_images_with_roi:
                        show_segmentation(bw, ax)
                        outPath = os.path.join(root_out_path, label)
                        outPath = os.path.join(outPath, folder)
                        if not os.path.exists(outPath): os.makedirs(outPath)
                        fig.savefig(outPath + os.sep + 'seg_' + str(ii) + '.png', format='png')
                if write_images_with_roi:
                    plt.close(fig)
            print("done!")
    if not os.path.exists(root_out_path): os.makedirs(root_out_path)
    np.save(os.path.join(root_out_path, label), data)
    print(f"Data writen to file!")


def bwareafilt(mask):
    image = mask.copy()
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    res = np.zeros(image.shape, np.uint8)
    try:
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        cv2.drawContours(res, [biggest_contour], -1, 255, -1)
    except ValueError as err:
        print("ValueError: {0}. Most likely no movement was detected in bwareafilt()".format(err))
        biggest_contour = 0

    return biggest_contour, res


def boundingBox(mask):
    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mx = (0, 0, 0, 0)
    mx_area = 0
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w * h
        if area > mx_area:
            mx = x, y, w, h
            mx_area = area
    return mx


def overlay_mask(mask, image):
    mask = np.array(mask, dtype=np.uint8)
    image = np.array(image, dtype=np.uint8)
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    rgb_mask[:, :, 1:2] = 0
    img = cv2.addWeighted(rgb_mask, 0.1, image, 1, 0)
    return img


def reshape(L1, S1, x):
    xy, p = L1.shape
    y = int(xy / x)
    L = np.zeros((x, y, p))
    S = np.zeros((x, y, p))
    for ii in range(p):
        L[:, :, ii] = L1[:, ii].reshape(x, y, order='F').copy()
        S[:, :, ii] = S1[:, ii].reshape(x, y, order='F').copy()
    return L, S


def hard_threshold(S):
    x, y, z = S.shape
    beta = np.power(0.5 * (3 * np.std(S[:].reshape(x * y * z, 1))), 2)
    O = 0.5 * S ** 2 > beta
    return O


def post_processing_method_2(bw):
    kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (5, 5))

    bw = sci.binary_opening(bw, structure=np.ones((5, 5))).astype(float)
    bw = sci.gaussian_filter(bw.astype(float), sigma=7)
    bw[bw > 0.02] = 255
    bw = np.array(bw, dtype=np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    contour_points, biggest_mask = bwareafilt(bw)
    bw = biggest_mask * bw
    return bw


def post_processing_method_3(bw, im):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    bw = sci.median_filter(bw, size=2).astype(float)
    bw = np.array(bw, dtype=np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    contour_points, biggest_mask = bwareafilt(bw)
    bw = biggest_mask * bw
    snakeInit = contour_points.reshape(contour_points.shape[0], 2, order='F')
    snake = active_contour(im, snakeInit, alpha=0.01, beta=5, gamma=0.0001, w_edge=1)
    ctr = np.array(snake).reshape((-1, 1, 2)).astype(np.int32)
    snake2BW = np.zeros(bw.shape, np.uint8)
    cv2.drawContours(snake2BW, [ctr], -1, 255, -1)
    bw = cv2.morphologyEx(snake2BW, cv2.MORPH_CLOSE, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    return bw


def pre_processing_method_1(im):
    #lbp = local_binary_pattern(im, 24, 3)
    color_feature = eq.equalize_hist(im)
    #beta = 0.1
    #res = beta * lbp + (1 - beta) * color_feature
    return color_feature


def show_localisation(mask, ax):
    x, y, w, h = boundingBox(mask)
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    return x, y, w, h


def show_segmentation(mask, ax):
    contours, _ = bwareafilt(mask)
    points = contours.reshape(contours.shape[0], 2, order='F')
    ax.plot(points[:, 0], points[:, 1], '-b', lw=1)


if __name__ == '__main__':
    segment("/home/tp/Downloads/CVSequences/CVSequences/damhirsch/dayvision", 'deer')
    segment("/home/tp/Downloads/CVSequences/CVSequences/badger/dayvision", 'badger')
