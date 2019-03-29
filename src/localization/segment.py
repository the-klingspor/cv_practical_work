import os

import cv2
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as sci
from skimage import exposure as eq
from src.localization.background_separation import foreground

"""Localization and Segmentation using Background-foreground with PCA and low-rank approximation:
 Sufian Zaabalawi"""


def segment(path, label, root_out_path='C:\\Users\Sufian\Downloads\data\DDD\out\\'):
    data = dict()
    print(f"Segmenting {label}")
    for folder in os.listdir(path):
        if folder:
            seqPath = os.path.join(path, folder)
            imageList = os.listdir(seqPath)
            for file in imageList:
                # ignore text file with empty information
                if os.path.basename(file) == "empty.txt":
                    imageList.remove(file)
            p = len(imageList)
            print(f'Processing sequence {folder} with {p} images')

            # retrieve list of all empty images
            empty_file_list = []
            empty_path = os.path.join(seqPath, "empty.txt")
            if os.path.exists(empty_path):
                with open(empty_path, 'r') as empty_file:
                    empty_file_list = empty_file.readlines()

            x, y, _ = plt.imread(os.path.join(seqPath, imageList[0])).shape
            x -= (x - 1504) + 33
            M = np.zeros((x, y, p))

            empty_index = []
            for i in range(p):
                image_path = os.path.join(seqPath, imageList[i])

                # save if i-th image is empty or not
                if image_path in empty_file_list:
                    empty_index.append(True)
                else:
                    empty_index.append(False)

                im = plt.imread(image_path)[33:1504, :, :]
                M[:, :, i] = pre_processing_method_1(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            M = np.reshape(M, (x * y, p)).copy()
            S = foreground(M, 1)
            S = np.reshape(S, (x, y, p)).copy()
            O = hard_threshold(S)
            S = None
            M = None
            print('Finished low-rank separation - start finding ROIs')
            for ii in range(p):
                # append only non-empty images to file with regions of interest
                if not empty_index[ii]:
                    bw = O[:, :, ii]
                    bw = post_processing_method_2(bw)
                    # fig, ax = plt.subplots(1)
                    # ax.imshow(bw, cmap='gray')
                    # show_localisation(bw, ax)
                    # show_segmentation(bw, ax)
                    # plt.show()
                    x, y, w, h = boundingBox(bw)
                    # outPath = os.path.join(os.path.join(root_out_path, label), folder)
                    # if not os.path.exists(outPath): os.makedirs(outPath)
                    # fig.savefig(outPath + os.sep + 'seg_' + str(ii) + '.jpg', format='jpg')
                    if w > 64 and h > 48:
                        data[seqPath + os.sep + imageList[ii]] = x, y, w, h
            print("Done!")
    if not os.path.exists(root_out_path):
        os.makedirs(root_out_path)
    np.save(os.path.join(root_out_path, label), data)
    print(f"Data written to file!")


"""
calculate the biggest connected mask
@:returns biggest mask
:author: Sufian Zaabalawi
"""


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


"""
calculate bounding box of a mask
@:returns x, y, w, h are postion of the box and its 
:author: Sufian Zaabalawi
"""


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


def hard_threshold(S):
    """
    apply Hardthresh holding on foreground image S
    :author: Sufian Zaabalawi
    """
    x, y, z = S.shape
    beta = np.power(0.5 * (3 * np.std(S[:].reshape(x * y * z, 1))), 2)
    O = 0.5 * S ** 2 > beta
    return O


def post_processing_method_2(bw):
    """
    Post processing pipeline to extract the mask from foreground image
    :author: Sufian Zaabalawi
    """
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


def pre_processing_method_1(im):
    """
       pre processing pipeline before applying background foreground separation
       :author: Sufian Zaabalawi
       """
    color_feature = eq.equalize_hist(im)
    return color_feature


def show_localisation(mask, ax):
    """
       visualize localization bounding box
       :author: Sufian Zaabalawi
       """

    x, y, w, h = boundingBox(mask)
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    return x, y, w, h


def show_segmentation(mask, ax):
    """
    visualize segmentation bounding box
    :author: Sufian Zaabalawi
    """

    contours, _ = bwareafilt(mask)
    points = contours.reshape(contours.shape[0], 2, order='F')
    ax.plot(points[:, 0], points[:, 1], '-b', lw=1)


if __name__ == '__main__':
    segment("C:\\Users\Sufian\Downloads\data\DDD\ordered", 'deer')
    # segment("/home/tp/Downloads/CVSequences/CVSequences/badger/dayvision", 'badger')
