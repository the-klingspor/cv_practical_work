import numpy as np


def get_roi(img, roi):
    """
    Extract the bounding box region of interest for a given image as numpy array
    and the coordinates of the roi as left x coordinate, lower y coordinate,
    width and height.

    :author: Thomas Poschadel, Joschka Strüber
    """
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


def get_roi_with_aspect_ratio(img, roi, asp_ratio):
    """
    Extract the bounding box region of interest for a given image as numpy array
    and the coordinates of the roi as left x coordinate, lower y coordinate,
    width and height. But fits the ROI to have the same aspect ratio as used in asp_ratio

    :param asp_ratio: Provides the desired aspect ratio
    :author: Thomas Poschadel
    """
    x, y, w, h = roi
    current_ratio = w/h
    if current_ratio > asp_ratio:
        h_temp = int(w / asp_ratio)
        h_old = h
        y = max(int(y - (h_temp - h)/2), 0)
        h = h_temp
        pass
    else:
        w_temp = int(asp_ratio * h)
        w_old = w
        x = max(int(x - (w_temp - w)/2), 0)
        w = w_temp
        pass
    return get_roi(img, (x, y, w, h))


def map_labels_to_int(labels):
    """
    Map labels that are given as strings to integers that can be used with
    predefined classifiers that expect int labels.

    :author: Thomas Poschadel, Joschka Strüber
    """
    int_labels = []
    label_map = []
    for label in labels:
        if label not in label_map:
            label_map.append(label)
        int_label = label_map.index(label)
        int_labels.append(int_label)
    return np.array(int_labels)
