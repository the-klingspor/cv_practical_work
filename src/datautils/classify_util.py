import numpy as np


def get_roi(img, roi):
    """
    Extract the bounding box region of interest for a given image as numpy array
    and the coordinates of the roi as left x coordinate, lower y coordinate,
    width and height.

    :author: Thomas Poschadel, Joschka Strüber
    """
    y = roi[1]
    x = roi[0]
    w = roi[2]
    h = roi[3]
    return img[y:y + h, x:x + w]


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
