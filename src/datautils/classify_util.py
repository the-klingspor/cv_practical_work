import numpy as np


def get_roi(img, roi):
    y = roi[1]
    x = roi[0]
    w = roi[2]
    h = roi[3]
    return img[y:y + h, x:x + w]


def map_labels_to_int(labels):
    int_labels = []
    label_map = []
    for label in labels:
        if label not in label_map:
            label_map.append(label)
        int_label = label_map.index(label)
        int_labels.append(int_label)
    return np.array(int_labels)
