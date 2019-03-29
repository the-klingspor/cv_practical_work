import numpy as np
from sklearn.model_selection import train_test_split
import os
from matplotlib import pyplot as plt, patches
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from functools import reduce
import pickle

"""Sliding-Windows using Principle component analysis author:
 Sufian Zaabalawi"""

target_names = ['deer', 'badger', 'empty']
patch_size = (100, 100)
vec_patch_size = reduce((lambda x, y: x * y), patch_size)


def rescalePatch(img, scale=patch_size):
    """
   rescale image patch using cubic interpolation
   @:param image to rescale
   @:param scale is target scale
   @:return rescaled image
    :author: Sufian Zaabalawi
    """
    return cv2.resize(img, scale, interpolation=cv2.INTER_CUBIC)


def slidingWindows(image, w, h, stepSize=None):
    """
    calculate sliding-windows with different scales
     @:param image to rescale
    :returns a tuple of image patch width height and its original postion in image
     :author: Sufian Zaabalawi
     """
    factor = 0.8
    width, height = image.shape
    while factor > 0.1:
        new_w = int(h * factor)
        new_h = int(h * factor)
        step_w = int(new_w / 4)
        step_h = int(new_w / 4)
        for y in range(0, width - new_w, step_w):
            for x in range(0, height - new_h, step_h):
                yield (x, y, new_w, new_h, image[y:y + new_w, x:x + new_h])
        factor = factor - 0.1


def localisation(knn, pca, im, target=''):
    """
         localization using the classification method KNN and PCA transformation
         @:param knn Model
         @:param pca Model
         @:param im image to classify
         @:param target searching for target label in image
          :author: Sufian Zaabalawi
          """
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    fig, ax = plt.subplots(1)
    roi = []
    closest_distance = -1
    sliding_window = slidingWindows(im, im.shape[0], im.shape[1])
    for x, y, w, h, image in sliding_window:
        # Projiziere das Bild in den r-Unterraum
        proj = pca.transform(np.reshape(rescalePatch(image), (vec_patch_size)).reshape(1, -1))

        # Finde den nächsten Nachbarn zwischen den projizierten Trainingsbildern Y und proj
        prediction = knn.predict(proj.reshape(1, -1))[0]
        p = (np.amax(knn.predict_proba(proj.reshape(1, -1))))
        dist = knn.kneighbors(proj.reshape(1, -1), n_neighbors=1, return_distance=True)[0][0][0]
        if target == target_names[prediction] and p == 1.0 and closest_distance > dist or closest_distance is -1:
            closest_distance = dist
            roi.append((x, y, w, h))
    x, y, w, h = mergeRois(roi)
    ax.imshow(im, cmap='gray')
    rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    fig.show()


def load_images(paths):
    """
       loads images from folders and assigns labels according to their order
       @:param array of paths
       :returns the whole data set as data and labels
       :author: Sufian Zaabalawi
       """
    data = []
    labels = []
    for path in paths:
        for file in os.listdir(path):
            im = plt.imread(os.path.join(path, file))
            if len(im.shape) == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = np.reshape(rescalePatch(im), (vec_patch_size))
            data.append(im)
            labels.append(paths.index(path))
    return np.array(data), np.array(labels)


def showPCA(reduced_data, cluster=5):
    """
        visualize PC1 and PC2 from pca-model in Voronoi Diagram
        @:param reduced_data PCA space
        :author: Sufian Zaabalawi
        """
    kmeans = KMeans(init='k-means++', n_clusters=cluster, n_init=3)
    kmeans.fit(reduced_data[:, :2])
    h = .02
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap='Blues',
               aspect='auto', origin='lower', shape='')
    col = kmeans.labels_.astype(float)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=col, s=2 * (col + 1), marker='s')
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='+', s=70, linewidths=20,
                c='r', zorder=3)
    plt.title(' vs '.join([str(label) for label in target_names]) + ' dataset')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def mergeRois(roi):
    """
       merge array of bounding boxes to a single bounding box
       @:param roi array of bounding boxes
       :returns x, y, w, h of a bounding box
       :author: Sufian Zaabalawi
       """
    x = 0
    y = 0
    xEnd = 0
    yEnd = 0
    for i, j, k, l in roi:
        if i < x or x == 0:
            x = i
        if j < y or y == 0:
            y = j
        if i + k > xEnd or xEnd == 0:
            xEnd = i + k
        if j + l > yEnd or yEnd == 0:
            yEnd = j + l
    h = abs(yEnd - y)
    w = abs(xEnd - x)
    return x, y, w, h


def TrainingsPhase():
    """
       Train a new model or load already trained model from *.sav files
      :returns trained model with decider
       :author: Sufian Zaabalawi
       """
    pcaPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pca.sav')
    knnPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'knn.sav')
    try:
        return pickle.load(open(knnPath, 'rb')), pickle.load(open(pcaPath, 'rb'))
    except FileNotFoundError as err:
        print('Model nicht gefunden..Training ist angefangen')
        # Phase I: Lade die Trainingsbilder. Die Schnittbilder sollen in Quadrate das Tier enthalten
        # redefine target_names accordingly
        pathFirstLabel = "Pfad für Schnittbilder von Label deer"
        pathSecondLabel = "Pfad für Schnittbilder von Label badger"
        pathThirdLabel = "Pfad für Schnittbilder von Label empty"
        x_data, y_data = load_images([pathFirstLabel, pathSecondLabel, pathThirdLabel])
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=0)

        pca = PCA(n_components=150, svd_solver='randomized',
                  whiten=True).fit(X_train)
        pca_train = pca.transform(X_train)

        # Initialisierung von K-Nächste Nachbar
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(pca_train, y_train)

        # Visualisierung von Trainingsbilder in Voronoi Diagram
        showPCA(pca_train, 3)

        pickle.dump(pca, open(pcaPath, 'wb'))
        pickle.dump(knn, open(knnPath, 'wb'))
        return knn, pca


if __name__ == '__main__':
    # Phase I: Initialisierung
    knn, pca = TrainingsPhase()

    # Phase II: Lokalisirung Testen
    imagesPaths = ['image', 'image..']
    for path in imagesPaths:
        image = plt.imread(path)[33:1465, :, :]  # Crop header and footer with 33, 1465 respectively
        localisation(knn, pca, image, 'deer')  # possible labels ['deer', 'badger', 'empty']
