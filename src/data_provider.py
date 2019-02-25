# class
# gets a folder that contains *.npy
# file contains pathes to image files and ROIS. The file name is the label
# the user can select the max percentage of the images are used
# he can also select if the same amount of images per class should be used
# he than has a next function, that provides the possibility of iterate over every image that should be analyzed
# in addtion ther should be a function that grants them all at once
# maybe provide a evaluate  function
import os
import numpy as np
import random


class DataProvider:

    npy_folder_path = ""
    max_percentage = 0.0
    train_with_equal_amount = True
    shuffle_data = True
    random_seed = 0
    data = dict()
    data_keys = dict()


    def _shuffle_data(self):
        if self.random_seed != 0:
            random.seed(self.random_seed)
        for dictionary in self.data_keys.items():
            label = dictionary[0]
            keys = dictionary[1]
            random.shuffle(keys)


    def __init__(self, npy_folder_path, max_percentage = 0.5, train_with_equal_amount = True, shuffle_data = True, random_Seed = 0):
        self.npy_folder_path = npy_folder_path
        self.max_percentage = max_percentage
        self.train_with_equal_amount = train_with_equal_amount
        self.shuffle_data = shuffle_data
        self.random_seed = random_Seed

        for npy_file in os.listdir(self.npy_folder_path):
            if npy_file.endswith(".npy"):
                label = npy_file.split(".")[0]
                self.data[label] = np.load(os.path.join(self.npy_folder_path, npy_file)).item()
                self.data_keys[label] = list(self.data[label].keys())

        if self.shuffle_data:
            self._shuffle_data()
        print("Done")


if __name__ == '__main__':
    provider = DataProvider("/home/tp/Downloads/CVSequences/out")



