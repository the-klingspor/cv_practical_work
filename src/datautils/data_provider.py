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
import segment
import sequences


class DataProvider:
    """
    The root working directory. As default this folder should be the root folder for the unprocessed data, the separated
    sequences, the numpy files with the rois as well as the output images visualizing rois and showing labels. However,
    it is possible to separate these folders and locate them on independent directories.
    """
    working_dir = ""
    image_data_dir = ""
    sequences_data_dir = ""
    images_with_roi_dir = ""
    show_images_with_roi = False
    labeled_images_with_roi_dir = ""
    segments_dir = ""
    folder_names_to_process = {}
    max_training_data_percentage = 0.0
    train_with_equal_image_amount = True
    shuffle_data = True
    seed = 0
    data = dict()
    data_keys = dict()

    def sequence(self):
        """This function provides convenient access for sequence.py

        In order to use this function create a DataProvider object using the XXXXXXX, set the 'image_data_dir', the
        'sequences_data_dir' and the 'folder_names_to_progress' and then call the function. After
        calling this function every and subfolder of the 'image_data_dir' will be progressed. The 'image_data_dir'
        should contain seperate folders for each animal that is provided. Than each of the animal subfolder names will
        be tested if present in the 'folder_names_to_progress' list. If it is present all containing images will be
        added to a data file, which final will be passed on to sequences.py which than should be seperate the images
        into sequences. These sequences should be stored under 'sequences_data_dir'

        E.g.: in the 'image_data_dir' is a sub folder 'damhirsch'. And let assume 'damhirsch' contains a folder
        'dayvision', 'nightvision' and 'empty'. Lets further asume 'empty' contains the folder 'day' and 'night'.
        Furthermore the list 'folder_names_to_progress' contains {"dayvision", "empty", "day"}. Than will all images in
        the pathes image_data_dir/damhirsch/dayvision, image_data_dir/damhirsch/empty and
        image_data_dir/damhirsch/empty/day be selected and passed to sequences.py as well with the output path for the
        sequences 'sequences_data_dir'

        No data is returned by this function. Currently the sequence.py writes the sequences into separate folder on the
        hard disk.
        """
        for animal_folder_name in os.listdir(self.image_data_dir):
            print(f"Processing folder {animal_folder_name}")
            animal_folder_path = os.path.join(self.image_data_dir, animal_folder_name)
            data = []
            for path, _, _ in os.walk(animal_folder_path):
                last_folder_name = path.split(os.sep)
                last_folder_name = last_folder_name[len(last_folder_name)-1]
                if last_folder_name in self.folder_names_to_process:
                    data.extend(sequences.read_images(os.path.join(animal_folder_path, last_folder_name)))
            sequences.order_by_sequences(data, os.path.join(self.sequences_data_dir, animal_folder_name))

    def segment(self):
        """This function provides convenient access for segment.py

        In order to use this function create a DataProvider object using the XXXXXXX, set the 'sequences_data_dir',
        'images_with_roi_dir', 'show_images_with_roi' and 'segments_dir'. Finally call the function. After
        calling this function every subfolder of the 'sequences_data_dir' will treated as label for a animal. Than each
        of the animal subfolders, which should contain the sequences, will processed using the segment.py code.
        As output numpy arrays will be written to the 'segments_dir' folder and contain the image pathes and the
        ROIS detected in segment.py. The file name is the label of the animal.
        """
        for animal_folder_name in os.listdir(self.sequences_data_dir):
            path = os.path.join(self.sequences_data_dir, animal_folder_name)
            if os.path.isdir(path):
                print(f"Processing folder {path}")
                segment.segment(path, animal_folder_name, self.segments_dir)

    def get_training_data(self):
        pass

    def get_test_data(self):
        pass

    def _shuffle_data(self):
        if self.seed != 0:
            random.seed(self.seed)
        for dictionary in self.data_keys.items():
            label = dictionary[0]
            keys = dictionary[1]
            random.shuffle(keys)

    def __init__(self, npy_folder_path, max_percentage=0.5, train_with_equal_amount=True, shuffle_data=True,
                 random_Seed=0):
        self.npy_folder_path = npy_folder_path
        self._max_percentage = max_percentage
        self._train_with_equal_amount = train_with_equal_amount
        self.shuffle_data = shuffle_data
        self.seed = random_Seed

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
