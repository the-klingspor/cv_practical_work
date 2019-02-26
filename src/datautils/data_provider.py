import os
import numpy as np
import random
import segment
import sequences


class DataProvider:
    """A class to handle all data related processes.

    This class provides easy to use access to segment.py, sequences.py and selects the amount of images that should be
    used for training and testing purposes of a classifier.
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

    def generate_sequences(self):
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

    def segment_sequences(self):
        """This function provides convenient access for segment.py

        In order to use this function create a DataProvider object using the XXXXXXX, set the 'sequences_data_dir',
        'images_with_roi_dir', 'show_images_with_roi' and 'segments_dir'. Finally call the function. After
        calling this function every subfolder of the 'sequences_data_dir' will treated as label for a animal. Than each
        of the animal subfolders, which should contain the sequences, will processed using the segment.py code.
        As output numpy arrays will be written to the 'segments_dir' folder and contain the image paths and the
        ROIS detected in segment.py. The file name is the label of the animal.
        """
        for animal_folder_name in os.listdir(self.sequences_data_dir):
            path = os.path.join(self.sequences_data_dir, animal_folder_name)
            if os.path.isdir(path):
                print(f"Processing folder {path}")
                segment.segment(path, animal_folder_name, self.segments_dir, self.show_images_with_roi)

    def get_training_data(self):
        """Provides the training data fraction of the entire data available

        This method returns the training data fraction of the entire data available as a list of tuples. Each tuple
        consist out of the entries (<File Name>, <ROI>, <label>). <File Name> represents the file location; <ROI> is
        a tuple (x, y, width, height); And <label> is the animal name.
        The data size depends on the 'max_training_data_percentage' value. Using this value the fraction of all
        available images of one animal type is calculated and added to the training data list. If equal training size is
        required the boolean 'train_with_equal_image_amount' can be set to true. In this case smallest training data
        length is calculated and for each animal type this value is used to generate the test data.
        Fürthermore the test data can be shuffled. If 'shuffle_data' is set to True the data is shuffled. In order to
        create reproducible shuffled experiments the seed for the random generator can be set with the 'seed' attribute.

        :return A list of tuples containing (<File Name>, <ROI>, <label>). Depending on the settings for each
        the same amount of images can be added or a fraction of all available image of a animal are returned.
        """
        training_data = []
        if self.train_with_equal_image_amount:
            min_number = self._get_min_test_data_length()
            for animal_dict in self.data_keys.items():
                label = animal_dict[0]
                image_names = animal_dict[1]
                image_names = image_names[: min_number]
                for image_name in image_names:
                    training_data.append((image_name, self.data[label][image_name], label))
        else:
            for animal_dict in self.data_keys.items():
                label = animal_dict[0]
                image_names = animal_dict[1]
                image_names = image_names[: int(len(image_names) * self.max_training_data_percentage)]
                for image_name in image_names:
                    training_data.append((image_name, self.data[label][image_name], label))
        return training_data

    def get_test_data(self):
        """Provides a list of tuples containing all data not used for training

        :return A list of tuples containing (<File Name>, <ROI>, <label>). For each animal all remaining images not
        used tor training are returned.
        """
        test_data = []
        if self.train_with_equal_image_amount:
            min_number = self._get_min_test_data_length()
            for animal_dict in self.data_keys.items():
                label = animal_dict[0]
                image_names = animal_dict[1]
                image_names = image_names[min_number + 1:]
                for image_name in image_names:
                    test_data.append((image_name, self.data[label][image_name], label))
        else:
            for animal_dict in self.data_keys.items():
                label = animal_dict[0]
                image_names = animal_dict[1]
                image_names = image_names[int(len(image_names) * self.max_training_data_percentage) + 1:]
                for image_name in image_names:
                    test_data.append((image_name, self.data[label][image_name], label))
        return test_data

    def _get_min_test_data_length(self):
        """Private function to calculate the amount of images that are selected for the animal with the lowest image amount.
        """
        min_number = float('inf')
        for animal_dict in self.data_keys.items():
            image_names = animal_dict[1]
            temp_min_number = int(len(image_names) * self.max_training_data_percentage)
            if min_number > temp_min_number:
                min_number = temp_min_number
        return min_number

    def _shuffle_data(self):
        """Private function to shuffle the data.

        This function is called if the boolean 'shuffle_data' is set"""
        if self.seed != 0:
            random.seed(self.seed)
        for dictionary in self.data_keys.items():
            label = dictionary[0]
            keys = dictionary[1]
            random.shuffle(keys)

    def __init__(self, npy_folder_path, max_percentage=0.5, train_with_equal_amount=True, shuffle_data=False,
                 random_Seed=0):
        self.npy_folder_path = npy_folder_path
        self.max_training_data_percentage = max_percentage
        self.train_with_equal_amount = train_with_equal_amount
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
    # provider.train_with_equal_image_amount = False
    training_data = provider.get_training_data()
    test_data = provider.get_test_data()
    pass
