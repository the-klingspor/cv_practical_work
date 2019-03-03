import os
import sys
import numpy as np
import random
import segment
import sequences


class DataProvider:
    """A class to handle all data related processes.

    This class provides easy to use access to segment.py, sequences.py and selects the amount of images that should be
    used for training and testing purposes of a classifier.
    """
    image_data_dir: str
    sequences_data_dir: str
    show_images_with_roi: bool
    segments_dir: str
    folder_names_to_process = {}
    max_training_data_percentage: float
    train_with_equal_image_amount: bool
    shuffle_data: bool
    seed: int

    _data = dict()
    _data_keys = dict()
    _is_shuffled = False
    _seed_is_set = False

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
                    data.extend(sequences.read_images(path))
            sequences.order_by_sequences(data, os.path.join(self.sequences_data_dir, animal_folder_name))

    def segment_sequences(self):
        """This function provides convenient access for segment.py

        In order to use this function create a DataProvider object using the XXXXXXX, set the 'sequences_data_dir',
        'show_images_with_roi' and 'segments_dir'. Finally call the function. After
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
        length is calculated and for each animal type this value is used to generate the predict data.
        Fürthermore the predict data can be shuffled. If 'shuffle_data' is set to True the data is shuffled. In order to
        create reproducible shuffled experiments the seed for the random generator can be set with the 'seed' attribute.

        :return A list of tuples containing (<File Name>, <ROI>, <label>). Depending on the settings for each
        the same amount of images can be added or a fraction of all available image of a animal are returned.
        """
        if not self._data:
            self._read_segmentation_data()
        training_data = []
        if self.train_with_equal_image_amount:
            min_number = self._get_min_test_data_length()
            for animal_dict in self._data_keys.items():
                label = animal_dict[0]
                image_names = animal_dict[1]
                image_names = image_names[: min_number]
                for image_name in image_names:
                    training_data.append((image_name, self._data[label][image_name], label))
        else:
            for animal_dict in self._data_keys.items():
                label = animal_dict[0]
                image_names = animal_dict[1]
                image_names = image_names[: int(len(image_names) * self.max_training_data_percentage)]
                for image_name in image_names:
                    training_data.append((image_name, self._data[label][image_name], label))
        return training_data

    def get_test_data(self):
        """Provides a list of tuples containing all data not used for training

        :return A list of tuples containing (<File Name>, <ROI>, <label>). For each animal all remaining images not
        used tor training are returned.
        """
        if not self._data:
            self._read_segmentation_data()
        test_data = []
        if self.train_with_equal_image_amount:
            min_number = self._get_min_test_data_length()
            for animal_dict in self._data_keys.items():
                label = animal_dict[0]
                image_names = animal_dict[1]
                image_names = image_names[min_number + 1:]
                for image_name in image_names:
                    test_data.append((image_name, self._data[label][image_name], label))
        else:
            for animal_dict in self._data_keys.items():
                label = animal_dict[0]
                image_names = animal_dict[1]
                image_names = image_names[int(len(image_names) * self.max_training_data_percentage) + 1:]
                for image_name in image_names:
                    test_data.append((image_name, self._data[label][image_name], label))
        return test_data

    def _get_min_test_data_length(self):
        """Private function to calculate the amount of images that are selected for the animal with the lowest image amount.
        """
        min_number = float('inf')
        for animal_dict in self._data_keys.items():
            image_names = animal_dict[1]
            temp_min_number = int(len(image_names) * self.max_training_data_percentage)
            if min_number > temp_min_number:
                min_number = temp_min_number
        return min_number

    def _shuffle_data(self):
        """Private function to shuffle the data.

        This function is called if the boolean 'shuffle_data' is set"""
        if self.seed != 0 and not self._seed_is_set:
            self._seed_is_set = True
            random.seed(self.seed)
        elif not self._seed_is_set:
            self._seed_is_set = True
            # use a random but KNOWN seed!
            self.seed = random.randrange(sys.maxsize)
            random.seed(self.seed)
            print("Seed was:", self.seed)
        for dictionary in self._data_keys.items():
            label = dictionary[0]
            keys = dictionary[1]
            random.shuffle(keys)

    def __init__(self, image_data_dir: str, sequences_data_dir: str,
                 segments_dir: str, show_images_with_roi: bool,
                 folder_names_to_process, max_training_data_percentage: float,
                 train_with_equal_image_amount: bool, shuffle_data: bool, seed: int):
        """Initialises an DataProvider object

        :param image_data_dir: image data / location of the animal folders that contains all unordered images
        :param sequences_data_dir: The path were the images sorted into sequences should be stored
        :param segments_dir: The path were the ROIs correlated to image file names are stored. (The *.npy files)
        :param show_images_with_roi: If the ROIs should be printed to the output. They are stored were the *.npy files are generated
        :param folder_names_to_process: the subfoldernames that are used for sequence separations
        :param max_training_data_percentage: the maximum % of images of a kind that are used as training data
        :param train_with_equal_image_amount: If any animal should be trained with equal amount of images
        :param shuffle_data: if the images should be shuffled
        :param seed: the random seed for shuffle. If 0 is choosen the seed is random too. Any other number can be choosen to increase the reproducibility of the experiment
        """
        self.image_data_dir = image_data_dir
        self.sequences_data_dir = sequences_data_dir
        self.segments_dir = segments_dir
        self.show_images_with_roi = show_images_with_roi
        self.folder_names_to_process = folder_names_to_process
        self.max_training_data_percentage = max_training_data_percentage
        self.train_with_equal_image_amount = train_with_equal_image_amount
        self.shuffle_data = shuffle_data
        self.seed = seed

        print("Done")

    def _read_segmentation_data(self):
        """Private helper function to read data from *.npy files

        This function is called if there is a attempt to retrive image data. This function also manages if the data
        still needs to be shuffled"""
        for npy_file in os.listdir(self.segments_dir):
            if npy_file.endswith(".npy"):
                label = npy_file.split(".")[0]
                self._data[label] = np.load(os.path.join(self.segments_dir, npy_file)).item()
                self._data_keys[label] = list(self._data[label].keys())
        if not self._is_shuffled and self.shuffle_data:
            self._shuffle_data()
            self._is_shuffled = True

if __name__ == '__main__':
    # This is an usage example and not supposed to be run as main. It just provides an easy manual predict case
    provider = DataProvider("/home/tp/Downloads/CVSequences/data", # image data / location of the animal folders that contains all unordered images
        "/home/tp/Downloads/CVSequences/sequ", # The path were the images sorted into sequences should be stored
        "/home/tp/Downloads/CVSequences/npy", # The path were the ROIs correlated to image file names are stored. (The *.npy files)
        True, # If the ROIs should be printed to the output. They are stored were the *.npy files are generated
        {"dayvision", "day"}, # the subfolder names that are used for sequence separations
        0.4, # the maximum % of images of a kind that are used as training data
        True, # If any animal should be trained with equal amount of images
        True, # if the images should be shuffled
        123) # the random seed for shuffle. If 0 is choosen the seed is random too. Any other number can be choosen to increase the reproducibility of the experiment

    # perform sequenc seperation
    provider.generate_sequences()
    # perform segmentation
    provider.segment_sequences()
    #get training tuples
    training_images = provider.get_training_data()
    # get predict tuples
    test_images = provider.get_test_data()
    pass



