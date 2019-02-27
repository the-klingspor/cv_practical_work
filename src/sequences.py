import os
import shutil

from operator import itemgetter
from datetime import datetime, timedelta
from pathlib import Path

from src.exiftool import ExifTool

"""
@author: Joschka Strüber

This script orders camera trap images into consecutive sequences by using the 
images' EXIF tags.
"""

# Constants for access of elements in an image list
SN = 0
CREATION = 1
PATH = 2
EMPTY = 3

# Digital image file types
TYPES = ('.jpg',
         '.JPG',
         '.png',
         '.PNG',
         '.tif',
         '.TIF')

def read_images(path, empty=False):
    """
    Loads the relevant EXIF metadata for all image files in the given directory.
    Returns this data in a list or an empty list, if the directory did not exist
    or if it contained no image files.

    :author: Joschka Strüber
    :param path: Path of the directory with image files, which EXIF tags will
    be read and returned.
    :param empty: Boolean (default = False)
        Information if this image is empty or contains a relevant subject (e.g.
        an animal in camera trap images).
    :return: List of tuples with a serial number, date of creation, filename and
    the empty information of all image files in the given path.
    """
    if not os.path.exists(path):
        print("Input directory '{}' does not exist".format(path))
        return []
    # read the EXIF data of all images in path
    et = ExifTool()
    et.start()
    images = []     # list of tuples [(string, datetime, string, boolean), ...]
    for image_path in os.listdir(path):
        if image_path.endswith(TYPES):
            metadata = et.get_metadata(os.path.join(path, image_path))
            images.append((metadata['MakerNotes:SerialNumber'],
                           datetime.strptime(metadata['EXIF:CreateDate'],
                                             '%Y:%m:%d %H:%M:%S'),
                           os.path.join(path, metadata['File:FileName']),
                           empty))
    et.terminate()
    return images

def order_db_by_sequences(path_from, path_to, copy=True, empty=True):
    """
    Orders a database of image files into sequences based on the serial number
    of the camera, the creation date and the animal directory. Images from the
    "mismatch" directory will be ignored. The user can choose if the sequences
    should be copied or moved. Additionally, a text file with the information
    whether or not an image is from the "empty" directory of a species and
    contains no animal can be written to every sequence.

    :param path_from: Path with directories of all camera trap images.
        Example:
        .../CVSequences:
            /meles_meles_dachs:
                /dayvision
                /empty:
                    /day
                    /night
                /mismatch
                /nightvision
            /dama_dama_damhirsch:
                /dayvision
                /empty:
                    /day
                    /night
                /nightvision

    :param path_to: Path where the sequences of all species will be moved to.
        Example:
        .../DDD:
            /meles_meles_dachs:
                /seq_0:
                    /empty.txt
                    /IMG0001.JPG
                    ...
                /seq_1:
                    /empty.txt
                    /IMG0010.JPG
                    ...
                ...
            /dama_dama_damhirsch:
                /seq_0:
                    /empty.txt
                    /IMG0020.JPG
                    ...
                ...
    :param copy: Boolean (default = True)
        If the image files should be copied or moved to their sequence
         directories.
    :param empty: Boolean (default = True)
        Whether or not the empty information of images will be saved in a text
        file for every sequence. If "False" is selected, the empty images will
        not be read.
    :return: None
    """
    if not os.path.exists(path_from):
        print("Input directory '{}' does not exist".format(path_from))
        return
    if not os.path.exists(path_to):
        print("Output directory '{}' does not exist".format(path_to))

    # for every animal species
    species_subdirs = [f.path for f in os.scandir(path_from) if f.is_dir()]
    for species in species_subdirs:
        # read all images that are relevant
        images = []
        dayvision_subdir = os.path.join(species, 'dayvision')
        if os.path.isdir(dayvision_subdir):
            images.extend(read_images(dayvision_subdir))

        nightvision_subdir = os.path.join(species, 'nightvision')
        if os.path.isdir(nightvision_subdir):
            images.extend(read_images(nightvision_subdir))

        if empty:
            empty_day_subdir = os.path.join(species, 'empty', 'day')
            if os.path.isdir(empty_day_subdir):
                images.extend(read_images(empty_day_subdir, empty=True))
            empty_night_subdir = os.path.join(species, 'empty', 'night')
            if os.path.isdir(empty_night_subdir):
                images.extend(read_images(empty_night_subdir, empty=True))

        species_name = os.path.dirname(species)
        path_to_species = os.path.join(path_to, species_name)

        # todo: make exception safe if directory already exists

        os.mkdir(path_to_species)
        order_by_sequences(images, path_to_species, copy, empty)


def order_by_sequences(images, path_to, copy=True, empty=True):
    """
    Orders tuples of image data into consecutive sequences based on the serial
    number of the camera and the creation date. Images from different cameras
    or with a too large time difference cannot be from the same camera trap
    sequence. Pictures of the same sequence will be copied or moved into the
    same directory, which is a subdirectory of path_to. If wanted, a text file
    with the information of every image of a sequence whether or not it is
    empty, can be written to the sequence directories.

    :author: Joschka Strüber
    :param images: List of tuples of image data: [(serial number, create date,
        file name, empty information), ...]
    :param path_to: The directory where the sequence directories will be written
        to.
    :param copy: Boolean (default = True)
        Whether the image files of the sequence should be copied or moved.
    :param empty: Boolean (default = True)
        Whether or not the empty information of images will be saved in a text
        file for every sequence.
    :return: None
    """
    if not os.path.exists(path_to):
        print("Output directory '{}' does not exist and will be created automatically.".format(path_to))
        os.makedirs(path_to)
        # return

    images.sort(key=itemgetter(SN, CREATION))
    # split them into sequences based on their time and copy them into path_to
    seq_start = 0
    seq_serial_number = images[0][SN]
    seq_number = 0
    for counter, image in enumerate(images):
        if counter == 0:
            continue
        time_diff = images[counter-1][CREATION] - image[CREATION]
        # copy sequence if serial number changes or diff > 10min
        if image[SN] != seq_serial_number or \
                not(timedelta(minutes=-10) < time_diff < timedelta(minutes=10)):
            create_sequence(seq_number, path_to, images, seq_start, counter,
                            copy, empty)
            seq_start = counter
            seq_serial_number = image[SN]
            seq_number += 1
    # copy last sequence as well
    create_sequence(seq_number, path_to, images, seq_start, len(images), copy,
                    empty)


def create_sequence(seq_number, path_to, images, start, end, copy=True,
                    empty=True):
    """
    Arrange all images that belong to the same sequence into a new directory
    named after their sequence number. The files can be moved or copied.
    Additionally, a file with information of all empty images of the sequence
    can be written to the output path.

    :author; Joschka Strüber
    :param seq_number: The sequence number which the new directory will be named
        after.
    :param path_to: The directory where the new sequence directory will be made.
    :param images: A list of tuples with image data (serial number, creation
        date, path and whether or not the image is empty and shows no animal).
    :param start: Start index of the sequence.
    :param end: Index behind the last image of the sequence.
    :param copy: Boolean (default = True)
        Whether the image files of the sequence should be copied or moved.
    :param empty: Boolean (default = True)
        Whether or not the empty information of images will be saved in a text
        file.
    :return: None
    """
    if not os.path.exists(path_to):
        print("Output directory '{}' does not exist.".format(path_to))
        return
    path_to_seq = os.path.join(path_to, "seq_" + str(seq_number))

    # todo: make exception safe if directory already exists

    os.mkdir(path_to_seq)
    for i in range(start, end):
        path_from_image = images[i][PATH]
        filename = os.path.basename(path_from_image)
        path_to_image = os.path.join(path_to_seq, filename)
        if copy:
            shutil.copyfile(path_from_image, path_to_image)
        else:
            shutil.move(path_from_image, path_to_image)
    if empty:
        empty_path = os.path.join(path_to_seq, "empty.txt")
        with open(empty_path, "w") as file:
            for i in range(start, end):
                image = images[i]
                if image[EMPTY]:
                    file_name = os.path.basename(image[PATH])
                    file.write(file_name)

if __name__ == '__main__':
    damhirsch_images = read_images("/home/tp/Downloads/CVSequences/CVSequences/damhirsch/dayvision")
    damhirsch_empty = read_images("/home/tp/Downloads/CVSequences/CVSequences/damhirsch/empty/day")
    damhirsch_images.extend(damhirsch_empty)
    order_by_sequences(damhirsch_images,
                       "/home/tp/Downloads/CVSequences/CVSequences/damhirsch/dayvision")

    badger_images = read_images("/home/tp/Downloads/CVSequences/CVSequences/badger/dayvision")
    badger_empty = read_images("/home/tp/Downloads/CVSequences/CVSequences/badger/empty/day")
    badger_images.extend(badger_empty)
    order_by_sequences(badger_images,
                       "/home/tp/Downloads/CVSequences/CVSequences/badger/dayvision")
