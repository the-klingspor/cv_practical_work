import os
import exiftool
import shutil
from operator import itemgetter
from datetime import datetime, timedelta


"""
@author: Joschka Strüber

This script orders camera trap images into consecutive sequences by using the 
images' EXIF tags.
"""


def order_by_sequences(path_from, path_to=None):
    # read the EXIF data of all images in path_from
    et = exiftool.ExifTool()
    et.start()
    images = []     # list of tuples [(string, datetime, string), ...]
    for image_path in os.listdir(path_from):
        if image_path.endswith(('.jpg', '.JPG')):
            metadata = et.get_metadata(os.path.join(path_from, image_path))
            images.append((metadata['MakerNotes:SerialNumber'],
                           datetime.strptime(metadata['EXIF:CreateDate'],
                                             '%Y:%m:%d %H:%M:%S'),
                           metadata['File:FileName']))
    et.terminate()
    images.sort(key=itemgetter(0, 1))
    # split them into sequences based on their time and copy them into path_to
    seq_start = 0
    seq_serial_number = images[0][0]
    seq_number = 0
    for counter, image in enumerate(images):
        if counter == 0:
            continue
        timediff = images[counter-1][1] - image[1]
        # copy sequence if serial number changes or diff >10min
        if image[0] != seq_serial_number or \
                not(timedelta(minutes=-10) < timediff < timedelta(minutes=10)):
            copy_sequence(seq_number, path_from, path_to, images, seq_start,
                          counter)
            seq_start = counter
            seq_serial_number = image[0]
            seq_number += 1
    copy_sequence(seq_number, path_from, path_to, images, seq_start,
                  len(images))


def copy_sequence(seq_number, path_from, path_to, images, start, end):
    path_to_seq = os.path.join(path_to, "seq_" + str(seq_number))
    os.mkdir(path_to_seq)
    for i in range(start, end):
        path_from_image = os.path.join(path_from, images[i][2])
        path_to_image = os.path.join(path_to_seq, images[i][2])
        shutil.copyfile(path_from_image, path_to_image)


order_by_sequences("/home/joschi/Documents/testDDD/dama_dama_damhirsch/dayvision",
                   "/home/joschi/Documents/testDDD_seq/dama_dama_damhirsch/dayvision")

order_by_sequences("/home/joschi/Documents/testDDD/meles_meles_dachs/dayvision",
                   "/home/joschi/Documents/testDDD_seq/meles_meles_dachs/dayvision")
