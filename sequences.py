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

def read_images(path, empty=False):
    if not os.path.exists(path):
        print("Input directory '{}' does not exist".format(path))
        return
    # read the EXIF data of all images in path_from
    et = exiftool.ExifTool()
    et.start()
    images = []     # list of tuples [(string, datetime, string), ...]
    for image_path in os.listdir(path):
        if image_path.endswith(('.jpg', '.JPG')):
            metadata = et.get_metadata(os.path.join(path, image_path))
            images.append((metadata['MakerNotes:SerialNumber'],
                           datetime.strptime(metadata['EXIF:CreateDate'],
                                             '%Y:%m:%d %H:%M:%S'),
                           os.path.join(path, metadata['File:FileName']),
                           empty))
    et.terminate()
    return images


def order_by_sequences(images, path_to):
    if not os.path.exists(path_to):
        print("Output directory '{}' does not exist.".format(path_to))
        return

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
            copy_sequence(seq_number, path_to, images, seq_start,
                          counter)
            seq_start = counter
            seq_serial_number = image[0]
            seq_number += 1
    # copy last sequence as well
    copy_sequence(seq_number, path_to, images, seq_start,
                  len(images))


def copy_sequence(seq_number, path_to, images, start, end):
    if not os.path.exists(path_to):
        print("Output directory '{}' does not exist.".format(path_to))
        return
    path_to_seq = os.path.join(path_to, "seq_" + str(seq_number))
    os.mkdir(path_to_seq)
    for i in range(start, end):
        path_from_image = images[i][2]
        path_to_image = os.path.join(path_to_seq, os.path.basename(images[i][2]))
        shutil.copyfile(path_from_image, path_to_image)

damhirsch_images = read_images("/home/joschi/Documents/testDDD/dama_dama_damhirsch/dayvision")
damhirsch_empty = read_images("/home/joschi/Documents/testDDD/dama_dama_damhirsch/empty/day")
damhirsch_images.extend(damhirsch_empty)
order_by_sequences(damhirsch_images,
                   "/home/joschi/Documents/testDDD_seq/dama_dama_damhirsch/dayvision")
