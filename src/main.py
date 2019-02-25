import sequences
import segment
import os
from enum import Enum, unique

@unique
class Task(Enum):
    SEQUENCES = 0
    SEGMENT = 1
    LEARN = 2
    CLASSIFY = 3


WORKING_DIR = "/home/tp/Downloads/CVSequences/"
DATA_DIR = os.path.join(WORKING_DIR, "data")
SEQUENCES_DIR = os.path.join(WORKING_DIR, "sequences")
OUTPUT_DIR = os.path.join(WORKING_DIR, "out")
TASK = Task.SEGMENT
FOLDERS_TO_PROCESS = {"dayvision", "empty"}


if __name__ == '__main__':
    if TASK == Task.SEQUENCES:
        for animal_folder_name in os.listdir(DATA_DIR):
            print(f"Processing folder {animal_folder_name}")
            path = os.path.join(DATA_DIR, animal_folder_name)
            data = []
            for folder_name in os.listdir(path):
                if folder_name in FOLDERS_TO_PROCESS:
                    if folder_name == "empty":
                        data.extend(sequences.read_images(os.path.join(path, folder_name + os.sep + 'day')))
                    else:
                        data.extend(sequences.read_images(os.path.join(path, folder_name)))
            sequences.order_by_sequences(data, os.path.join(SEQUENCES_DIR, animal_folder_name))

    elif TASK == Task.SEGMENT:
        for animal_folder_name in os.listdir(SEQUENCES_DIR):
            path = os.path.join(SEQUENCES_DIR, animal_folder_name)
            if os.path.isdir(path):
                print(f"Processing folder {path}")
                segment.segment(path, animal_folder_name, OUTPUT_DIR)

    elif TASK == Task.LEARN:
        pass

    elif TASK == Task.CLASSIFY:
        pass

