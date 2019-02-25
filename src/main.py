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
    root = WORKING_DIR
    pattern = "*.py"

    for path, subdirs, files in os.walk(root):
        pass

