import glob
import cv2
import numpy as np
FOLDER_PATH = "/home/tp/Downloads/CVSequences/CVSequences/dama_dama_damhirsch/dayvision"

filenames = glob.glob(f"{FOLDER_PATH}/*")
data = dict()

for img_path in filenames:
    img = cv2.imread(img_path)
    window_name = f"Select ROI for {img_path}"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    x, y, w, h = cv2.selectROI(window_name, img)
    cv2.destroyAllWindows()
    data[img_path] = x, y, w, h

print(f"Writing to npy file ...")
np.save("/home/tp/Downloads/CVSequences/CVSequences/dama_dama_damhirsch/dayvision/manually_damhirsch", data)




