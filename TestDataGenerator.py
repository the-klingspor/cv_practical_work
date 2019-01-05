import glob
import cv2

FOLDER_PATH = "images/hirsch"

filenames = glob.glob(f"{FOLDER_PATH}/*")
img_paths = []
rois = []

for img_path in filenames:
    img = cv2.imread(img_path)
    window_name = f"Select ROI for {img_path}"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    roi = cv2.selectROI(window_name, img)
    cv2.destroyAllWindows()
    img_paths.append(img_path)
    rois.append(roi)

print(f"Writing to Python file ...")
file = open("data.py", "w")
file.write(f"data = [ \n")
for img_path in img_paths:
    if img_path == img_paths[len(img_paths)-1]:
        file.write(f"   '{img_path}' \n")
    else:
        file.write(f"   '{img_path}',  \n")

file.write("] \n")
file.write(" \n")
file.write("rois = [ \n")
for roi in rois:
    if roi == rois[len(rois)-1]:
        file.write(f"   ({roi[0]}, {roi[1]}, {roi[2]}, {roi[3]}) \n")
    else:
        file.write(f"   ({roi[0]}, {roi[1]}, {roi[2]}, {roi[3]}),  \n")

file.write("]")
file.write("")
file.close()




