from skimage import exposure
from skimage import feature
import cv2
img = cv2.imread("/home/tp/Downloads/CVSequences/CVSequences/IMG_0534.JPG",cv2.IMREAD_GRAYSCALE)
# (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(312, 312), cells_per_block=(2, 2), transform_sqrt=True, block_norm = "L1",visualize = True)
# hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
# hogImage = hogImage.astype("uint8")

gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

cv2.imshow("HOG Image", gx)
cv2.waitKey(0)
cv2.destroyAllWindows()