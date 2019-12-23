import plate_detection
import Segmentation
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True)
parser.add_argument("-vi", "--video")
args = parser.parse_args()

img = cv2.imread(args("image"))
dil_img = plate_detection.pre_processing(img)
plate = plate_detection.find_contour(dil_img)

final = Segmentation.segmen(plate)
cv2.imshow("Plate number", final)