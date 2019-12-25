import plate_detection
import plate_segmentation
import argparse
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True)
parser.add_argument("-vi", "--video")
args = vars(parser.parse_args())
def video_process(video):
    cap = cv2.VideoCapture(args["video"])
    i = 0
    while (cap.isOpened()):
        ret,frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(os.path.join(r"C:\Users\yaiba\UIT\Introduction_CV\plate_recognition_uit\Data",'frame'+ str(i)+'.jpg'),frame)
        i+=1
    cap.release()
    cv2.destroyAllWindows()

def main(image):

    img = cv2.imread(image)
    plate = plate_detection.pre_processing(img)

    final = plate_segmentation.segmen(plate)
    
    cv2.namedWindow("Plate number",cv2.WINDOW_NORMAL)
    cv2.imshow("Plate number", final)
    cv2.waitKey()

if __name__== "__main__":
    main(args["image"])    