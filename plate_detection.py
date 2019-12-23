#libraries
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt




def pre_processing(img):
    im_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #turn img to gray scale
    noise_removal = cv2.bilateralFilter(im_gray,9,75,75)
    equal_histogram = cv2.equalizeHist(noise_removal)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=20)
    sub_morp_image = cv2.subtract(equal_histogram,morph_image)
    ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
    canny_image = cv2.Canny(thresh_image,250,255)
    kernel = np.ones((3,3), np.uint8)
    dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
    return dilated_image

def find_contour(dilated_image):
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True) 
        if len(approx) == 4:
            screenCnt = approx
            break
    mask = np.zeros(im_gray.shape,np.uint8)
    final = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    final = cv2.bitwise_and(im,im,mask=mask)
    #plt.imshow(new_image)
    return final

