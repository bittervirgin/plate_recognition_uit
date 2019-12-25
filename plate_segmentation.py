import cv2
import numpy as np
from matplotlib import pyplot as plt

def segmen(new_img):
    roi_gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
    roi_blur = cv2.GaussianBlur(roi_gray,(3,3),1)
    ret,thre = cv2.threshold(roi_blur,120,255,cv2.THRESH_BINARY_INV)
    kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    thre_mor = cv2.morphologyEx(thre,cv2.MORPH_DILATE,kerel3)
    cont,hier = cv2.findContours(thre_mor,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    areas_ind = {}
    areas = []
    for ind,cnt in enumerate(cont) :
        area = cv2.contourArea(cnt)
        areas_ind[area] = ind
        areas.append(area)
    areas = sorted(areas,reverse=True)[2:12]
    for i in areas:
        (x,y,w,h) = cv2.boundingRect(cont[areas_ind[i]])
        cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,255,0),2)
    final_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB)
    return final_img