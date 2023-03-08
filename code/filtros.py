#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import ROI

cv.namedWindow("input")

# Variable global que indica si el filtro gausiano debe aplicarse
apply_gaussian = False
apply_byn = False
apply_box = False
apply_median = False
apply_min = False
apply_max = False
apply_onlyroi = False
apply_anyfilter = False


def nada(v):
    pass

# Definir nombres y rangos de las trackbars para cada filtro
region = ROI("input")
roi = None
frame = None 
isRoi = False
def apply_filter(frame,roi=None):
    # Si no se define la ROI, aplicar el filtro en toda la imagen
    if roi is not None:
        if apply_gaussian:
            roi = cv.GaussianBlur(roi,(25,25),0)
        if apply_byn:
            frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            roi = frame[y1:y2+1, x1:x2+1]
            roi = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)
        if apply_box:
            roi = cv.boxFilter(roi,-1,(50,50))
        if apply_median:
            roi = cv.medianBlur(roi,5)
        if apply_min:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (15,15))
            roi = cv.erode(roi, kernel)
        if apply_max:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (15,15))
            roi = cv.dilate(roi, kernel)
        if apply_anyfilter:
            roi = roi 
        return roi 
    else :
        if apply_gaussian:
            frame = cv.GaussianBlur(frame,(25,25),0)
        if apply_byn:
            frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        if apply_box:
            frame = cv.boxFilter(frame,-1,(50,50))
        if apply_median:
            frame = cv.medianBlur(frame,5)
        if apply_min:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (15,15))
            frame = cv.erode(frame, kernel)
        if apply_max:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (15,15))
            frame = cv.dilate(frame, kernel)
        if apply_anyfilter:
           frame = frame 
        return frame

for key, frame in autoStream():
    if cv.waitKey(1) & 0xFF == ord('q'):
        break     
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        roi = frame[y1:y2+1, x1:x2+1]
        isRoi = True 
    if isRoi :
        if key == ord('g'):
            apply_gaussian = not apply_gaussian #Invertir el estado actual del filtro
        if key == ord('b'):
            apply_byn = not apply_byn  
        if key == ord('n'):
            apply_anyfilter = not apply_anyfilter
        if key == ord('v'):
            apply_box = not apply_box
        if key == ord('m'):
            apply_median = not apply_median
        if key == ord('l'):
            apply_min = not apply_min
        if key == ord('k'):
            apply_max = not apply_max
        if key == ord('r'):
            apply_onlyroi = not apply_onlyroi
        if apply_onlyroi :
            roi = apply_filter(frame,roi)
            frame[y1:y2+1, x1:x2+1] = roi
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2) 
        if key == ord('x'):
            isRoi = False
            region.roi=[]
    else : 
        if key == ord('g'):
            apply_gaussian = not apply_gaussian #Invertir el estado actual del filtro
        if key == ord('b'):
            apply_byn = not apply_byn  
        if key == ord('n'):
            apply_anyfilter = not apply_anyfilter
        if key == ord('v'):
            apply_box = not apply_box
        if key == ord('m'):
            apply_median = not apply_median
        if key == ord('l'):
            apply_min = not apply_min
        if key == ord('k'):
            apply_max = not apply_max
        if not apply_onlyroi :
            frame = apply_filter(frame)                   

                       
    cv.imshow('input',frame)

cv.destroyAllWindows()