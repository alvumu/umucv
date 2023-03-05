#!/usr/bin/env python

# ejemplo de selección de ROI

import numpy as np
import cv2 as cv

from umucv.util import ROI, putText
from umucv.stream import autoStream


cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

region = ROI("input")
firstframe = None
trozo = None


for key, frame in autoStream():
   
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        if key == ord('c'):
            trozo = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)
            trozo = cv.cvtColor(trozo, cv.COLOR_BGR2GRAY)
            trozo = cv.GaussianBlur(trozo,(25,25),0)
            if firstframe is None:
                firstframe = trozo
        if key == ord('x'):
            region.roi = []   
    
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

                   
    if trozo is not None:
        nextframe = frame[y1:y2+1, x1:x2+1]
        bgframe = cv.cvtColor(nextframe, cv.COLOR_BGR2GRAY)
        bgframe = cv.GaussianBlur(bgframe,(25,25),0)
        if firstframe is None : 
            firstframe = trozo
            continue
        dif = cv.absdiff(firstframe, bgframe)
        threshold = cv.threshold(dif,35,255,cv.THRESH_BINARY)[1]
        cv.imshow("threshold", threshold)


    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)

