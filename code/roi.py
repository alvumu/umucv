#!/usr/bin/env python

# ejemplo de selección de ROI

import numpy as np
import cv2 as cv

from umucv.util import ROI, putText
from umucv.stream import autoStream
from umucv.util import Video
from collections import deque
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filter', type=str, help='Background subtraction method (KNN, MOG2, Manual).', default='MOG2')
args = parser.parse_args()

if args.filter == "1":
    backSub = cv.createBackgroundSubtractorMOG2()
elif args.filter == "2":
    backSub = cv.createBackgroundSubtractorKNN()
else : backSub = None

cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

region = ROI("input")
firstframe = None
trozo = None
video = Video(fps=15)
video.ON = True
d = deque(maxlen=3)

for key, frame in autoStream():
   
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        if key == ord('c'):
            trozo = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)
            trozo = cv.cvtColor(trozo, cv.COLOR_BGR2GRAY)
            trozo = cv.GaussianBlur(trozo,(25,25),0)
            if (x2-x1) == (y2-y1) : 
                area = (x2-x1)*(y2-y1)
            else:
                area = ((x2-x1) * (y2-y1))/2
            pixActive = area * 0.3
            if firstframe is None:
                firstframe = trozo
        if key == ord('x'):
            region.roi = []   
    
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))             
    if trozo is not None and backSub is not None:        
        nextframe = frame[y1:y2+1, x1:x2+1].astype(np.float32)
        bgframe = cv.cvtColor(nextframe, cv.COLOR_BGR2GRAY)
        bgframe = cv.GaussianBlur(bgframe,(25,25),0)
        if firstframe is None : 
            firstframe = trozo
            continue
        dif = cv.absdiff(firstframe, bgframe)
        threshold = cv.threshold(dif,35,255,cv.THRESH_BINARY)[1]
        cv.imshow("threshold", threshold)
        d.appendleft(nextframe)
        contours,_ = cv.findContours(threshold, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
   
        for contour in contours : 
            contourArea = cv.contourArea(contour)
            if contourArea > pixActive :  
                print("Movimiento detectado")
                fgMask = backSub.apply(d[-1])
                video.write(fgMask)
                continue
    elif trozo is not None and backSub is None:
        nextframe = frame[y1:y2+1, x1:x2+1]
        d.appendleft(nextframe)
        bgframe = cv.cvtColor(nextframe, cv.COLOR_BGR2GRAY)
        bgframe = cv.GaussianBlur(bgframe,(25,25),0)
        bgbackground = np.mean(d,axis=0)
        dif = cv.absdiff(nextframe, bgbackground.astype("uint8"))
        threshold = cv.threshold(dif,pixActive,255,cv.THRESH_BINARY)[1]
        copyframe = nextframe.copy()
        copyframe[threshold == 0] = 0
        cv.imshow("threshold", copyframe)
       
        contours,_ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        for contour in contours : 
            contourArea = cv.contourArea(contour)
            if contourArea > pixActive :  
                print("Movimiento detectado")
                video.write(copyframe)
                continue
        
    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)

