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
MAX_FRAME = 5
d = deque(maxlen=MAX_FRAME)
backgroundModel = None
bgframe = None
frameCounts = 0

for key, frame in autoStream():
    movement = False
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        if trozo is None : 
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
            if key == ord('x'):
                region.roi = []   
        
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))      
    dif = None          
    if trozo is not None : 
        nextframe = frame[y1:y2+1, x1:x2+1] 
        bgframe = cv.cvtColor(nextframe, cv.COLOR_BGR2GRAY)
        bgframe = cv.GaussianBlur(bgframe,(25,25),0) 
        if backSub is None :        
            if (len(d) == MAX_FRAME):
                backgroundModel = np.mean(d,axis=0).astype("uint8")
            if backgroundModel is not None:
                dif = cv.absdiff(bgframe,backgroundModel)
        if backSub is not None : 
                dif = cv.absdiff(bgframe,trozo)
        threshold = cv.threshold(dif,35,255,cv.THRESH_BINARY)[1]
        contours,_ = cv.findContours(threshold, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        for contour in contours : 
            contourArea = cv.contourArea(contour)
            if contourArea > pixActive :  
                if not movement :
                    movement = True
                    video.ON = True
                if backSub is not None :                
                    fgMask = backSub.apply(nextframe)
                else :
                    fgMask = threshold
                copyframe = nextframe.copy()
                copyframe[fgMask== 0] = 0
                if frameCounts <= 15 * 3:
                    video.write(copyframe)
                frameCounts += 1
        if not movement :
            frameCounts = 0
            d.appendleft(bgframe)
                
    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)

