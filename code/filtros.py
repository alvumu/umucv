#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import ROI
cv.namedWindow("input")

def nada (v):
    pass
# Definir nombres y rangos de las trackbars para cada filtro
region = ROI("input")
roi = None
for key, frame in autoStream():
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        if key == ord('c'):
            roi = frame[y1:y2+1, x1:x2+1]
        if roi is not None:
            if key == ord('g'):
               mode = 1
               while mode == 1 :
                roi = cv.GaussianBlur(roi,(25,25),0)
                frame[y1:y2+1, x1:x2+1] = roi
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)

    cv.imshow('input',frame)

cv.destroyAllWindows()