#!/usr/bin/env python


# > ./filtros.py --dev=help
import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import ROI,putText
from umucv.util import Help
from math import sqrt
import time

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
apply_color = False
apply_separability = False
apply_cascading = False


def nada(v):
    pass

help = Help(
"""
HELP WINDOW DEMO


g: gaussian
c: color
b: monocromo
v: box
m: median
l: min
k: max

r: only roi
x: del roi

o: cascading
p: separability

SPC: pausa

h: show/hide help
""")

# Definir nombres y rangos de las trackbars para cada filtro
region = ROI("input")
roi = None
frame = None 
isRoi = False


    

def mytrackbar(param,window,X,a,b):
    def fun(v):
        X[0] = v*(b-a)/200 + a
    cv.createTrackbar(param, window, int((X[0]-a)*200/(b-a)), b, fun)


cv.namedWindow("trackbars")

SIGMA=[3]
SIGMACASC = [3]
B=[50]
SIZE = [15]
KSIZE = [1]

mytrackbar("Sigma","trackbars",SIGMA,1,100)
mytrackbar("Sigma cascading","trackbars",SIGMACASC,1,100)
mytrackbar("Kernel","trackbars",KSIZE,1,101)

def apply_filter(frame,roi=None):
    global SIZE,SIGMA,B,SIGMACASC
    SIGMA = cv.getTrackbarPos("Sigma", "trackbars")
    SIGMACASC = cv.getTrackbarPos("Sigma cascading", "trackbars")
    KSIZE = cv.getTrackbarPos("Kernel","trackbars")
    KSIZE= KSIZE*2+1
    # Si no se define la ROI, aplicar el filtro en toda la imagen
    if roi is not None:

        if apply_gaussian and apply_cascading:
            cv.namedWindow("Cascading")
            roiCasc = roi
            sigma_EQ = sqrt(pow(SIGMA,2)+pow(SIGMACASC,2))

            roi = cv.GaussianBlur(roi,(KSIZE,KSIZE),SIGMA)            
            roi = cv.GaussianBlur(roi,(KSIZE,KSIZE),SIGMACASC) 
            roiCasc = cv.GaussianBlur(roiCasc,(KSIZE,KSIZE),sigma_EQ)    
            dif = cv.absdiff(roi,roiCasc)
            mean = np.mean(dif) / 255
            putText(frame, f'Gaussian filter + Cascading',orig=(5,30),div = 0)
            putText(roiCasc, f'Difference : {mean}')
            
            cv.imshow("Cascading",roiCasc)

        if apply_gaussian and apply_separability:
            roi = cv.GaussianBlur(roi,(KSIZE,KSIZE),SIGMA)
            # Descomponemos el filtro gaussiano 2D en dos filtros 1D separables
            gaussian_kernel_x = cv.getGaussianKernel((KSIZE,KSIZE)[0], SIGMA)
            gaussian_kernel_y = cv.getGaussianKernel((KSIZE,KSIZE)[1], SIGMA)
            roi = cv.sepFilter2D(roi, -1, gaussian_kernel_x, gaussian_kernel_y)
            putText(frame, f'Gaussian filter + Separability',orig=(5,30),div = 0) 


        if apply_gaussian and not apply_separability:
            roi = cv.GaussianBlur(roi,(KSIZE,KSIZE),SIGMA)
            putText(frame, f'Gaussian filter',orig=(5,30),div = 0)  

        if apply_byn:
            frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            roi = frame[y1:y2+1, x1:x2+1]
            roi = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)
            putText(frame, f'Black and white',orig=(5,50),div = 0)  
        if not apply_byn:
            roi = roi
            
        if apply_box:
            roi = cv.boxFilter(roi,-1,(KSIZE,KSIZE))
            putText(frame, f'Box filter',orig=(5,70),div = 0) 

        if apply_median:
            roi = cv.medianBlur(roi,KSIZE)
            putText(frame, f'Median filter',orig=(5,90),div = 0) 
        if apply_min:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (KSIZE,KSIZE))
            roi = cv.erode(roi, kernel)
            putText(frame, f'Min filter',orig=(5,110),div = 0)
        if apply_max:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (KSIZE,KSIZE))
            roi = cv.dilate(roi, kernel)
            putText(frame, f'Max filter',orig=(5,130),div = 0) 

        if apply_anyfilter:
            roi = roi 
        
        return roi 
    else :
        if apply_gaussian and apply_cascading:
            
            cv.namedWindow("Cascading")
            frameCasc = frame
            sigma_EQ = sqrt(pow(SIGMA,2)+pow(SIGMACASC,2))

            frame = cv.GaussianBlur(frame,(KSIZE,KSIZE),SIGMA)            
            frame = cv.GaussianBlur(frame,(KSIZE,KSIZE),SIGMACASC) 

            frameCasc = cv.GaussianBlur(frameCasc,(KSIZE,KSIZE),sigma_EQ)  
            dif = cv.absdiff(frame,frameCasc)
            mean = np.mean(dif) / 255
            putText(frame, f'Gaussian filter + Cascading',orig=(5,30),div = 0)  
            putText(frameCasc, f'Difference : {mean}',orig=(5,30),div = 0)         
            cv.imshow("Cascading",frameCasc)
            
        if apply_gaussian and apply_separability:
            frame = cv.GaussianBlur(frame,(KSIZE,KSIZE),SIGMA)
            frame = cv.GaussianBlur(frame,(KSIZE,KSIZE),SIGMA)
            # Descomponemos el filtro gaussiano 2D en dos filtros 1D separables
            gaussian_kernel_x = cv.getGaussianKernel((KSIZE,KSIZE)[0], SIGMA)
            gaussian_kernel_y = cv.getGaussianKernel((KSIZE,KSIZE)[1], SIGMA)
            frame = cv.sepFilter2D(frame, -1, gaussian_kernel_x, gaussian_kernel_y)
            putText(frame, f'Gaussian filter + Separability',orig=(5,30),div = 0) 

        if apply_gaussian and not apply_separability:
            frame = cv.GaussianBlur(frame,(KSIZE,KSIZE),SIGMA)
            putText(frame, f'Gaussian filter',orig=(5,30),div = 0)  
        if apply_byn:
            frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            putText(frame, f'Black and white',orig=(5,50),div = 0)  
        if apply_box:
            frame = cv.boxFilter(frame,-1,(KSIZE,KSIZE))
            putText(frame, f'Box filter',orig=(5,70),div = 0) 
        if apply_median:
            frame = cv.medianBlur(frame,KSIZE)
            putText(frame, f'Median filter',orig=(5,90),div = 0) 
        if apply_min:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (KSIZE,KSIZE))
            frame = cv.erode(frame, kernel)
            putText(frame, f'Min filter',orig=(5,110),div = 0) 
        if apply_max:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (KSIZE,KSIZE))
            frame = cv.dilate(frame, kernel)
            putText(frame, f'Max filter',orig=(5,130),div = 0) 
        if apply_anyfilter:
           frame = frame 
        return frame

for key, frame in autoStream():
    if cv.waitKey(1) & 0xFF == ord('q'):
        break     
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        roi = frame[y1:y2+1, x1:x2+1]
        if roi is not None:
            if key == ord('g'):
                apply_gaussian = not apply_gaussian #Invertir el estado actual del filtro
            if key == ord('b'):
                apply_byn = not apply_byn  
            if key == ord('c'):
                apply_byn = False
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
            if key == ord('o'):
                if apply_cascading :
                    cv.destroyWindow("Cascading")
                apply_cascading = not apply_cascading
            if key == ord('p'):
                apply_separability = not apply_separability
            if key == ord('r'):
                apply_onlyroi = not apply_onlyroi
            cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2) 
            if key == ord('x'):
                region.roi=[]
                apply_onlyroi = False
    else : 
        if key == ord('g'):
            apply_gaussian = not apply_gaussian #Invertir el estado actual del filtro
        if key == ord('b'):
            apply_byn = not apply_byn  
        if key == ord('c'):
            apply_byn = False
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
        if key == ord('o'):
                if apply_cascading :
                    cv.destroyWindow("Cascading")
                apply_cascading = not apply_cascading
        if key == ord('p'):
            apply_separability = not apply_separability

    help.show_if(key, ord('h'))
    if apply_onlyroi and region.roi:
            roi = apply_filter(frame,roi)
            frame[y1:y2+1, x1:x2+1] = roi
            putText(frame, f'Only Roi',orig=(5,150),div = 0) 

    if not apply_onlyroi:
            frame = apply_filter(frame)                   

    print(apply_cascading)         
    cv.imshow('input',frame)

cv.destroyAllWindows()