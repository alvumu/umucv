#!/usr/bin/env python

import cv2   as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import ROI, putText , Video

from collections import deque

def bgr2gray(x):
    return cv.cvtColor(x,cv.COLOR_BGR2GRAY).astype(float)/255


#Ventana para mostrar la zona escogida
cv.namedWindow("input")
cv.moveWindow('input',0,0)
region = ROI("input") #Para controlar las regiones

video = Video(fps=30)
video.ON = True
tracks = []
trozo = None
threshold_area = 500

#Para quitar el fondo
bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)

#Para generar modelo de fondo
NUM_FRAMES = 15 #Frames en medio segundo
d = deque(maxlen=NUM_FRAMES)
modeloFondo = None
moving = False

#Para suavizado
auto = (0,0) # tamaño de la máscara automático, dependiendo de sigma
sigma = 3

for key,frame in autoStream():
    
    moving = False
    
    #Para el ROI
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        if key == ord('c'):
            trozo = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)
            #Cambiamos a escala de grises
            trozo = cv.cvtColor(trozo, cv.COLOR_BGR2GRAY)
            #Realizamos suavizado
            trozo = cv.GaussianBlur(trozo, auto,sigma)
        if key == ord('x'):
            region.roi = []

        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))
        
        #Para la deteccion de movimiento
    if trozo is not None:
        roiActual = frame[y1:y2+1,x1:x2+1]
        roiActual = cv.cvtColor(roiActual, cv.COLOR_BGR2GRAY)
        roiActual = cv.GaussianBlur(roiActual, auto,sigma)
        
        frame_diff = cv.absdiff(roiActual, trozo)
        cv.imshow("diff", frame_diff)
        threshold=cv.threshold(frame_diff,35,255, cv.THRESH_BINARY)[1]
        cv.imshow('threshold',threshold)
        contours, h =cv.findContours(threshold,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #Sacamos los contornos para comprobar que el area supera la marcada
        for contour in contours:
            area = cv.contourArea(contour)
            if area > threshold_area:
                if moving is False:
                    moving = True
                #Para el primer método utilizando las funciones de backsub
                fgmask = bgsub.apply(frame_diff,learningRate=0)  #Aplicamos extraccion de fondo
                masked = frame[y1:y2+1,x1:x2+1].copy()
                masked[fgmask==0] = 0
                cv.imshow('object BACKSUB', masked)
                video.write(masked)
            #Para el segundo método utilizando un propio modelo de fondo
                if len(d) == NUM_FRAMES:
                    modeloFondo = np.mean(d, axis=0).astype('uint8')
                if modeloFondo is not None:
                    modelFGmask = cv.absdiff(roiActual, modeloFondo)
                    modelFGfgmask = cv.threshold(modelFGmask, 25, 255, cv.THRESH_BINARY)[1]
                    maskedModel = frame[y1:y2+1,x1:x2+1].copy()
                    maskedModel[modelFGfgmask == 0] = 0
                    cv.imshow('object MODEL FG', maskedModel)
            else:
                continue
        
        
        print("MOVIMIENTO ->" + str(moving))
        if not moving:
            print("AÑADIMOS FRAME YA QUE NO HAY MOVIMIENTO")
            d.appendleft(roiActual)

    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)

cv.destroyAllWindows()
video.release()