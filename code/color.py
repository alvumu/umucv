#!/usr/bin/env python

# ./color.py --dev=dir:../images/naranjas/naranjas2.png

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText

#Definimos la ventana en la que mostraremos los frames 
cv.namedWindow("input")
cv.moveWindow('input', 0, 0)


def nada(v):
     pass

cv.createTrackbar("umbral", "input", 0, 255, nada)
cv.createTrackbar("area object detected", "input", 0, 100, nada)

#Variable global que indicará el numero de clicks que se han dado
contadorClicks = 0
#Array que almacenará las coordenadas del ratón
coordenadas = []

#Esta funcion obtendrá el las coordenadas del raton cuando se haga click en la imagen
def on_mouse(event, x, y, flags, params):
    global contadorClicks
    if event == cv.EVENT_LBUTTONDOWN  :
            if contadorClicks >= 0 & contadorClicks <= 2 : 
                color = frame[y, x]
                coordenadas.append((y,x))
                contadorClicks = contadorClicks+1

#Funcion que indica si es el contorno es superior al area que indica el trackbar
def razonable(c):
    area= cv.getTrackbarPos("area object detected",'input')
    return cv.arcLength(c, closed=True) >= area
        
cv.setMouseCallback("input", on_mouse)
for key, frame in autoStream(): 
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    elif contadorClicks == 2 :
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        #Calculamos las medias de los colores HSV con las distintas coordenadas en las que hemos trackeado el raton
        h1 = int(np.mean([hsv[coordenadas[0]][0],hsv[coordenadas[1]][0]]))
        h2 = int(np.mean([hsv[coordenadas[0]][1],hsv[coordenadas[1]][1]]))
        h3 = int(np.mean([hsv[coordenadas[0]][2],hsv[coordenadas[1]][2]]))

        #Obtenemos el valor de la tolerancia 
        tolerance= cv.getTrackbarPos('umbral','input')

        #Estipulamos unos rangos minimos y maximos para definir la mascara 
        lower_range = np.array([h1-tolerance,h2-tolerance,h3-tolerance])
        upper_range = np.array([h1+tolerance,h2+tolerance,h3+tolerance])

        #Obtenemos la máscara resultante 
        mask = cv.inRange(hsv,lower_range, upper_range)

        #Eliminamos pequeños puntos de ruido en la mascara
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones([3,3], np.uint8) )

        #Aplicamos la mascara al frame que se está capturando
        frameMasked = frame.copy()
        frameMasked[mask==0]=(0,0,0)
        cv.imshow("frameMasked",frameMasked)

        #Obtenemos los contornos para poder conocer el numero de objetos que se perciben en la imagen 
        (contours,_) = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        #Almacenamos los contornos que cumplen la condicion de la funcion razonable
        ok = [c for c in contours if razonable(c) ]

        #Dibujamos los contornos 
        cv.drawContours(frame, ok, -1, (0,255,0), cv.FILLED)
        
        #Indicamos en el frame el numero de objetos 
        putText(frame, f'Objects count: {len(ok)}')

    cv.imshow('input',frame)

cv.destroyAllWindows()


