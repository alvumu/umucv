
#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from collections import deque
import numpy as np
from umucv.util import putText


points = deque(maxlen=2)
fov = 0 

# Crear el objeto ArgumentParser
parser = argparse.ArgumentParser(description='FOV')

# Añadir el argumento numérico
parser.add_argument('fov', type=int, help='Introduzca el fov de la camara si lo conoce')

# Analizar los argumentos de la línea de comandos
args = parser.parse_args()

if args.fov : 
    fov = args.fov

def calculateF (fov) : 
    if fov != 0 : 
        rad = radians(fov)
        tangent = math.tan(rad/2)*2
        f = frame.shape[1] / tangent
 

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))

cv.namedWindow("webcam")
cv.setMouseCallback("webcam", fun)

for key, frame in autoStream():
    for p in points:
        cv.circle(frame, p,3,(0,0,255),-1)
    if len(points) == 2:
        cv.line(frame, points[0],points[1],(0,0,255))
        c = np.mean(points, axis=0).astype(int)
        d = np.linalg.norm(np.array(points[1])-points[0])
        if fov != 0:
            ang = math.atan((d/2) / f) * 2 
            ang = round(degrees(ang),1)
            putText(frame,f'{angle} deg',c)
        else : 
            putText(frame,f'{d:.1f} pix',c)
            
    cv.imshow('webcam',frame)
    
cv.destroyAllWindows()
