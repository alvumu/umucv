#!/usr/bin/env python

import numpy as np
import cv2
import time



#Función para aplicar una máscara de convolución a una imagen
def convolucion(img, mascara):
    m, n = mascara.shape
    # Rellenamos la imagen para evitar bordes negros
    img_padded = np.pad(img, ((m//2, m//2), (n//2, n//2)), mode='constant')
    # Creamos una matriz de ceros para almacenar la imagen convolucionada
    img_conv = np.zeros_like(img)
    # Recorremos la imagen y aplicamos la máscara en cada píxel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_conv[i, j] = np.sum(img_padded[i:i+m, j:j+n] * mascara)
    return img_conv

# Cargamos una imagen en escala de grises
img = cv2.imread('../images/cube3.png', cv2.IMREAD_GRAYSCALE)

# Definimos una máscara de convolución
ks = 11
kernel = np.ones([ks,ks]) 
kernel = kernel/np.sum(kernel)
mascara = kernel

# Aplicamos la convolución utilizando nuestra función
inicio = time.time()
img_conv = convolucion(img, mascara)
tiempo_convolucion = time.time() - inicio

# Aplicamos la convolución utilizando la función de OpenCV
inicio = time.time()
img_conv_opencv = cv2.filter2D(img, -1, mascara)
tiempo_convolucion_opencv = time.time() - inicio

# Mostramos las imágenes resultantes y los tiempos de convolución
cv2.imshow('Imagen original', img)
cv2.imshow('Imagen convolucionada', img_conv)
cv2.imshow('Imagen convolucionada con OpenCV', img_conv_opencv)
cv2.moveWindow("Imagen convolucionada con OpenCV", 100, 100)
print(f'Tiempo de convolución: {tiempo_convolucion:.5f} segundos')
print(f'Tiempo de convolución con OpenCV: {tiempo_convolucion_opencv:.5f} segundos')
cv2.waitKey(0)
cv2.destroyAllWindows()