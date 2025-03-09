from __future__ import absolute_import, division, print_function
import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import cv2

# Argument parser
parser = argparse.ArgumentParser(description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument('-viz', dest='viz', action='store_true', help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()

# Cargar imágenes
img1_path = 'examples/000045_10.png'
img2_path = 'examples/000045_11.png'

im1 = np.array(Image.open(img1_path))
im2 = np.array(Image.open(img2_path))

# Verificar que las imágenes se cargaron correctamente
if im1 is None or im2 is None:
    raise ValueError("Error: No se pudieron cargar las imágenes. Verifica las rutas.")

# Convertir a tipo float y normalizar
im1 = im1.astype(float) / 255.
im2 = im2.astype(float) / 255.

# Asegurar que las imágenes tengan 3 dimensiones
if im1.ndim == 2:  # Si es escala de grises (H, W), convertir a (H, W, 1)
    im1 = np.expand_dims(im1, axis=2)
    im2 = np.expand_dims(im2, axis=2)

# Parámetros del flujo óptico
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 1  # 0 para RGB, 1 para escala de grises

# Calcular flujo óptico
s = time.time()
u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth,
                                      nOuterFPIterations, nInnerFPIterations,
                                      nSORIterations, colType)
e = time.time()

print(f'Time Taken: {e - s:.2f} seconds for image of size {im1.shape}')

# Guardar el flujo óptico
flow = np.concatenate((u[..., None], v[..., None]), axis=2)
np.save('examples/outFlow.npy', flow)

# Visualizar resultados si se activa el flag `-viz`
if args.viz:
    hsv = np.zeros((im1.shape[0], im1.shape[1], 3), dtype=np.uint8)  # Asegurar 3 canales
    hsv[..., 1] = 255  # Saturación máxima

    # Convertir flujo óptico en magnitud y ángulo
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # Normalizar ángulo
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Normalizar magnitud

    # Convertir HSV a BGR y guardar imágenes
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('examples/outFlow_new.png', rgb)
    cv2.imwrite('examples/car2Warped_new.jpg', (im2W[:, :, ::-1] * 255).astype(np.uint8))

    # Mostrar resultado
    cv2.imshow("Optical Flow", rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
