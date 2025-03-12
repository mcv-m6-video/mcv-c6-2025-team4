from __future__ import absolute_import, division, print_function
import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import cv2


def read_flow_gt(flow_file):
    flow_raw = cv2.imread(flow_file, cv2.IMREAD_UNCHANGED).astype(np.double)

    u = (flow_raw[:, :, 2] - 2**15) / 64.0  # Componente u del flujo
    v = (flow_raw[:, :, 1] - 2**15) / 64.0  # Componente v del flujo
    valid = flow_raw[:, :, 0] == 1  # Máscara de validez (0: inválido, 1: válido)

    # Establecer valores de flujo inválidos a 0
    u[valid == 0] = 0
    v[valid == 0] = 0

    return np.stack((u, v, valid), axis=2)  # El flujo de referencia tiene la forma (h, w, 3)


def compute_msen_pepn(flow_pred, flow_gt, tau=3):
    # Calcular el error cuadrático entre el flujo predicho y el flujo de referencia
    square_error_matrix = (flow_pred[:, :, 0] - flow_gt[:, :, 0]) ** 2 + (flow_pred[:, :, 1] - flow_gt[:, :, 1]) ** 2
    
    # Máscara de validez (1 si es válido, 0 si es inválido)
    valid_mask = flow_gt[:, :, 2]  # Ya tiene la forma (h, w)
    
    # Aplicar la máscara de validez al error cuadrático
    square_error_matrix_valid = square_error_matrix * valid_mask

    # Contar píxeles válidos (no ocluidos)
    non_occluded_pixels = np.sum(valid_mask != 0)

    # Calcular el error por píxel (error euclidiano)
    pixel_error_matrix = np.sqrt(square_error_matrix_valid)

    # Calcular MSEN (Mean Square Error Norm)
    msen = np.sum(pixel_error_matrix) / non_occluded_pixels

    # Calcular PEPN (Percentage of Erroneous Pixels) y expresarlo en porcentaje
    erroneous_pixels = np.sum(pixel_error_matrix > tau)
    pepn = (erroneous_pixels / non_occluded_pixels) * 100  # Convertir a porcentaje

    return msen, pepn




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
    raise ValueError("Error: Images not loaded")

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
colType = 0  # 0 para RGB, 1 para escala de grises

# Calcular flujo óptico
s = time.time()
u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth,
                                      nOuterFPIterations, nInnerFPIterations,
                                      nSORIterations, colType)
e = time.time()

print(f'Time Taken: {e - s:.2f} seconds for image of size {im1.shape}')

# Guardar el flujo óptico
flow = np.concatenate((u[..., None], v[..., None]), axis=2)
np.save('examples/outFlow_new.npy', flow)

flow_pred = np.dstack((u, v))
flow_gt = read_flow_gt('examples/000045_10_gt.png')
msen, pepn = compute_msen_pepn(flow_pred, flow_gt)

# Imprimir resultados
print(f"MSEN: {msen:.4f}")
print(f"PEPN: {pepn:.4f}%")

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
