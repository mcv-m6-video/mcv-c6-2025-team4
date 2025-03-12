from __future__ import absolute_import, division, print_function
import numpy as np
from PIL import Image
import time
import argparse
import cv2


def read_flow_gt(flow_file):
    flow_raw = cv2.imread(flow_file, cv2.IMREAD_UNCHANGED).astype(np.double)

    u = (flow_raw[:, :, 2] - 2**15) / 64.0  # Componente u del flujo
    v = (flow_raw[:, :, 1] - 2**15) / 64.0  # Componente v del flujo
    valid = flow_raw[:, :, 0] == 1  # Máscara de validez (0: inválido, 1: válido)

    u[valid == 0] = 0
    v[valid == 0] = 0

    return np.stack((u, v, valid), axis=2)  # (h, w, 3)


def compute_msen_pepn(flow_pred, flow_gt, tau=3):
    square_error_matrix = (flow_pred[:, :, 0] - flow_gt[:, :, 0]) ** 2 + \
                          (flow_pred[:, :, 1] - flow_gt[:, :, 1]) ** 2

    valid_mask = flow_gt[:, :, 2]
    square_error_matrix_valid = square_error_matrix * valid_mask
    non_occluded_pixels = np.sum(valid_mask != 0)
    pixel_error_matrix = np.sqrt(square_error_matrix_valid)

    msen = np.sum(pixel_error_matrix) / non_occluded_pixels
    erroneous_pixels = np.sum(pixel_error_matrix > tau)
    pepn = (erroneous_pixels / non_occluded_pixels) * 100

    return msen, pepn


# Argument parser
parser = argparse.ArgumentParser(description='Farneback Optical Flow Computation')
parser.add_argument('-viz', dest='viz', action='store_true', help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()

# Cargar imágenes
img1_path = 'examples/000045_10.png'
img2_path = 'examples/000045_11.png'

im1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if im1 is None or im2 is None:
    raise ValueError("Error: Images not loaded")

# Parámetros de Farneback
pyr_scale = 0.5     # Escala de la pirámide (reduce la resolución en cada nivel)
levels = 3          # Número de niveles en la pirámide
winsize = 15        # Tamaño de la ventana para calcular el flujo
iterations = 3      # Número de iteraciones por nivel
poly_n = 5          # Tamaño del polinomio usado
poly_sigma = 1.2    # Sigma del suavizado de la derivada
flags = 0           # Usa 0 para el método normal

# Calcular flujo óptico de Farneback
s = time.time()
flow = cv2.calcOpticalFlowFarneback(im1, im2, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
e = time.time()

print(f'Time Taken: {e - s:.2f} seconds for image of size {im1.shape}')

# Guardar el flujo óptico
np.save('examples/outFlow_farneback.npy', flow)

flow_pred = flow  # (h, w, 2)
flow_gt = read_flow_gt('examples/000045_10_gt.png')
msen, pepn = compute_msen_pepn(flow_pred, flow_gt)

# Imprimir resultados
print(f"MSEN: {msen:.4f}")
print(f"PEPN: {pepn:.4f}%")

# Visualizar resultados si se activa el flag `-viz`
if args.viz:
    hsv = np.zeros((im1.shape[0], im1.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255  # Saturación máxima

    # Convertir flujo en magnitud y ángulo
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convertir HSV a BGR y guardar imágenes
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('examples/outFlow_farneback.png', rgb)

    # Mostrar resultado
    cv2.imshow("Optical Flow (Farneback)", rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
