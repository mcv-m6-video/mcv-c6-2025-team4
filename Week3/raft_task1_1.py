import torch
import numpy as np
import cv2
import argparse
import time
from PIL import Image
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

# Cargar modelo RAFT preentrenado
def load_raft_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft-sintel.pth', help="Ruta al modelo preentrenado")
    parser.add_argument('--small', action='store_true', help="Usar RAFT-Small (versión ligera)")
    parser.add_argument('--mixed_precision', action='store_true', help="Usar precisión mixta para acelerar")
    parser.add_argument('--alternate_corr', action='store_true', help="Usar correlación alterna")

    args = parser.parse_args([])  # Esto evita que tome argumentos de línea de comandos

    model = torch.nn.DataParallel(RAFT(args))
    device = torch.device("cpu")  # Usar CPU en vez de CUDA
    model.load_state_dict(torch.load(args.model, map_location=device)) 

    model = model.module
    model.to(device)
    model.eval()
    
    return model

def load_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path)  # Lee imagen 1
    img2 = cv2.imread(img2_path)  # Lee imagen 2

    if img1 is None or img2 is None:
        raise ValueError("Error: No se pudieron cargar las imágenes. Verifica las rutas.")

    # Redimensionar imágenes a múltiplos de 64
    height = (img1.shape[0] // 64) * 64
    width = (img1.shape[1] // 64) * 64
    img1 = cv2.resize(img1, (width, height))
    img2 = cv2.resize(img2, (width, height))

    # Convertir imágenes a float32 y normalizar
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    # Asegurar que tengan 3 canales (Si es en escala de grises, convertir a RGB)
    if img1.ndim == 2:  
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    # Convertir a tensores de PyTorch y mover a CPU
    device = torch.device("cpu")  # Usar CPU
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0).to(device)

    return img1, img2



# Ejecutar RAFT para calcular flujo óptico con medición de tiempo
def compute_optical_flow(model, img1, img2):
    start_time = time.time()  # Iniciar cronómetro

    with torch.no_grad():
        flow_low, flow_up = model(img1, img2, iters=20, test_mode=True)

    end_time = time.time()  # Terminar cronómetro
    execution_time = end_time - start_time  # Calcular tiempo de ejecución
    return flow_up, execution_time

# Calcular métricas MSEN y PEPN
def compute_msen_pepn(flow_pred, flow_gt, tau=3):
    flow_pred = flow_pred.squeeze().permute(1, 2, 0).cpu().numpy()

    # Redimensionar flow_gt al tamaño de flow_pred
    flow_gt_resized = cv2.resize(flow_gt, (flow_pred.shape[1], flow_pred.shape[0]))

    # Calcular el error cuadrático
    square_error_matrix = (flow_pred[:, :, 0:1] - flow_gt_resized[:, :, 0:1]) ** 2 + \
                          (flow_pred[:, :, 1:2] - flow_gt_resized[:, :, 1:2]) ** 2

    # Aplicar máscara de validez del GT
    valid_mask = flow_gt_resized[:, :, 2] != 0
    square_error_matrix_valid = square_error_matrix[valid_mask]

    # Contar píxeles válidos
    non_occluded_pixels = np.sum(valid_mask)

    # Calcular MSEN
    pixel_error_matrix = np.sqrt(square_error_matrix_valid)
    msen = np.mean(pixel_error_matrix)

    # Calcular PEPN
    erroneous_pixels = np.sum(pixel_error_matrix > tau)
    pepn = (erroneous_pixels / non_occluded_pixels) * 100  # Convertir a porcentaje

    return msen, pepn


# Cargar ground truth del flujo óptico
def read_flow_gt(flow_file):
    flow_raw = cv2.imread(flow_file, cv2.IMREAD_UNCHANGED).astype(np.double)
    
    u = (flow_raw[:, :, 2] - 2**15) / 64.0
    v = (flow_raw[:, :, 1] - 2**15) / 64.0
    valid = flow_raw[:, :, 0] == 1

    u[valid == 0] = 0
    v[valid == 0] = 0

    return np.stack((u, v, valid), axis=2)

# Visualizar flujo óptico
def visualize_flow(flow):
    flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
    flow_img = flow_viz.flow_to_image(flow_np)
    cv2.imwrite("optical_flow_raft.png", flow_img)
    cv2.imshow("Optical Flow", flow_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to RAFT model", default="models/raft-sintel.pth")
    parser.add_argument("--img1", help="Path to first image", default="examples/000045_10.png")
    parser.add_argument("--img2", help="Path to second image", default="examples/000045_11.png")
    parser.add_argument("--gt", help="Path to ground truth flow file", default="examples/000045_10_gt.png")
    args = parser.parse_args()

    # Cargar modelo RAFT
    model = load_raft_model()

    # Cargar imágenes y ground truth
    img1, img2 = load_images(args.img1, args.img2)
    flow_gt = read_flow_gt(args.gt)
    print("Tamaño img1:", img1.shape)
    print("Tamaño img2:", img2.shape)

    # Calcular flujo óptico y tiempo de ejecución
    flow_pred, execution_time = compute_optical_flow(model, img1, img2)

    # Calcular métricas
    msen, pepn = compute_msen_pepn(flow_pred, flow_gt)
    
    # Imprimir resultados
    print(f"MSEN (RAFT): {msen:.4f}")
    print(f"PEPN (RAFT): {pepn:.2f}%")
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")

    # Visualizar flujo óptico
    visualize_flow(flow_pred)
