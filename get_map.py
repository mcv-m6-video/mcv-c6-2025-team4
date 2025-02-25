import cv2
import os
import numpy as np
from src import read_data, metrics

# -------------------------
# 1. Configuración inicial
# -------------------------
video_path = "./output_videos/BGS.avi"
path_annotation = "./data/ai_challenge_s03_c010-full_annotation.xml"

# Cargar anotaciones de ground truth
gt_data, _ = read_data.parse_annotations_xml(path_annotation, isGT=True)
gt_dict = {}
for item in gt_data:
    frame_no = item["frame"]
    box = item["bbox"][0].tolist() if isinstance(item["bbox"], np.ndarray) else item["bbox"]
    gt_dict.setdefault(frame_no, []).append(box)

# -------------------------
# 2. Abrir el video de máscaras de LOBSTER
# -------------------------
cap = cv2.VideoCapture(video_path)

# Verificar FPS y tamaño
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

print("Processing frames and calculating mAP@50...")

# Inicializar los acumuladores para las predicciones y las cajas de GT por todo el video
all_pred_boxes = []
all_gt_boxes = []

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Fin del video

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Obtener las cajas de ground truth para el frame actual
    gt_boxes = gt_dict.get(frame_idx, [])

    # Extraer bounding boxes de la máscara de LOBSTER
    pred_boxes = metrics.extract_bounding_boxes(gray_frame, min_area=500)

    # Acumular las predicciones y las cajas GT para todo el video
    all_pred_boxes.append(pred_boxes)
    all_gt_boxes.append(gt_boxes)

    frame_idx += 1

# Liberar recursos
cap.release()

# -------------------------
# 3. Calcular mAP@50 global
# -------------------------
final_map50 = metrics.compute_video_average_precision(all_pred_boxes, all_gt_boxes, iou_threshold=0.5)

# Imprimir resultados
print(f"\nFinal mAP@50: {final_map50:.4f}")
print("\nProcessing completed!")
