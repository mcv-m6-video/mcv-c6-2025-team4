import cv2
import os
import numpy as np
from src import load_data, read_data, metrics

# Directorio de salida
output_dir = "./output_videos"
os.makedirs(output_dir, exist_ok=True)

# Video path
video_path = "./data/AICity_data/train/S03/c010/vdo.avi"
path_annotation = "./data/ai_challenge_s03_c010-full_annotation.xml"

# Load total frames
total_frames = load_data.get_total_frames(video_path)
training_end = int(total_frames * 0.25)

# Load ground truth annotations
gt_data, _ = read_data.parse_annotations_xml(path_annotation, isGT=True)
gt_dict = {}
for item in gt_data:
    frame_no = item["frame"]
    box = item["bbox"][0].tolist() if isinstance(item["bbox"], np.ndarray) else item["bbox"]
    gt_dict.setdefault(frame_no, []).append(box)

# Inicializar los métodos de background subtraction
mog = cv2.bgsegm.createBackgroundSubtractorMOG()
mog2 = cv2.createBackgroundSubtractorMOG2()
lsbp = cv2.bgsegm.createBackgroundSubtractorLSBP()

# Abrir el video con OpenCV VideoCapture
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, training_end)  # Empezar en la zona de test

# Obtener propiedades del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

# Inicializar VideoWriters para cada método
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec AVI

out_mog = cv2.VideoWriter(os.path.join(output_dir, "MOG.avi"), fourcc, fps, frame_size, isColor=False)
out_mog2 = cv2.VideoWriter(os.path.join(output_dir, "MOG2.avi"), fourcc, fps, frame_size, isColor=False)
out_lsbp = cv2.VideoWriter(os.path.join(output_dir, "LSBP.avi"), fourcc, fps, frame_size, isColor=False)

print("Processing frames and calculating mAP@50...")

# Inicializar los acumuladores de predicciones y GT para cada método
all_pred_boxes = {"MOG": [], "MOG2": [], "LSBP": []}
all_gt_boxes = {"MOG": [], "MOG2": [], "LSBP": []}

frame_idx = training_end  # Comenzamos desde la zona de test
while cap.isOpened():
    ret, frame_rgb = cap.read()
    if not ret:
        break  # Si el video terminó, salir del loop

    # Convertir a escala de grises para mejorar la compresión
    gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

    # Aplicar los métodos de background subtraction
    fg_mask_mog = mog.apply(gray_frame)
    fg_mask_mog2 = mog2.apply(gray_frame)
    fg_mask_lsbp = lsbp.apply(gray_frame)

    # Guardar los resultados de cada frame en el video de salida
    out_mog.write(fg_mask_mog)
    out_mog2.write(fg_mask_mog2)
    out_lsbp.write(fg_mask_lsbp)

    # Obtener las cajas de ground truth para este frame
    gt_boxes = gt_dict.get(frame_idx, [])

    # Acumular las predicciones y las cajas GT por cada método
    for method_name, fg_mask in [
        ("MOG", fg_mask_mog),
        ("MOG2", fg_mask_mog2),
        ("LSBP", fg_mask_lsbp),
    ]:
        # Extraer las cajas de la máscara predicha
        pred_boxes = metrics.extract_bounding_boxes(fg_mask, min_area=500)
        
        # Almacenar las predicciones y las GT para este método
        all_pred_boxes[method_name].append(pred_boxes)
        all_gt_boxes[method_name].append(gt_boxes)

    frame_idx += 1  # Incrementar el índice del frame

# Liberar recursos
cap.release()
out_mog.release()
out_mog2.release()
out_lsbp.release()

# Calcular el mAP para cada método usando la función compute_video_average_precision
final_map50 = {}
for method in ["MOG", "MOG2", "LSBP"]:
    map50 = metrics.compute_video_average_precision(all_pred_boxes[method], all_gt_boxes[method], iou_threshold=0.5)
    final_map50[method] = map50

# Imprimir resultados finales
print("\nFinal mAP@50 Results:")
for method, map50 in final_map50.items():
    print(f"{method}: {map50:.4f}")

print(f"\nProcessing completed! Videos saved in {output_dir}")
