import cv2
import os
import numpy as np
import threading
from src import read_data, metrics

# -------------------------
# 1. Configuración inicial
# -------------------------
video_paths = [
    "./output_videos/MOG2.avi",
    "./output_videos/BGS.avi",
    "./output_videos/LSBP.avi",
    "./output_videos/LOBSTER.avi",
    "./output_videos/MOG.avi"
]

path_annotation = "./data/ai_challenge_s03_c010-full_annotation.xml"

# Cargar anotaciones de ground truth
gt_data, _ = read_data.parse_annotations_xml(path_annotation, isGT=True)
gt_dict = {}
for item in gt_data:
    frame_no = item["frame"]
    box = item["bbox"][0].tolist() if isinstance(item["bbox"], np.ndarray) else item["bbox"]
    gt_dict.setdefault(frame_no, []).append(box)

# Diferentes valores de min_area a probar
min_area_values = [500, 750, 1000, 1500, 2000]

# Almacenar resultados
results = {video: {} for video in video_paths}


# -------------------------
# 2. Función para procesar un video
# -------------------------
def process_video(video_path):
    global results

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Procesando {video_path} - FPS: {fps}, Tamaño: {frame_width}x{frame_height}")

    for min_area in min_area_values:
        print(f"  -> Evaluando min_area = {min_area}...")

        # Inicializar listas de predicciones y GT
        all_pred_boxes = []
        all_gt_boxes = []

        frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Fin del video

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gt_boxes = gt_dict.get(frame_idx, [])

            # Extraer bounding boxes con la configuración actual
            pred_boxes = metrics.extract_bounding_boxes(gray_frame, min_area=min_area)

            all_pred_boxes.append(pred_boxes)
            all_gt_boxes.append(gt_boxes)
            frame_idx += 1

        # Calcular mAP@50 para este min_area
        final_map50 = metrics.compute_video_average_precision(all_pred_boxes, all_gt_boxes, iou_threshold=0.5)

        # Guardar resultado en el diccionario
        results[video_path][min_area] = final_map50

    cap.release()


# -------------------------
# 3. Procesar los 5 videos en paralelo
# -------------------------
threads = []
for video in video_paths:
    thread = threading.Thread(target=process_video, args=(video,))
    threads.append(thread)
    thread.start()

# Esperar a que todos los hilos terminen
for thread in threads:
    thread.join()

# -------------------------
# 4. Imprimir resultados finales
# -------------------------
print("\nResultados Finales (mAP@50):")
for video, min_area_results in results.items():
    print(f"\n📌 Video: {video}")
    for min_area, score in min_area_results.items():
        print(f"  - min_area {min_area}: mAP@50 = {score:.4f}")

print("\n✅ Procesamiento completado!")
