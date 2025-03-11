import cv2
import numpy as np
import torch
from torchvision import models, transforms
from src.sort import Sort  # Ya tienes implementado el algoritmo SORT
import torchvision.ops as ops
import json
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_detections_from_json(file_path):
    try:
        with open(file_path, "r") as f:
            detections = json.load(f)
        return {str(k): np.array(v) for k, v in detections.items()}  # 🔴 Mantener claves como str
    except FileNotFoundError:
        return {}



# Function to save detections in JSON format
def save_detections_to_json(detections, file_path):
    with open(file_path, "w") as f:
        json.dump({k: [box.tolist() for box in v] for k, v in detections.items()}, f)
        
# Función para guardar los resultados en formato MOTChallenge
def save_mot_format(tracked_objects, output_mot_path):
    output_dir = os.path.dirname(output_mot_path)
    if output_dir:
        os.makedirs(os.path.dirname(output_mot_path), exist_ok=True)
    with open(output_mot_path, "w") as f:
        for frame_idx, objects in tracked_objects.items():
            if isinstance(objects, dict):
                for obj_id, (box, _) in objects.items():
                    x1, y1, x2, y2 = map(int, box)
                    width, height = x2 - x1, y2 - y1
                    f.write(f"{frame_idx}, {obj_id}, {x1}, {y1}, {width}, {height}, 1, 1, 1\n")
            elif isinstance(objects, tuple) and len(objects) == 2:
                box, _ = objects
                x1, y1, x2, y2 = map(int, box)
                width, height = x2 - x1, y2 - y1
                obj_id = frame_idx  # Si no se tiene ID, usar el frame como ID
                f.write(f"{frame_idx}, {obj_id}, {x1}, {y1}, {width}, {height}, 1, 1, 1\n")
            else:
                print(f"Warning: Unhandled data structure at frame {frame_idx}: {objects}")

# Función para detectar objetos en un frame usando el modelo de Faster R-CNN
def detect_objects(frame, model, framework='torch'):
    original_height, original_width = frame.shape[:2]
    if framework == 'torch':
        transform = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize((800, 800)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        input_frame = transform(frame).unsqueeze(0)
        model.eval()

        with torch.no_grad():
            predictions = model(input_frame)

        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        # Filtrar solo detecciones de coches (suponiendo que la clase 1 es el coche)
        car_indices = labels == 1
        boxes = boxes[car_indices]
        scores = scores[car_indices]

        # Aplicar NMS (Non-Maximum Suppression)
        keep = ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.15)
        boxes = boxes[keep.numpy()]
        scores = scores[keep.numpy()]

        # Volver a escalar las cajas a las dimensiones originales
        scale_x = original_width / 800
        scale_y = original_height / 800
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        return boxes
    else:
        raise ValueError(f"Unsupported framework: {framework}")

# Cargar el modelo pre-entrenado Faster R-CNN
def load_model(model_path, framework='torch'):
    if framework == 'torch':
        model = models.detection.fasterrcnn_resnet50_fpn(weights=None)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
        return model
    else:
        raise ValueError(f"Unsupported framework: {framework}")

# Función para calcular el flujo óptico entre dos frames consecutivos
def compute_optical_flow(prev_frame, next_frame):
    # Convertir a escala de grises para calcular el flujo
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Calcular el flujo óptico usando el algoritmo de Farneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

# Función para predecir la posición de los objetos en el siguiente fotograma utilizando Optical Flow
def predict_positions_with_of(flow, boxes):
    predictions = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # Calcular el centro de la caja y asegurarse de que sean enteros
        cx, cy = np.round([(x1 + x2) / 2, (y1 + y2) / 2]).astype(int)

        # Verificar que cx y cy estén dentro de los límites de la imagen
        h, w = flow.shape[:2]
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)

        # Predecir la nueva posición basándonos en el flujo óptico
        dx, dy = flow[cy, cx]

        # Aplicar el desplazamiento en la predicción de la caja
        new_x1 = x1 + dx
        new_y1 = y1 + dy
        new_x2 = x2 + dx
        new_y2 = y2 + dy

        predictions.append([new_x1, new_y1, new_x2, new_y2])
    
    return np.array(predictions)


# Configuración de video
video_path = "./data/AICity_data/train/S03/c010/vdo.avi"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inicializar escritor de video
output_path = 'task2_sort_of2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Cargar el modelo de detección de objetos
model_path = 'Week3/0_fold_fine_tuned_faster_rcnn_05.pth'
model = load_model(model_path, 'torch')

# Inicializar el algoritmo SORT (que ya tiene Kalman Filter)
mot_tracker = Sort()

# Total de fotogramas
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_rate = 1
selected_frames = range(0, frame_total, sample_rate)

# Cargar o inicializar las detecciones
detections_path = "./Week3/detections.json"
saved_detections = load_detections_from_json(detections_path)
if saved_detections:
    print("Using saved detections")
else:
    print("Generating new detections")
    detections_path = "./detections_new.json"
    saved_detections = {}

# Diccionario para almacenar los resultados del seguimiento
tracked_dict = {}

# Leer el primer fotograma
ret, prev_frame = cap.read()

# Procesar los fotogramas
for frame_idx in tqdm(selected_frames, desc="Processing video"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar objetos en el frame
    if str(frame_idx) not in saved_detections:
        detections_path = "./detections_new.json"
        detections = detect_objects(frame, model, 'torch')
        saved_detections[str(frame_idx)] = detections
        save_detections_to_json(saved_detections, detections_path)

    else:
        detections = saved_detections[str(frame_idx)]

    # Si no es el primer frame, calcular el flujo óptico y predecir las posiciones
    if frame_idx > 0:
        flow = compute_optical_flow(prev_frame, frame)
        predicted_boxes = predict_positions_with_of(flow, detections)

        # Realizar el seguimiento de objetos con SORT utilizando las posiciones predichas por OF
        tracked_objects = mot_tracker.update(np.array([np.append(box, 1.0) for box in predicted_boxes]))

        # Almacenar los resultados del seguimiento
        frame_tracking = {}
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)
            frame_tracking[track_id] = ([x1, y1, x2, y2], frame_idx)

            # Dibujar las cajas de seguimiento y agregar el ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Guardar los resultados del seguimiento
        tracked_dict[frame_idx] = frame_tracking

    # Guardar el frame actual como anterior para el siguiente ciclo
    prev_frame = frame

    # Escribir el frame procesado en el video de salida
    out.write(frame)

# Guardar los resultados en formato MOTChallenge
output_mot_path = "output_mot_format_of2.txt"
save_mot_format(tracked_dict, output_mot_path)

# Liberar recursos
cap.release()
out.release()
