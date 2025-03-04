import cv2
import numpy as np
import torch
from torchvision import models, transforms
from Week2.src.sort import Sort
import torchvision.ops as ops
import json
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_mot_format(tracked_objects, output_mot_path):
    os.makedirs(os.path.dirname(output_mot_path), exist_ok=True)
    with open(output_mot_path, "w") as f:
        for frame_idx, objects in tracked_objects.items():
            if isinstance(objects, dict):  
                # Manejo original cuando objects es un diccionario
                for obj_id, (box, _) in objects.items():
                    x1, y1, x2, y2 = map(int, box)
                    width, height = x2 - x1, y2 - y1
                    f.write(f"{frame_idx}, {obj_id}, {x1}, {y1}, {width}, {height}, 1, 1, 1\n")

            elif isinstance(objects, tuple) and len(objects) == 2:
                # Nuevo manejo cuando objects es una tupla con (box, frame_number)
                box, _ = objects  # Extraer la caja
                x1, y1, x2, y2 = map(int, box)
                width, height = x2 - x1, y2 - y1
                obj_id = frame_idx  # Si no hay un ID de objeto, podemos usar el frame_idx como referencia
                f.write(f"{frame_idx}, {obj_id}, {x1}, {y1}, {width}, {height}, 1, 1, 1\n")

            else:
                print(f"Warning: Unhandled data structure at frame {frame_idx}: {objects}")

# Función para generar colores únicos por ID
def generate_color(track_id):
    np.random.seed(track_id)
    return tuple(np.random.randint(0, 255, 3).tolist())

def load_detections_from_json(file_path):
    try:
        with open(file_path, "r") as f:
            detections = json.load(f)
        return {int(k): np.array(v) for k, v in detections.items()}
    except FileNotFoundError:
        return None

def save_detections_to_json(detections, file_path):
    with open(file_path, "w") as f:
        json.dump({k: [box.tolist() for box in v] for k, v in detections.items()}, f)

# Función de detección de objetos
def detect_objects(frame, model, framework='torch'):
    original_height, original_width = frame.shape[:2]

    if framework == 'torch':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_frame = transform(frame).unsqueeze(0)
        model.eval()

        with torch.no_grad():
            predictions = model(input_frame)

        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        car_indices = labels == 1
        boxes = boxes[car_indices]
        scores = scores[car_indices]

        keep = ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.15)
        boxes = boxes[keep.numpy()]
        scores = scores[keep.numpy()]

        original_height, original_width = frame.shape[:2]
        scale_x = original_width / 800
        scale_y = original_height / 800

        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        return boxes

    else:
        raise ValueError(f"Unsupported framework: {framework}")

# Función para cargar el modelo
def load_model(model_path, framework='torch'):
    if framework == 'torch':
        model = models.detection.fasterrcnn_resnet50_fpn(weights=None)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
        return model
    else:
        raise ValueError(f"Unsupported framework: {framework}")

# Configuración del video
video_path = "./data/AICity_data/train/S03/c010/vdo.avi"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = './output_videos/task2_2rcnn.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Cargar el modelo
model_path = 'faster_rcnn_resnet50_05_og.pth'
framework = 'torch'
model = load_model(model_path, framework)

mot_tracker = Sort()
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_rate = 1
selected_frames = range(0, frame_total, sample_rate)

detections_path = "./detections.json"
saved_detections = load_detections_from_json(detections_path)
if saved_detections:
    print("Using saved detections")
else:
    print("Generating new detections")
    saved_detections = {}

tracked_dict = {}

for frame_idx in tqdm(selected_frames, desc="Processing video"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    if str(frame_idx) not in saved_detections:
        detections = detect_objects(frame, model, framework)
        saved_detections[str(frame_idx)] = detections
    else:
        detections = saved_detections[str(frame_idx)]
        
    if frame_idx % 50 == 0:
        save_detections_to_json(saved_detections, detections_path)

    if len(detections) > 0:
        dets = np.array([np.append(box, 1.0) for box in detections if len(box) == 4])
    else:
        dets = np.empty((0, 5))

    tracked_objects = mot_tracker.update(dets)

    # Convertimos tracked_objects en un diccionario con la estructura correcta
    frame_tracking = {}
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj.astype(int)
        frame_tracking[track_id] = ([x1, y1, x2, y2], frame_idx)

        color = generate_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Guardamos la información en tracked_dict
    tracked_dict[frame_idx] = frame_tracking

    out.write(frame)

cap.release()
out.release()

# Guardar en formato MOTChallenge
if not isinstance(tracked_dict, dict):
    print("Error: tracked_objects is not a dictionary!", type(tracked_dict))
else:
    # Guardar formato MOTChallenge
    output_mot_path = "Week2/MOTS-train.txt"
    save_mot_format(tracked_dict, output_mot_path)

