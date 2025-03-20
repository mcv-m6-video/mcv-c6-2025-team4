import os

# Set environment variable to allow expandable segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys

sys.path.append('core')

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from src.sort import Sort  # Ya tienes implementado el algoritmo SORT
import torchvision.ops as ops
import json
from tqdm import tqdm

# -------------------------------
# FlowFormer Imports and Setup
from FlowFormer.configs.things_eval import get_cfg as get_things_cfg
from FlowFormer.core.FlowFormer import build_flowformer
from FlowFormer.core.utils.utils import InputPadder

# Device selection for optical flow
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device for optical flow:", device)

# -------------------------------
# Initialize FlowFormer model (optical flow) on CUDA with FP16
FLOWFORMER_MODEL_PATH = '/home/toukapy/Documentos/Master CV/C6/mcv-c6-2025-team4/Week3/sintel.pth'
cfg = get_things_cfg()
cfg.model = FLOWFORMER_MODEL_PATH
flow_model = torch.nn.DataParallel(build_flowformer(cfg))
flow_model.load_state_dict(torch.load(cfg.model, map_location=device))
flow_model = flow_model.to(device)

flow_model.eval()


# -------------------------------
# JSON I/O functions for detections and MOT format
def load_detections_from_json(file_path):
    try:
        with open(file_path, "r") as f:
            detections = json.load(f)
        return {str(k): np.array(v) for k, v in detections.items()}  # Mantener claves como str
    except FileNotFoundError:
        return {}


def save_detections_to_json(detections, file_path):
    with open(file_path, "w") as f:
        json.dump({k: [box.tolist() for box in v] for k, v in detections.items()}, f)


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


# -------------------------------
# Object Detection using Faster R-CNN on CPU
def detect_objects(frame, model, framework='torch'):
    original_height, original_width = frame.shape[:2]
    if framework == 'torch':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # Note: we do not move input to GPU since detection runs on CPU
        input_frame = transform(frame).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            predictions = model(input_frame)
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        # Filtrar detecciones de coches (suponiendo que la clase 1 es el coche)
        car_indices = labels == 1
        boxes = boxes[car_indices]
        scores = scores[car_indices]
        # Aplicar NMS
        keep = ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.15)
        boxes = boxes[keep.numpy()]
        scores = scores[keep.numpy()]
        # Escalar cajas a dimensiones originales
        scale_x = original_width / 800
        scale_y = original_height / 800
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        return boxes
    else:
        raise ValueError(f"Unsupported framework: {framework}")


# -------------------------------
# Load Faster R-CNN detection model on CPU
def load_model(model_path, framework='torch'):
    if framework == 'torch':
        model = models.detection.fasterrcnn_resnet50_fpn(weights=None)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        return model
    else:
        raise ValueError(f"Unsupported framework: {framework}")


# -------------------------------
# Compute optical flow using FlowFormer (runs on CUDA, FP16)
def compute_optical_flow_lowres(pred_frame, next_frame, model, device, target_size=(640, 360)):
    """
    Resize the input frames to a fixed target resolution (e.g., 640x360),
    compute optical flow using FlowFormer (in float32 on CUDA),
    and return the flow in the target resolution coordinate space.
    """
    # Convert BGR frames to RGB and normalize to [0,1] in float32
    pred_rgb = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    next_rgb = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Resize images to target resolution
    target_w, target_h = target_size
    pred_resized = cv2.resize(pred_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    next_resized = cv2.resize(next_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Convert resized images to torch tensors with shape (1, 3, H, W)
    pred_tensor = torch.from_numpy(pred_resized.transpose(2, 0, 1)).unsqueeze(0).to(device)
    next_tensor = torch.from_numpy(next_resized.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Pad the tensors (model requirement)
    padder = InputPadder(pred_tensor.shape)
    pred_tensor, next_tensor = padder.pad(pred_tensor, next_tensor)

    # Run inference with FlowFormer in default float32 precision
    with torch.no_grad():
        flow_output = model(pred_tensor, next_tensor)

    # Unpad the flow output and convert to numpy array (target resolution flow)
    flow_tensor = padder.unpad(flow_output[0]).cpu()
    flow_lowres = flow_tensor[0].numpy().transpose(1, 2, 0)

    torch.cuda.empty_cache()
    return flow_lowres





# -------------------------------
# Predict positions for bounding boxes using the computed flow
def predict_positions_with_of(flow, boxes):
    predictions = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # Calcular el centro de la caja
        cx, cy = np.round([(x1 + x2) / 2, (y1 + y2) / 2]).astype(int)
        h, w = flow.shape[:2]
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)
        # Desplazamiento basado en el flujo en el centro de la caja
        dx, dy = flow[cy, cx]
        new_x1 = x1 + dx
        new_y1 = y1 + dy
        new_x2 = x2 + dx
        new_y2 = y2 + dy
        predictions.append([new_x1, new_y1, new_x2, new_y2])
    return np.array(predictions)

def get_color(track_id):
    # If track_id is not in the dictionary, assign a random color
    if track_id not in track_colors:
        # Generate a random BGR color (values 0-255)
        color = np.random.randint(0, 256, size=3).tolist()
        track_colors[track_id] = color
    return track_colors[track_id]


# -------------------------------
# Main video processing configuration
video_path = "./data/AICity_data/train/S03/c010/vdo.avi"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inicializar escritor de video
output_path = 'task2_sort_of2_bad_flow_verybad.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Cargar el modelo de detección de objetos (Faster R-CNN) on CPU
det_model_path = '/home/toukapy/Documentos/Master CV/C6/mcv-c6-2025-team4/faster_rcnn_resnet50_05_og.pth'
det_model = load_model(det_model_path, 'torch')

# Inicializar el algoritmo SORT (ya implementado con Kalman Filter)
mot_tracker = Sort()

frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_rate = 1
selected_frames = range(0, frame_total, sample_rate)

# Cargar o inicializar las detecciones
detections_path = "./detections_new.json"
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

track_colors = {}

# Procesar los fotogramas
for frame_idx in tqdm(selected_frames, desc="Processing video"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    # Assume original frame size is obtained from the video
    original_h, original_w = frame.shape[:2]
    target_w, target_h = (640, 360)  # target resolution used in optical flow

    # Compute scaling factors (from full resolution to target resolution)
    scale_down_x = target_w / original_w
    scale_down_y = target_h / original_h
    scale_up_x = original_w / target_w
    scale_up_y = original_h / target_h

    # Detectar objetos en el frame
    if str(frame_idx) not in saved_detections:
        detections_path = "./detections_new.json"
        detections = detect_objects(frame, det_model, 'torch')
        saved_detections[str(frame_idx)] = detections
        save_detections_to_json(saved_detections, detections_path)

    else:
        detections = saved_detections[str(frame_idx)]

    # For frame_idx > 0, compute low-resolution optical flow
    if frame_idx > 0:
        flow_lowres = compute_optical_flow_lowres(prev_frame, frame, flow_model, device,
                                                  target_size=(target_w, target_h))

        # Scale detection boxes from full resolution to target resolution
        detections_lowres = []
        for box in detections:
            x1, y1, x2, y2 = box
            detections_lowres.append([x1 * scale_down_x, y1 * scale_down_y, x2 * scale_down_x, y2 * scale_down_y])
        detections_lowres = np.array(detections_lowres)

        # Use the low-resolution flow to predict new positions on the target scale
        predicted_boxes_lowres = predict_positions_with_of(flow_lowres, detections_lowres)

        # Scale the predicted boxes back to full resolution
        predicted_boxes = []
        for box in predicted_boxes_lowres:
            new_box = [box[0] * scale_up_x, box[1] * scale_up_y, box[2] * scale_up_x, box[3] * scale_up_y]
            predicted_boxes.append(new_box)
        predicted_boxes = np.array(predicted_boxes)

        # Proceed with SORT update using the predicted_boxes in full resolution
        tracked_objects = mot_tracker.update(np.array([np.append(box, 1.0) for box in predicted_boxes]))
        frame_tracking = {}
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)
            frame_tracking[track_id] = ([x1, y1, x2, y2], frame_idx)
            # Get unique color for this track_id
            color = get_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        tracked_dict[frame_idx] = frame_tracking

    # Guardar el frame actual para el siguiente ciclo
    prev_frame = frame
    out.write(frame)
    torch.cuda.empty_cache()

# Guardar resultados en formato MOTChallenge
output_mot_path = "good_005_with.txt"
save_mot_format(tracked_dict, output_mot_path)

cap.release()
out.release()


