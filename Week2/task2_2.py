import cv2
import numpy as np
import tensorflow as tf
import torch
from torchvision import models, transforms
from sort import Sort
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights,retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights,ssd300_vgg16, SSD300_VGG16_Weights,ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from tqdm import tqdm
import json 
import torchvision.ops as ops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate unique colors for each ID
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

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# Object detection function (supports TensorFlow, PyTorch, and OpenCV models)
def detect_objects(frame, model, framework='tensorflow'):
    original_height, original_width = frame.shape[:2]

    if framework == 'torch':
        # Preprocesar la imagen para el modelo de PyTorch
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_frame = transform(frame).unsqueeze(0)  # Añadir batch dimension
        model.eval()  # Poner el modelo en modo evaluación

        # Realizar la predicción
        with torch.no_grad():
            predictions = model(input_frame)
        
        # Obtener las cajas delimitadoras de las predicciones
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        car_indices = labels == 1  
        boxes = boxes[car_indices]
        scores = scores[car_indices]
        
        keep = ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.3)
        boxes = boxes[keep.numpy()]

        # Ajustar las coordenadas de las cajas al tamaño original de la imagen
        original_height, original_width = frame.shape[:2]
        scale_x = original_width / 800
        scale_y = original_height / 800

        # Ajustar las cajas a las dimensiones originales
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        return boxes

    else:
        raise ValueError(f"Unsupported framework: {framework}")

import torch

# def load_model(model_path, framework='torch'):
#     if framework == 'torch':
#         # IMPORTANTE: Define aquí tu modelo con la misma arquitectura usada para guardarlo
#         from my_model_definition import MyModel  # Reemplaza con tu clase de modelo

#         model = MyModel()  # Crea una instancia de la arquitectura
#         model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#         model.eval()  # Poner en modo evaluación
#         return model

#     else:
#         raise ValueError(f"Unsupported framework: {framework}")

# Load model
def load_model(model_path, framework='tensorflow'):
    if framework == 'tensorflow':
        return tf.saved_model.load(model_path)
    elif framework == 'torch':
        model = fasterrcnn_resnet50_fpn(weights=None)  # No cargar pesos preentrenados de COCO
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Poner en modo evaluación
        return model
    elif framework == 'opencv':
        return cv2.dnn.readNetFromTensorflow(model_path)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

# Video processing
video_path = "./data/AICity_data/train/S03/c010/vdo.avi"  # Ruta al video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crear el objeto VideoWriter para guardar el video
output_path = './output_videos/task2_2.mp4'  # Ruta de salida para el video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para guardar el video en formato .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Cargar el modelo que se va a utilizar
model_path = './Week2/0_fold_fine_tuned_faster_rcnn_05.pth'  # Ruta al modelo
framework = 'torch'  # Cambiar a 'torch' o 'opencv' según el modelo que uses
model = load_model(model_path, framework)
# weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
# model = retinanet_resnet50_fpn_v2(weights=weights, box_score_thresh=0.85)
# model.to(device)
# next(model.parameters()).device
# model.eval()
mot_tracker = Sort()

frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_rate = 4
selected_frames = range(0, frame_total, sample_rate)

detections_path = "./detections.json"
saved_detections = load_detections_from_json(detections_path)
if saved_detections:
    print("Using saved detections")
else:
    print("Generating new detections")
    saved_detections = {}

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

    # Convertir cajas a formato SORT (x1, y1, x2, y2, score)
    if len(detections) > 0:
        dets = np.array([np.append(box, 1.0) for box in detections if len(box) == 4])
    else:
        dets = np.empty((0, 5))

    # Actualizar el tracker con las detecciones
    tracked_objects = mot_tracker.update(dets)

    # Dibujar cajas y IDs
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj.astype(int)
        color = generate_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)

cap.release()
out.release()
