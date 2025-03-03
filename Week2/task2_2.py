import cv2
import numpy as np
import tensorflow as tf
import torch
from torchvision import models, transforms
from sort import Sort
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights,retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights,ssd300_vgg16, SSD300_VGG16_Weights,ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate unique colors for each ID
def generate_color(track_id):
    np.random.seed(track_id)
    return tuple(np.random.randint(0, 255, 3).tolist())

# Object detection function (supports TensorFlow, PyTorch, and OpenCV models)
def detect_objects(frame, model, framework='tensorflow'):
    if framework == 'tensorflow':
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = np.expand_dims(input_frame, axis=0)
        input_frame = tf.image.resize(input_frame, (800, 800))
        detections = model(input_frame)
        boxes = detections['detection_boxes'].numpy()[0]
        return boxes

    elif framework == 'torch':
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((800, 800)), transforms.ToTensor()])
        input_frame = transform(frame).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            predictions = model(input_frame)
        boxes = predictions[0]['boxes'].cpu().numpy()
        return boxes

    elif framework == 'opencv':
        blob = cv2.dnn.blobFromImage(frame, 1.0, (800, 800), (0, 0, 0), swapRB=True, crop=False)
        model.setInput(blob)
        output = model.forward(["detection_out_final", "detection_masks"])
        boxes = []
        for i in range(output[0].shape[2]):
            confidence = output[0][0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(output[0][0, 0, i, 3] * frame.shape[1])
                y1 = int(output[0][0, 0, i, 4] * frame.shape[0])
                x2 = int(output[0][0, 0, i, 5] * frame.shape[1])
                y2 = int(output[0][0, 0, i, 6] * frame.shape[0])
                boxes.append([x1, y1, x2, y2])
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
        model = torch.load(model_path)
        model.eval()
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
# model_path = 'path_to_your_model'  # Ruta al modelo
framework = 'torch'  # Cambiar a 'torch' o 'opencv' según el modelo que uses
# model = load_model(model_path, framework)
weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
model = retinanet_resnet50_fpn_v2(weights=weights, box_score_thresh=0.85)
model.to(device)
next(model.parameters()).device
model.eval()
mot_tracker = Sort()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar objetos en el frame actual
    detections = detect_objects(frame, model, framework)

    # Convertir cajas a formato SORT (x1, y1, x2, y2, score)
    if len(detections) > 0:
        # print("si")
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
