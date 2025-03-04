import cv2
import numpy as np
import tensorflow as tf
import torch
from torchvision import models, transforms
from src import metrics
from torchvision.models.detection import (
    retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights,
    ssd300_vgg16, SSD300_VGG16_Weights,
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
)
import torchvision.ops as ops 
from tqdm import tqdm
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
next_id = 0

# Función para guardar detecciones en JSON (conversión de ndarray a lista)
def save_detections_to_json(detections, file_path):
    with open(file_path, "w") as f:
        json.dump({k: [box.tolist() for box in v] for k, v in detections.items()}, f)

# Función para cargar detecciones desde JSON
def load_detections_from_json(file_path):
    try:
        with open(file_path, "r") as f:
            detections = json.load(f)
        return {int(k): np.array(v) for k, v in detections.items()}
    except FileNotFoundError:
        return None

# Función para hacer el seguimiento de objetos entre dos cuadros
def track_objects(boxes, frame_number):
    global next_id
    updated_objects = {}
    for obj_id, (prev_box, last_seen) in tracked_objects.items():
        if frame_number - last_seen > 10:  # Si el objeto desapareció por 5 frames, descartarlo
            continue
        
        best_iou = 0
        best_match = None
        for i, box in enumerate(boxes):
            current_iou = metrics.compute_iou(prev_box, box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_match = (i, box)
        
        if best_iou > 0.5:  # Umbral para considerar el mismo objeto
            updated_objects[obj_id] = (best_match[1], frame_number)
            del boxes[best_match[0]]
    
    # Asignar nuevos IDs a objetos nuevos
    for box in boxes:
        updated_objects[next_id] = (box, frame_number)
        next_id += 1
    
    return updated_objects

# Función para realizar la detección de objetos dependiendo del modelo cargado
def detect_objects(frame, model, framework='tensorflow'):
    """
    Detecta objetos usando el modelo proporcionado.
    
    Parameters:
    - frame: El cuadro del video.
    - model: El modelo preentrenado para la detección de objetos.
    - framework: 'tensorflow', 'torch', o 'opencv'.
    
    Returns:
    - boxes: Lista de cajas delimitadoras detectadas.
    """

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

        # print(f"Labels detectados: {set(labels)}")

        # Filtrar solo coches (si la clase 1 es la de coches)
        car_indices = labels == 1
        boxes = boxes[car_indices]
        scores = scores[car_indices]

        # Aplicar NMS para reducir solapamientos
        keep = ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.15)
        boxes = boxes[keep.numpy()]
        scores = scores[keep.numpy()]

        # Filtrar detecciones demasiado cercanas
        filtered_boxes = []
        min_dist = 20  # Distancia mínima entre los centros de las cajas

        for box in boxes:
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            if all(np.linalg.norm(np.array([center_x, center_y]) - np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])) > min_dist for b in filtered_boxes):
                filtered_boxes.append(box)

        filtered_boxes = np.array(filtered_boxes)

        # **Ajustar las coordenadas de las cajas al tamaño original de la imagen**
        original_height, original_width = frame.shape[:2]
        scale_x = original_width / 800
        scale_y = original_height / 800

        if len(filtered_boxes) > 0:
            filtered_boxes[:, [0, 2]] *= scale_x
            filtered_boxes[:, [1, 3]] *= scale_y

        return filtered_boxes

    elif framework == 'opencv':
        # Preprocesar la imagen para el modelo Mask R-CNN de OpenCV
        blob = cv2.dnn.blobFromImage(frame, 1.0, (800, 800), (0, 0, 0), swapRB=True, crop=False)
        model.setInput(blob)
        output = model.forward(["detection_out_final", "detection_masks"])

        # Extraer las cajas delimitadoras
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
        raise ValueError(f"Framework '{framework}' no soportado.")

# Función para cargar un modelo preentrenado basado en el framework
def load_model(model_path, framework='tensorflow'):
    if framework == 'tensorflow':
        model = tf.saved_model.load(model_path)
        return model
    elif framework == 'torch':
        model = fasterrcnn_resnet50_fpn(weights=None)  # No cargar pesos preentrenados de COCO
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Poner en modo evaluación
        return model
    elif framework == 'opencv':
        net = cv2.dnn.readNetFromTensorflow(model_path)
        return net
    else:
        raise ValueError(f"Framework '{framework}' no soportado.")

# Función para generar un color aleatorio basado en el ID de seguimiento
def generate_color(track_id):
    np.random.seed(track_id)  # Semilla basada en track_id para obtener colores consistentes
    color = np.random.randint(0, 255, 3).tolist()  # Generar un color aleatorio (BGR)
    return tuple(color)

# Video de entrada
video_path = "./data/AICity_data/train/S03/c010/vdo.avi"  # Ruta al video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Video could not be opened")
    exit()

# Obtener propiedades del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crear el objeto VideoWriter para guardar el video
output_path = './output_videos/task2_1b.mp4'  # Ruta de salida para el video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para guardar el video en formato .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Cargar el modelo que se va a utilizar
model_path = './Week2/0_fold_fine_tuned_faster_rcnn_05.pth'  # Ruta al modelo
framework = 'torch'  # Cambiar a 'torch' o 'opencv' según el modelo que uses
model = load_model(model_path, framework)

tracked_objects = {}
frame_count = 0

frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_rate = 4
selected_frames = range(0, frame_total, sample_rate)

detections_path = "./Week2/detections.json"
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
    
    # Detectar objetos en el cuadro N+1
    if str(frame_idx) not in saved_detections:
        boxes = detect_objects(frame, model, framework)
        saved_detections[str(frame_idx)] = boxes
    else:
        boxes = saved_detections[str(frame_idx)]
        
    if frame_idx % 50 == 0:
        save_detections_to_json(saved_detections, detections_path)
    
    # Realizar el seguimiento de objetos
    tracked_objects = track_objects(list(boxes), frame_idx)
    
    # Dibujar las cajas de los objetos y los IDs en el cuadro
    for obj_id, (box, _) in tracked_objects.items():
        x1, y1, x2, y2 = map(int, box)
        
        # Generar un color único para cada track_id
        color = generate_color(obj_id)
        
        # Dibujar el rectángulo con el color generado
        if x1 is None or y1 is None or x2 is None or y2 is None:
            print(f"Error: Not valid values ({x1}, {y1}, {x2}, {y2})")
        else:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Colocar el texto con el track_id encima del rectángulo
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Escribir el cuadro procesado en el archivo de salida
    out.write(frame)
    
    # Actualizar el cuadro anterior para la siguiente iteración
    frame_count += 1

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
